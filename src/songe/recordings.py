"""Upkeep of the recorded conversations in experience.sqlite3 (issue #73).

- dedupe: rows recorded before schema v3 keep each request's whole history.
  They are rewritten in place, a few at a time, to share repeated parts
  (parts.py). Each batch is one transaction that also moves a saved cursor,
  so a stopped run loses nothing and the next one carries on where it was.
  Every rewritten row is read back and compared before it replaces the old
  one. Rows not yet rewritten stay plain JSON, which every reader reads.
- compact: give the freed space back to the disk, in the background.
- limit: when the store is over `recordings_limit_gb`, delete the oldest
  recordings that have already been learned from (see `learned_sessions`);
  never one that has not.
"""

import json
import logging
import shutil
import time
import uuid

from . import parts

logger = logging.getLogger(__name__)

# Rows rewritten per transaction. Old rows can hold a whole agent history
# (megabytes each); a small batch keeps the write lock short for the proxy's
# recording thread and memory low in the guarded child.
DEDUPE_BATCH = 16
# Free space below this is left for SQLite to reuse for new recordings:
# rewriting the file to give it back would cost more than it returns.
COMPACT_MIN_BYTES = 64 * 1024 * 1024
# Pages given back per transaction (4 KiB pages: 16 MiB), so a recording
# waiting for the write lock never waits long.
COMPACT_STEP_PAGES = 4096

# A conversation with a request this recent may still be continued by its
# agent (opencode keeps a session open all day). Its recordings are kept even
# when learned from: once deleted, a continuation could no longer be linked
# to it, and would never be learned from (or deleted) itself.
RESUMABLE_SECONDS = 24 * 3600
# Nights in these states depend on the recordings as they are (a Dream under
# way, resumable or waiting to be applied): nothing is deleted meanwhile.
UNDER_WAY = ("running", "interrupted", "ready", "waiting")

_TABLES = (
    ("exchanges", "id", "request", parts.REQUEST_LISTS),
    ("steps", "rowid", "payload", parts.STEP_LISTS),
)


def cursor_key(table):
    return f"parts_cursor_{table}"


def dedupe(store, stop=None):
    """Rewrite older rows to share repeated parts. Returns how many rows
    were rewritten. stop(): checked between batches."""
    rewritten = 0
    for table, key, column, lists in _TABLES:
        while stop is None or not stop():
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                saved = db.execute(
                    "SELECT value FROM state WHERE key=?", (cursor_key(table),)
                ).fetchone()
                cursor = json.loads(saved[0]) if saved else 0
                rows = db.execute(
                    f"SELECT {key} AS k, {column} AS body FROM {table}"
                    f" WHERE {key}>? AND refs IS NULL ORDER BY {key} LIMIT ?",
                    (cursor, DEDUPE_BATCH),
                ).fetchall()
                for row in rows:
                    if row["body"] is None:
                        continue
                    try:
                        value = json.loads(row["body"])
                    except ValueError:
                        # Unreadable before as after: left as it is, and the
                        # cursor moves past it so the rest goes on.
                        logger.warning("dedupe_unreadable %s %s", table, row["k"])
                        continue
                    skeleton, refs = parts.pack(db, value, lists)
                    if refs is None:
                        continue  # nothing large or repeated in it
                    back = parts.unpack(db, skeleton, refs)
                    if json.dumps(back) != json.dumps(value):
                        # Never replace a recording by something that reads
                        # back differently: it stays as it is (its unused
                        # parts are collected later).
                        logger.warning("dedupe_mismatch %s %s", table, row["k"])
                        parts.collect(db, parts.ids_of(refs))
                        continue
                    db.execute(
                        f"UPDATE {table} SET {column}=?, refs=? WHERE {key}=?",
                        (skeleton, refs, row["k"]),
                    )
                    if table == "steps":
                        # The evidence column copied the payload's evidence,
                        # which the payload keeps (log.Store._put_step).
                        db.execute(
                            "UPDATE steps SET evidence='[]' WHERE rowid=?", (row["k"],)
                        )
                    rewritten += 1
                if len(rows) < DEDUPE_BATCH:
                    # Every row up to here is done; newer ones were recorded
                    # sharing their parts already.
                    end = db.execute(f"SELECT COALESCE(MAX({key}),0) FROM {table}").fetchone()[0]
                else:
                    end = rows[-1]["k"]
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES (?,?)",
                    (cursor_key(table), json.dumps(max(cursor, end))),
                )
            if len(rows) < DEDUPE_BATCH:
                break
    if rewritten:
        logger.info("recordings_deduped rows=%s", rewritten)
    return rewritten


def sizes(store):
    """(bytes in use, free bytes inside the file, auto_vacuum mode)."""
    with store.connect() as db:
        page = db.execute("PRAGMA page_size").fetchone()[0]
        count = db.execute("PRAGMA page_count").fetchone()[0]
        free = db.execute("PRAGMA freelist_count").fetchone()[0]
        mode = db.execute("PRAGMA auto_vacuum").fetchone()[0]
    return (count - free) * page, free * page, mode


INCREMENTAL = 2  # PRAGMA auto_vacuum value


def tidy(store, limit=None):
    """The service's upkeep in an idle gap: keep the recordings under
    `limit` bytes (when given), then give freed space back. Skipped while
    the guarded experience child (reconstruction and dedupe) runs: it holds
    experience.lock, and the next idle gap tries again. Returns the bytes
    given back, or None when skipped."""
    from .platform.locking import acquire

    with (store.root / "experience.lock").open("a+") as lease:
        try:
            acquire(lease)
        except BlockingIOError:
            return None
        if limit is not None:
            keep_under(store, limit)
        return compact(store)


def learned_sessions(store, now=None):
    """[(session, exchange ids)] that may be deleted, oldest first.

    "Learned from" (issue #73): Dreams learn from whole sessions (the
    exchanges under one prefix-linking root, reading/timeline.sessions). A
    session has been learned from by a learner when

    1. that learner has read it at its current revision (read_sessions:
       revision >= its newest exchange id in the learner's window), so its
       signals (lessons, feedback, successes) are stored, and
    2. each lesson signal read from it that practice can use (LESSON_TYPES,
       with a lesson and the step it came from; practice.store) is done: trained into the version in use (trained_signals, or
       absorbed_signals: a Deep Dream adopted its practice), or practice was
       written for its cluster and either the lesson is held out of training
       for checks, or no scene was kept (nothing more will come of it).
       Practice scenes, signals and lessons keep their own copy; only
       writing practice needs the recording, and that is done.

    An exchange is learned from when at least one learner's window covers
    it (learning on: level new after its cutoff, or all) and every learner
    whose window covers it has learned from its session. A session may be
    deleted when all its exchanges are learned from, and its newest request
    is older than RESUMABLE_SECONDS. Exchanges in no session (session
    titles, failed or unlinkable requests, embeddings) and sessions of a
    model with learning off are never learned from, so never deleted.
    "Oldest" is the session's newest request: the one used longest ago goes
    first."""
    from .models import registry
    from .practice.pool import EXTRA_HELD_OUT, held_out
    from .practice.store import LESSON_TYPES, cluster_of
    from .reading import timeline
    from .reading.store import signals

    now = time.time() if now is None else now
    with store.connect() as db:
        every = timeline.sessions(db)
        recorded = dict(
            db.execute("SELECT id, COALESCE(recorded_at, created) FROM exchanges").fetchall()
        )
    covered = {}  # exchange id -> learned by every learner covering it so far
    for row in registry.entries(store):
        level = row.get("learning_level", "off")
        if level == "off":
            continue
        learner = store.scoped(row["id"])
        with learner.connect() as db:
            window = timeline.sessions(db, row.get("learning_cutoff") if level == "new" else None)
            read = dict(
                db.execute(
                    "SELECT session, revision FROM read_sessions WHERE model_id=?",
                    (learner.model_id,),
                ).fetchall()
            )
            practised = {
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT signal_id FROM practice WHERE model_id=?",
                    (learner.model_id,),
                )
            }
            kept = {
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT signal_id FROM practice WHERE model_id=? AND kept=1",
                    (learner.model_id,),
                )
            }
        trained = set(learner.get("trained_signals") or []) | set(
            learner.get("absorbed_signals") or []
        )
        extra = set(learner.get(EXTRA_HELD_OUT) or [])
        cluster = cluster_of(learner)
        open_sessions = set()
        for s in signals(learner, types=LESSON_TYPES):
            if not s.get("lesson") or not s.get("exchanges") or s["id"] in trained:
                continue  # nothing to practise from it, or already trained
            members = cluster.get(s["id"], [s["id"]])
            written = any(m in practised for m in members)
            finished = written and (
                held_out(s["id"], extra) or not any(m in kept for m in members)
            )
            if not finished:
                open_sessions.add(s.get("session"))
        for session, ids in window.items():
            learned = read.get(session, -1) >= max(ids) and session not in open_sessions
            for i in ids:
                covered[i] = covered.get(i, True) and learned
    found = []
    for session, ids in every.items():
        newest = max(recorded.get(i, now) for i in ids)
        if newest > now - RESUMABLE_SECONDS:
            continue
        if all(covered.get(i) for i in ids):
            found.append((newest, session, ids))
    return [(session, ids) for _, session, ids in sorted(found)]


def delete_session(store, session, ids):
    """Delete one session's recordings and what was built from them (steps,
    their prefix links and per-step rows), in one transaction, and the parts
    nothing uses any more. Read marks, signals, lessons and practice stay:
    they keep their own copies, and the session is remembered as deleted
    (pruned_sessions) so they are not taken for a vanished session's."""
    marks = ",".join("?" * len(ids))
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        steps = [
            r[0]
            for r in db.execute(
                f"SELECT id FROM steps WHERE json_extract(payload,'$.source_id') IN ({marks})",
                list(ids),
            )
        ]
        freed = [
            i
            for (refs,) in db.execute(f"SELECT refs FROM exchanges WHERE id IN ({marks})", list(ids))
            for i in parts.ids_of(refs)
        ]
        for chunk in parts._chunks(steps):
            step_marks = ",".join("?" * len(chunk))
            freed += [
                i
                for (refs,) in db.execute(
                    f"SELECT refs FROM steps WHERE id IN ({step_marks})", chunk
                )
                for i in parts.ids_of(refs)
            ]
            for table in ("prefixes", "metric_dirty", "step_signals", "scored_revisions", "pool"):
                db.execute(f"DELETE FROM {table} WHERE step_id IN ({step_marks})", chunk)
            db.execute(f"DELETE FROM steps WHERE id IN ({step_marks})", chunk)
        db.execute(f"DELETE FROM reconstructed WHERE exchange_id IN ({marks})", list(ids))
        db.execute(f"DELETE FROM exchanges WHERE id IN ({marks})", list(ids))
        parts.collect(db, freed)
        # What was read from it moves to a name of its own: a new
        # conversation opening the same way gets the same session key, and
        # reading it must not replace these signals (and their practice).
        kept = f"{session}#deleted-{uuid.uuid4().hex[:8]}"
        db.execute("UPDATE signals SET session=? WHERE session=?", (kept, session))
        db.execute("UPDATE read_sessions SET session=? WHERE session=?", (kept, session))
        db.execute(
            "INSERT OR REPLACE INTO pruned_sessions VALUES (?,?)", (kept, time.time())
        )
        db.execute(
            "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
            (json.dumps(str(uuid.uuid4())),),
        )
    return len(ids)


def keep_under(store, limit, now=None):
    """Delete the oldest learned-from sessions until the recordings use at
    most `limit` bytes. Never while a Dream is under way; never a session
    not learned from. Returns how many exchanges were deleted."""
    if sizes(store)[0] <= limit:
        return 0
    from .models import registry

    for row in registry.entries(store):
        if any(n["status"] in UNDER_WAY for n in store.scoped(row["id"]).nights()):
            return 0
    deleted = 0
    for session, ids in learned_sessions(store, now):
        deleted += delete_session(store, session, ids)
        if sizes(store)[0] <= limit:
            break
    if deleted:
        from .wording import recordings_deleted

        logger.info("recordings_limit_deleted exchanges=%s", deleted)
        store.event("record", recordings_deleted(deleted), model_id="")
    return deleted


def over_limit(config, store):
    """(bytes in use, limit) when the recordings are over the limit, else
    None. Read-only: works on a store the CLI only reads."""
    limit = limit_bytes(config)
    try:
        used = sizes(store)[0]
    except Exception:  # noqa: BLE001 - no store yet, or not readable
        return None
    return (used, limit) if used > limit else None


def limit_bytes(config):
    return float(getattr(config, "recordings_limit_gb", 2.0)) * 1e9


def compact(store, minimum=COMPACT_MIN_BYTES):
    """Give free space inside the file back to the disk. A store made
    before Songe could do this in steps is rewritten once (VACUUM, while
    nothing else runs: the caller waits for an idle moment); after that
    space goes back in short steps. Returns the bytes given back."""
    with store.connect() as db:
        # Parts nothing uses any more (normally collected where they were
        # freed); safe in one write transaction: a part and the row using it
        # are always written together.
        db.execute("BEGIN IMMEDIATE")
        parts.collect(db)
    used, free, mode = sizes(store)
    if free < minimum:
        return 0
    before = store.path.stat().st_size
    if mode != INCREMENTAL:
        # VACUUM copies the data in use to a temporary file and back
        # through the write-ahead log: room for two copies.
        if shutil.disk_usage(store.root).free < 2 * used:
            logger.warning("compact_skipped_disk used=%s", used)
            return 0
        with store.connect() as db:
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            db.execute("VACUUM")
    else:
        while True:
            with store.connect() as db:
                db.execute(f"PRAGMA incremental_vacuum({COMPACT_STEP_PAGES})").fetchall()
                left = db.execute("PRAGMA freelist_count").fetchone()[0]
            if not left:
                break
    with store.connect() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    after = store.path.stat().st_size
    logger.info("recordings_compacted before=%s after=%s", before, after)
    return max(0, before - after)
