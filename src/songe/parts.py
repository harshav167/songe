"""Repeated parts of recordings are stored once (issue #73).

Every agent request resends the whole history: the system prompt, the tool
list and each earlier message. A request (`exchanges.request`) and a step
(`steps.payload`, which copies a request's history as its context and the
later messages as its hindsight) are stored as a skeleton plus `refs`:

- a message list (`messages`; a step's `context` and `hindsight`): every
  message but the last is a part, referenced by id. The last one stays in the
  skeleton, so SQL that looks at it (`$.hindsight[#-1].role`, the compaction
  markers in a request's last message) and whether the list is empty still
  read the skeleton correctly.
- any other top-level value of WHOLE_MIN bytes or more (the tool list): one
  part.

`refs` is JSON {key: id | [ids]}. A part is the value's own JSON text
(`json.dumps` with its defaults, as the row was stored before), kept once in
`parts` under its SHA-256. Unpacking puts every value back in its place and
order, so a reader gets exactly the request or step that was recorded; a row
with no refs (recorded before, or with nothing to share) is plain JSON.

`part_uses` counts the rows (exchanges and steps) that reference each part;
triggers keep it, so every writer counts the same way. A part no row uses
any more can be deleted (`collect`).
"""

import hashlib
import json
from contextlib import contextmanager

REQUEST_LISTS = ("messages",)
STEP_LISTS = ("context", "hindsight", "evidence")
# A value smaller than this stays in the row: its reference (an id in `refs`
# and a row in `parts` with a 32-byte hash) would save little or nothing.
# Message list items are parts whatever their size: a history of short
# messages repeated in every request would otherwise still grow as its square.
WHOLE_MIN = 1024
# Ids per IN (...) query; SQLite builds before 3.32 allow 999 parameters.
CHUNK = 500
# Parts one load keeps decoded-ready (a session's messages repeat across its
# requests). Beyond this the cache starts over: a scan of every step must
# not hold the whole store in memory.
CACHE_LIMIT = 20000

TABLES = ("exchanges", "steps")

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS parts ("
    "id INTEGER PRIMARY KEY, hash BLOB NOT NULL UNIQUE, body TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS part_uses (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)",
)

_IDS = "SELECT value FROM json_tree({row}.refs) WHERE type='integer'"


def triggers(table):
    """Keep part_uses counting the rows of `table` that use each part."""
    add = f"UPDATE part_uses SET n=n+1 WHERE id IN ({_IDS.format(row='NEW')});"
    drop = f"UPDATE part_uses SET n=n-1 WHERE id IN ({_IDS.format(row='OLD')});"
    return [
        f"CREATE TRIGGER IF NOT EXISTS {table}_parts_{name} AFTER {event} ON {table}"
        f" WHEN {row}.refs IS NOT NULL BEGIN {body} END"
        for name, event, row, body in (
            ("insert", "INSERT", "NEW", add),
            ("delete", "DELETE", "OLD", drop),
            ("update_old", "UPDATE OF refs", "OLD", drop),
            ("update_new", "UPDATE OF refs", "NEW", add),
        )
    ]


def install(db):
    """Tables, `refs` columns and triggers; part of Store's schema step, in
    its transaction."""
    for statement in SCHEMA:
        db.execute(statement)
    for table in TABLES:
        if not has_refs(db, table):
            db.execute(f"ALTER TABLE {table} ADD COLUMN refs TEXT")
        for statement in triggers(table):
            db.execute(statement)


def has_refs(db, table):
    """Whether `table` has the refs column (a store not yet upgraded, read
    without changing it, has none)."""
    return any(r[1] == "refs" for r in db.execute(f"PRAGMA table_info({table})"))


def _chunks(items):
    items = list(items)
    for i in range(0, len(items), CHUNK):
        yield items[i : i + CHUNK]


def _put(db, texts):
    """Part ids for these texts, adding the new ones. Runs in the caller's
    write transaction, so a part found here cannot be collected before the
    row that uses it is written."""
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")
    hashes = [hashlib.sha256(t.encode()).digest() for t in texts]
    found = {}
    for chunk in _chunks(dict.fromkeys(hashes)):
        found.update(
            (bytes(h), i)
            for h, i in db.execute(
                f"SELECT hash, id FROM parts WHERE hash IN ({','.join('?' * len(chunk))})",
                chunk,
            )
        )
    for h, text in zip(hashes, texts, strict=True):
        if h not in found:
            found[h] = db.execute(
                "INSERT INTO parts(hash, body) VALUES (?,?)", (h, text)
            ).lastrowid
            db.execute("INSERT INTO part_uses(id, n) VALUES (?, 0)", (found[h],))
    return [found[h] for h in hashes]


def pack(db, value, lists):
    """(skeleton JSON text, refs JSON text or None) for a request or step."""
    if not isinstance(value, dict):
        return json.dumps(value), None
    skeleton, refs, texts, where = dict(value), {}, [], []
    for key, item in value.items():
        if key in lists and isinstance(item, list):
            # A list's last item always stays (see the module docstring).
            if len(item) > 1:
                where.append((key, len(texts), len(item) - 1))
                texts += [json.dumps(x) for x in item[:-1]]
                skeleton[key] = item[-1:]
        elif isinstance(item, (dict, list)):
            text = json.dumps(item)
            if len(text) >= WHOLE_MIN:
                where.append((key, len(texts), None))
                texts.append(text)
                skeleton[key] = None
    if not texts:
        return json.dumps(value), None
    ids = _put(db, texts)
    for key, start, count in where:
        refs[key] = ids[start] if count is None else ids[start : start + count]
    return json.dumps(skeleton), json.dumps(refs)


def bodies(db, ids, cache=None):
    """{id: JSON text} for these part ids (cache: shared across one load)."""
    cache = {} if cache is None else cache
    if len(cache) > CACHE_LIMIT:
        cache.clear()  # a long scan never holds every recording in memory
    for chunk in _chunks({i for i in ids if i not in cache}):
        cache.update(
            db.execute(
                f"SELECT id, body FROM parts WHERE id IN ({','.join('?' * len(chunk))})",
                chunk,
            )
        )
    return cache


def unpack(db, text, refs, cache=None):
    """The value `pack` was given, from its skeleton text and refs."""
    if text is None:
        return None
    value = json.loads(text)
    if not refs:
        return value
    refs = json.loads(refs)
    ids = [i for spec in refs.values() for i in (spec if isinstance(spec, list) else [spec])]
    found = bodies(db, ids, cache)
    missing = [i for i in ids if i not in found]
    if missing:
        raise LookupError(f"recording parts missing: {missing[:5]}")
    for key, spec in refs.items():
        if isinstance(spec, list):
            head = json.loads("[" + ", ".join(found[i] for i in spec) + "]")
            value[key] = head + value[key]
        else:
            value[key] = json.loads(found[spec])
    return value


@contextmanager
def snapshot(db):
    """One read transaction, so a row and its parts come from the same
    moment even while old recordings are being deleted."""
    if db.in_transaction:
        yield db
        return
    db.execute("BEGIN")
    try:
        yield db
    finally:
        if db.in_transaction:
            db.commit()


def request(db, row, cache=None):
    """The recorded request of an exchanges row (sqlite3.Row or dict)."""
    return unpack(db, row["request"], _refs(row), cache)


def request_of(db, exchange_id, column="request"):
    """The recorded request of one exchange (None when it is gone);
    column="*": the whole row as a dict with its request unpacked."""
    with snapshot(db):
        row = db.execute("SELECT * FROM exchanges WHERE id=?", (exchange_id,)).fetchone()
        if row is None:
            return None
        value = request(db, row)
    if column == "*":
        return {k: row[k] for k in row.keys() if k != "refs"} | {"request": value}
    return value


def payload(db, row, cache=None):
    """The stored step of a steps row (sqlite3.Row or dict)."""
    return unpack(db, row["payload"], _refs(row), cache)


def _refs(row):
    keys = row.keys()
    return row["refs"] if "refs" in keys else None


def collect(db, ids=None):
    """Delete parts no row uses (only among `ids` when given). Returns how
    many. Run it in the write transaction of the deletes that freed them."""
    if ids is None:
        gone = [r[0] for r in db.execute("SELECT id FROM part_uses WHERE n<=0")]
    else:
        gone = []
        for chunk in _chunks(set(ids)):
            gone += [
                r[0]
                for r in db.execute(
                    "SELECT id FROM part_uses WHERE n<=0 AND id IN "
                    f"({','.join('?' * len(chunk))})",
                    chunk,
                )
            ]
    for chunk in _chunks(gone):
        marks = ",".join("?" * len(chunk))
        db.execute(f"DELETE FROM parts WHERE id IN ({marks})", chunk)
        db.execute(f"DELETE FROM part_uses WHERE id IN ({marks})", chunk)
    return len(gone)


def ids_of(refs):
    """Part ids in one refs value."""
    if not refs:
        return []
    return [
        i
        for spec in json.loads(refs).values()
        for i in (spec if isinstance(spec, list) else [spec])
    ]
