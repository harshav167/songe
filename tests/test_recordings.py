"""Repeated parts of recordings are stored once (issue #73); readers see the
recorded requests and steps exactly as before."""

import json
import random
import sqlite3

import pytest

from songe import parts, recordings
from songe.log import SCHEMA, SCHEMA_V2, Store
from songe.reading import timeline

random.seed(73)
WORDS = "the a file read write test run error build module function class import".split()


def text(n):
    return " ".join(random.choice(WORDS) for _ in range(n // 5))


SYSTEM = {"role": "system", "content": text(52000)}  # about 13k tokens
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": f"tool{i}",
            "description": text(800),
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
    for i in range(20)
]


def looping_session(n, tool_result=1500):
    """An agent looping on one tool call: every request resends the system
    prompt, the tool list and the whole history."""
    messages = [SYSTEM, {"role": "user", "content": "Fix the failing test in parser.py"}]
    out = []
    for i in range(n):
        call = {"id": f"c{i}", "type": "function",
                "function": {"name": "tool1", "arguments": json.dumps({"x": "parser.py"})}}
        action = {"role": "assistant", "content": "", "tool_calls": [call]}
        out.append(({"model": "songe/m", "messages": list(messages), "tools": TOOLS, "stream": True},
                     {"choices": [{"message": action, "finish_reason": "tool_calls"}]}))
        messages += [action, {"role": "tool", "tool_call_id": f"c{i}", "content": text(tool_result)}]
    return out


def used_bytes(store):
    with store.connect() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    return recordings.sizes(store)[0]


ODD = [
    {"model": "m", "messages": [{"role": "user", "content": "é ✓ \u0000 \"quoted\""},
                                {"role": "assistant", "content": None, "tool_calls": []},
                                {"role": "user", "content": [{"type": "text", "text": "parts"}]}],
     "temperature": 0.1, "max_tokens": 10**12, "tools": TOOLS, "stop": None},
    {"model": "m", "messages": [{"role": "user", "content": "one message stays as it is"}]},
    {"model": "m", "prompt": text(5000)},  # legacy completions
    {"input": [text(3000), text(3000)], "model": "embed"},
    {},
    ["not", "an", "object"],
    {"messages": [SYSTEM, SYSTEM, SYSTEM, {"role": "user", "content": "same message three times"}]},
]


def test_recorded_requests_read_back_exactly(store):
    sent = ODD + [request for request, _ in looping_session(5)]
    for request in sent:
        store.record("/v1/chat/completions", request, {"choices": []})
    back = [row["request"] for row in store.exchanges()]
    assert [json.dumps(r) for r in back] == [json.dumps(r) for r in sent]
    assert all("refs" not in row for row in store.exchanges())
    with store.connect() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM exchanges ORDER BY id")]
        assert [parts.request_of(db, i) for i in ids] == back
        assert [row[1] for row in timeline.load(db, ids)] == back
        # Large or repeated parts are shared; the last message stays inline.
        stored = db.execute("SELECT request, refs FROM exchanges WHERE id=?", (ids[0],)).fetchone()
        assert json.loads(stored[0])["messages"] == sent[0]["messages"][-1:]
        assert json.loads(stored[1])["messages"] and json.loads(stored[1])["tools"]


def test_steps_read_back_exactly(store):
    for request, response in looping_session(6):
        store.record("/v1/chat/completions", request, response)
    store.reconstruct()
    steps = store.active_steps()
    assert len(steps) == 6
    with store.connect() as db:
        refs = [r[0] for r in db.execute("SELECT refs FROM steps")]
    assert all(refs)
    # Every step's context is its own request's history, read back whole.
    by_source = {s["source_id"]: s for s in steps}
    for exchange in store.exchanges():
        step = by_source[exchange["id"]]
        assert step["context"] == exchange["request"]["messages"]
        assert step["tools"] == TOOLS
    first = by_source[min(by_source)]
    # The rest of the loop: its tool result, then four more calls and results.
    assert len(first["hindsight"]) == 1 + 2 * 4
    # A step stored again (a later stage) keeps reading back the same.
    store.sync_steps([first])
    assert next(s for s in store.active_steps() if s["id"] == first["id"]) == first


def test_repeated_prompts_are_stored_once(store, tmp_path):
    session = looping_session(120)
    plain = sum(len(json.dumps(request)) for request, _ in session)
    for request, response in session:
        store.record("/v1/chat/completions", request, response)
    store.reconstruct()
    used = used_bytes(store)
    # The plain requests alone are 23 MB (the steps built from them as much
    # again); stored once, the store holds the new content of each exchange,
    # a few times over at most (page slack, the inline last message, ids).
    assert plain > 20e6
    assert used < plain / 10
    new_content = len(json.dumps(SYSTEM)) + len(json.dumps(TOOLS)) + sum(
        len(json.dumps(m)) for m in session[-1][0]["messages"][2:]
    )
    assert used < 10 * new_content


def old_store(root, rows):
    """A store as the previous release left it: schema v2, whole requests
    and steps as plain JSON."""
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "experience.sqlite3")
    db.executescript(SCHEMA)
    db.executescript(SCHEMA_V2)
    db.execute("ALTER TABLE exchanges ADD COLUMN recorded_at REAL")
    for i, (request, response) in enumerate(rows, 1):
        db.execute(
            "INSERT INTO exchanges(id,created,path,request,response,status,complete,binding,recorded_at)"
            " VALUES (?,?,?,?,?,200,1,'',?)",
            (i, float(i), "/v1/chat/completions", json.dumps(request), json.dumps(response), float(i)),
        )
    db.execute("PRAGMA user_version=2")
    db.commit()
    db.close()


def test_an_old_store_is_deduped_in_place(tmp_path):
    root = tmp_path / "old"
    session = looping_session(40)
    old_store(root, session)
    store = Store(root)
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM exchanges WHERE refs IS NULL").fetchone()[0] == 40
    # Not yet rewritten: every reader reads the plain rows.
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]
    store.reconstruct(batch=64)
    with store.connect() as db:  # steps written before this release: plain
        for row in db.execute("SELECT * FROM steps").fetchall():
            db.execute("UPDATE steps SET payload=?, refs=NULL WHERE id=?",
                       (json.dumps(parts.payload(db, row)), row["id"]))
        parts.collect(db)
    steps = sorted(store.active_steps(), key=lambda s: s["id"])
    before = used_bytes(store)

    assert recordings.dedupe(store) == 80
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM exchanges WHERE refs IS NULL").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM steps WHERE refs IS NULL").fetchone()[0] == 0
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]
    assert sorted(store.active_steps(), key=lambda s: s["id"]) == steps
    assert used_bytes(store) < before / 5
    assert recordings.dedupe(store) == 0  # done; a second run finds nothing

    # The space comes back to the disk: a store from before is rewritten
    # once, and gives space back in steps from then on.
    size = store.path.stat().st_size
    assert recordings.compact(store, minimum=0) > 0
    assert store.path.stat().st_size < size / 10
    assert recordings.sizes(store)[2] == recordings.INCREMENTAL
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]


def test_an_interrupted_dedupe_loses_nothing_and_resumes(tmp_path, monkeypatch):
    root = tmp_path / "old"
    session = looping_session(40)
    old_store(root, session)
    store = Store(root)

    # Stopped between batches: the cursor keeps what was done.
    checks = []
    first = recordings.dedupe(store, stop=lambda: checks.append(1) or len(checks) > 1)
    assert first == recordings.DEDUPE_BATCH
    with store.connect() as db:
        assert db.execute("SELECT value FROM state WHERE key='parts_cursor_exchanges'").fetchone()
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]

    # Killed in the middle of a batch: that batch is rolled back whole.
    real, calls = parts.pack, []

    def dying(db, value, lists):
        calls.append(1)
        if len(calls) == 5:
            raise KeyboardInterrupt("killed")
        return real(db, value, lists)

    monkeypatch.setattr(parts, "pack", dying)
    with pytest.raises(KeyboardInterrupt):
        recordings.dedupe(store)
    monkeypatch.setattr(parts, "pack", real)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM exchanges WHERE refs IS NULL").fetchone()[0] == 40 - first
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]

    assert recordings.dedupe(store) == 40 - first
    assert [r["request"] for r in store.exchanges()] == [r for r, _ in session]


def test_parts_no_row_uses_are_collected(store):
    for request, response in looping_session(4):
        store.record("/v1/chat/completions", request, response)
    other = {"messages": [SYSTEM, {"role": "user", "content": "another task"}], "tools": TOOLS}
    store.record("/v1/chat/completions", other, {"choices": []})
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        freed = [i for (r,) in db.execute("SELECT refs FROM exchanges WHERE id<=4") for i in parts.ids_of(r)]
        db.execute("DELETE FROM exchanges WHERE id<=4")
        assert parts.collect(db, freed) > 0
    # The system prompt and tools are still used by the other request.
    assert [r["request"] for r in store.exchanges()] == [other]
    with store.connect() as db:
        assert db.execute("SELECT MIN(n) FROM part_uses").fetchone()[0] >= 1


def test_forget_removes_the_parts(store):
    for request, response in looping_session(3):
        store.record("/v1/chat/completions", request, response)
    store.forget()
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == 0


def test_a_new_store_gives_space_back_in_steps(store):
    assert recordings.sizes(store)[2] == recordings.INCREMENTAL
    for request, response in looping_session(30, tool_result=40000):
        store.record("/v1/chat/completions", request, response)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM exchanges")
        parts.collect(db)
    size = store.path.stat().st_size
    assert recordings.compact(store, minimum=0) > 0
    assert store.path.stat().st_size < size / 2


def test_a_recording_waits_for_a_busy_store(store, monkeypatch):
    real, tries = store._record, []

    def busy(*args):
        tries.append(1)
        if len(tries) < 3:
            raise sqlite3.OperationalError("database is locked")
        return real(*args)

    monkeypatch.setattr(store, "_record", busy)
    assert store.record("/v1/chat/completions", {"messages": []}, {}) is not None
    assert len(tries) == 3


def test_a_store_not_yet_upgraded_reads_without_changes(tmp_path):
    """The CLI reads a store the running (older) service still owns without
    upgrading it (api.ReadOnlyStore)."""
    from songe.api import ReadOnlyStore

    root = tmp_path / "old"
    session = looping_session(3)
    old_store(root, session)
    view = ReadOnlyStore(root)
    assert [r["request"] for r in view.exchanges()] == [r for r, _ in session]
    with view.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_upkeep_waits_while_the_experience_child_runs(store):
    from songe.platform.locking import acquire

    with (store.root / "experience.lock").open("a+") as held:
        acquire(held)
        assert recordings.tidy(store) is None
    assert recordings.tidy(store) == 0  # nothing worth giving back


def test_an_unreadable_old_row_is_skipped_and_the_rest_goes_on(tmp_path):
    root = tmp_path / "old"
    session = looping_session(20)
    old_store(root, session)
    db = sqlite3.connect(root / "experience.sqlite3")
    db.execute("UPDATE exchanges SET request='{not json' WHERE id=3")
    db.commit()
    db.close()
    store = Store(root)
    assert recordings.dedupe(store) == 19
    assert recordings.dedupe(store) == 0  # not stuck on it
    with store.connect() as db:
        assert db.execute("SELECT request FROM exchanges WHERE id=3").fetchone()[0] == "{not json"


def test_parts_a_replaced_step_no_longer_uses_are_collected(store):
    for request, response in looping_session(3):
        store.record("/v1/chat/completions", request, response)
    store.reconstruct()
    step = store.active_steps()[0]
    with store.connect() as db:
        count = db.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    edited = step | {"hindsight": [{"role": "tool", "content": "rewritten " * 200}] * 3}
    store.sync_steps([edited])
    store.sync_steps([step])  # back: the rewritten part is used by nothing
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == count
        assert db.execute("SELECT COUNT(*) FROM part_uses WHERE n<=0").fetchone()[0] == 0
    assert next(s for s in store.active_steps() if s["id"] == step["id"]) == step


def test_upkeep_collects_parts_nothing_uses(store):
    for request, response in looping_session(3):
        store.record("/v1/chat/completions", request, response)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        parts._put(db, ["{\"orphan\": true}"])  # e.g. a failed rewrite's
    recordings.compact(store)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM part_uses WHERE n<=0").fetchone()[0] == 0
    assert len(store.exchanges()) == 3
