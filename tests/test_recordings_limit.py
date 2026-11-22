"""The size limit on recordings (issue #73): over it, the oldest recordings
already learned from are deleted; never one that has not been."""

import json
import time

from songe import api, recordings
from songe.cli import recordings_line
from songe.models import registry
from songe.reading import store as reading
from songe.reading import timeline
from songe.wording import recordings_note

SYSTEM = {"role": "system", "content": "You are an agent. " * 400}
TOOLS = [{"type": "function", "function": {"name": "shell", "description": "Run. " * 200,
                                           "parameters": {"type": "object"}}}]
LATER = time.time() + 2 * recordings.RESUMABLE_SECONDS


def reply(message):
    return {"choices": [{"message": message}]}


def session(store, task, turns=4):
    """One conversation: a task, then tool calls and their results."""
    history = [SYSTEM, {"role": "user", "content": task}]
    for i in range(turns):
        action = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"{task[:3]}{i}", "type": "function",
             "function": {"name": "shell", "arguments": json.dumps({"command": f"step {i}"})}}]}
        store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(action))
        history = history + [action, {"role": "tool", "tool_call_id": f"{task[:3]}{i}",
                                      "content": f"{task}: output {i} " * 50}]
    store.reconstruct(batch=200)


def sessions(store):
    with store.connect() as db:
        return timeline.sessions(db)


def read(learner, key, ids):
    """The learner's Dream read this session at its current revision."""
    with learner.connect() as db:
        db.execute("INSERT OR REPLACE INTO read_sessions VALUES (?,?,?,?)",
                   (key, learner.model_id, max(ids), time.time()))


def lesson(learner, key, signal_id, exchange):
    with learner.connect() as db:
        db.execute(
            "INSERT INTO signals(id,model_id,session,revision,payload) VALUES (?,?,?,?,?)",
            (signal_id, learner.model_id, key, exchange, json.dumps({
                "id": signal_id, "type": "correction", "source": "user", "session": key,
                "lesson": "Run the tests with pytest -q.", "text": "use pytest -q",
                "exchanges": {"1": exchange}, "revision": exchange})),
        )


def setup(config, store):
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    for row in registry.entries(store):
        if row["id"] != "learner":  # one learner here; another must read too
            registry.set_learning(store, row["id"], "off")
    learner = store.scoped("learner")
    for task in ("first task", "second task", "third task"):
        session(store, task)
    found = sorted(sessions(store).items(), key=lambda kv: min(kv[1]))
    assert len(found) == 3
    return learner, found


def requests(store):
    return {r["id"]: r["request"] for r in store.exchanges()}


def test_only_learned_sessions_go_oldest_first(config, store):
    learner, found = setup(config, store)
    (first, a), (second, b), (third, c) = found
    assert recordings.learned_sessions(store, LATER) == []  # nothing read yet
    read(learner, first, a)
    read(learner, second, b)
    # The second taught a lesson its Dream has not trained yet: not learned.
    lesson(learner, second, "sig-2", b[0])
    assert recordings.learned_sessions(store, LATER) == [(first, a)]
    learner.set("trained_signals", ["sig-2"])
    assert recordings.learned_sessions(store, LATER) == [(first, a), (second, b)]
    # A conversation the agent may still continue stays (used today).
    assert recordings.learned_sessions(store) == []


def test_over_the_limit_deletes_learned_oldest_first_and_keeps_the_rest(config, store):
    learner, found = setup(config, store)
    (first, a), (second, b), (third, c) = found
    read(learner, first, a)
    read(learner, second, b)
    lesson(learner, first, "sig-1", a[0])
    learner.set("trained_signals", ["sig-1"])
    before = requests(store)
    used = recordings.sizes(store)[0]

    # Just over: only the oldest goes.
    assert recordings.keep_under(store, used - 1, now=LATER) == len(a)
    after = requests(store)
    assert set(after) == set(b) | set(c)
    assert all(after[i] == before[i] for i in after)  # the rest reads the same
    assert first not in sessions(store)
    # What was read from it stays, under a name of its own.
    assert [s["id"] for s in reading.signals(learner)] == ["sig-1"]
    assert reading.forget_vanished(learner) == 0
    assert [s["id"] for s in reading.signals(learner)] == ["sig-1"]

    # Far over: every learned session goes, the unread one never does.
    assert recordings.keep_under(store, 1, now=LATER) == len(b)
    assert set(requests(store)) == set(c)
    assert requests(store) == {i: before[i] for i in c}
    assert recordings.keep_under(store, 1, now=LATER) == 0
    # The system prompt and tools it shares with the deleted ones are kept.
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] > 0
        assert db.execute("SELECT MIN(n) FROM part_uses").fetchone()[0] >= 1
        assert db.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == len(c)


def test_nothing_is_deleted_while_a_dream_is_under_way(config, store):
    learner, found = setup(config, store)
    for key, ids in found:
        read(learner, key, ids)
    night = learner.night(None)  # a Deep Dream running for this learner
    assert recordings.keep_under(store, 1, now=LATER) == 0
    learner.night(night, status="interrupted")
    assert recordings.keep_under(store, 1, now=LATER) == 0
    learner.night(night, status="accepted")
    assert recordings.keep_under(store, 1, now=LATER) == sum(len(ids) for _, ids in found)


def test_every_learner_learning_from_a_session_must_have_learned_from_it(config, store):
    learner, found = setup(config, store)
    (first, a), _, _ = found
    registry.observe(config, store, "other")
    registry.set_learning(store, "other", "all")
    read(learner, first, a)
    assert recordings.learned_sessions(store, LATER) == []
    read(store.scoped("other"), first, a)
    assert recordings.learned_sessions(store, LATER) == [(first, a)]


def test_a_model_with_learning_off_never_learned_from_anything(config, store):
    learner, found = setup(config, store)
    for key, ids in found:
        read(learner, key, ids)
    registry.set_learning(store, "learner", "off")
    assert recordings.keep_under(store, 1, now=LATER) == 0
    assert len(requests(store)) == sum(len(ids) for _, ids in found)


def test_over_the_limit_with_nothing_learned_says_so(config, store, tmp_path):
    setup(config, store)
    config.recordings_limit_gb = 1e-6  # 1 KB
    used = recordings.sizes(store)[0]
    assert recordings.keep_under(store, recordings.limit_bytes(config), now=LATER) == 0
    note = api.recordings_note(config, store)
    assert note == recordings_note(used, 1000.0)
    assert "over the 0.0 GB limit" in note and "waiting to be learned from" in note
    assert "waiting to be learned" in recordings_line(config, store.root)
    config.recordings_limit_gb = 2.0
    assert api.recordings_note(config, store) is None
    assert recordings_line(config, store.root).startswith("Recordings: 0.0 GB of the 2.0 GB limit")


def test_the_service_upkeep_applies_the_limit(config, store):
    learner, found = setup(config, store)
    (first, a), _, _ = found
    read(learner, first, a)
    with store.connect() as db:  # recorded two days ago
        db.execute("UPDATE exchanges SET recorded_at=?", (time.time() - 2 * recordings.RESUMABLE_SECONDS,))
    assert recordings.tidy(store, 1) is not None
    assert not set(a) & set(requests(store))


def test_the_limit_is_a_setting(tmp_path, config):
    from songe.config import Config

    assert Config().recordings_limit_gb == 2.0
    config.recordings_limit_gb = 0.5
    config.save(tmp_path)
    assert Config.load(tmp_path).recordings_limit_gb == 0.5
    config.recordings_limit_gb = 0
    try:
        config.validate()
    except ValueError as exc:
        assert "recordings_limit_gb" in str(exc)
    else:
        raise AssertionError("a zero limit must be refused")
