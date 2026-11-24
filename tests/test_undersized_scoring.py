"""The 445-example/zero-pending deadlock, without model or GPU work."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe import schedule
from songe.config import Config
from songe.log import Store
from songe.memory import Snapshot
from songe.models import registry
from songe.sleep import calibrate, worker
from songe.sleep.calibration_state import (
    measure_pool,
    needs_growth,
    policy_key,
    pool_key,
    progress,
    sample_pool,
)
from songe.sleep.recovery import RECOVERY_KEY
from songe.sleep.samples import SampleAdmission
from songe.sleep.select import select
from songe.status import experience_wait
from songe.wording import status_copy
from tests.test_partial_calibration import (
    GB,
    Tokenizer,
    original_rows,
    step,
)
from tests.test_partial_calibration import (
    measured as measured,
)
from tests.test_worker import Backend


def legacy_state(config, store, *, overwritten_policy=True):
    """Minimal equivalent of the snapshot, including the `new` cutoff join."""
    config.calibration_available = True
    config.artifact_kind = "mlx-adapter"
    config.force_cpu_training = True
    config.upstream = "mlx"
    config.save(store.root)
    Path(config.model).mkdir(parents=True, exist_ok=True)
    registry.register(config, store, "base", legacy=True)
    store.set("primary_model", "base")
    with store.connect() as db:
        db.execute("UPDATE models SET learning_level='new',learning_cutoff=1")
        db.executemany(
            "INSERT INTO exchanges(id,created,path,request,response,status,complete,binding,recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (i, 2, "/v1/chat/completions", "{}", "{}", 200, 1, "base", 2)
                for i in range(1, 1404)
            ],
        )
    train = [
        step(900, i) | {"action": {"role": "assistant", "content": "answer"}}
        for i in range(1, 446)
    ]
    held = [
        step(900, i)
        | {"split": "heldout", "action": {"role": "assistant", "content": "answer"}}
        for i in range(446, 479)
    ]
    store.sync_steps(train + held)
    store.mark_scored(train)
    samples, _, key = measure_pool(config, store, Tokenizer())
    size = dict(max_sequence=44, response_tokens=1, token_chunk=1, accumulation=4)
    saved = {"rows": original_rows(), "budget": size, "samples": samples}
    saved["progress"] = progress(
        replace(
            config,
            **{k: size[k] for k in ("max_sequence", "response_tokens", "token_chunk")},
        )
        if overwritten_policy
        else config,
        saved,
        44,
        key,
    )
    # A legacy completion flag describes its probe, not the sample distribution.
    saved["progress"]["complete"] = True
    store.set("calibration", saved)
    store.set("last_slice_size", size)
    if overwritten_policy:
        registry.save_config(replace(config, model_key="base", **size), store)
    for _ in range(2):
        store.night(
            None,
            status="waiting",
            phase="select",
            record={
                "steps_seen": 445,
                "slice_sizes": [size],
                "reason": "Waiting for learnable experience",
            },
        )
    with store.connect() as db:
        db.execute("DELETE FROM state WHERE key=?", (RECOVERY_KEY,))
    return train, held


@pytest.mark.parametrize("overwritten", [False, True])
def test_legacy_scoring_recovers_once_without_changing_learning_window(
    config, store, overwritten
):
    train, _ = legacy_state(config, store, overwritten_policy=overwritten)
    assert store.experience_counts() == (445, 33, 0)
    assert store.pending_learnable() == 0 and not store.pool()
    cutoff = registry.find(store, "base")["learning_cutoff"]
    recovered = Store(store.root)
    assert recovered.experience_counts() == (445, 33, 445)
    assert recovered.pending_learnable() == 445
    assert all(n["status"] == "discarded" for n in recovered.nights())
    assert registry.find(recovered, "base")["learning_cutoff"] == cutoff
    policy = Config.for_model(store.root, "base")
    assert (policy.max_sequence, policy.response_tokens, policy.token_chunk) == (
        2048,
        128,
        32,
    )
    assert needs_growth(policy, recovered)
    assert "measuring training size" in experience_wait(
        config, recovered, registry.summaries(config, recovered)
    )
    recovered.mark_scored(train[:1])
    generation = recovered.get("candidate_generation")
    reopened = Store(store.root)
    assert reopened.experience_counts() == (445, 33, 444)
    assert reopened.get("candidate_generation") == generation


async def test_data_distribution_triggers_growth_even_with_zero_pending(config, store):
    legacy_state(config, store, overwritten_policy=False)
    # No reopen/migration: test growth independently of revision recovery.
    saved = store.get("calibration")
    saved["progress"]["policy"] = policy_key(config)
    assert saved["progress"]["pool"] == pool_key(sample_pool(store))
    store.set("calibration", saved)
    assert store.pending_learnable() == 0
    assert needs_growth(config, store)
    scheduler = schedule.ModelScheduler(config, store, SimpleNamespace())
    assert scheduler.choose_model() == "base"
    assert await scheduler.next_model() == "base"


def test_mixed_slice_keeps_long_tail_pending_until_it_fits(config, store):
    short, long = step(900, 1), step(1000, 2)
    store.sync_steps([short, long])
    rows = [
        original_rows()[0]
        | {"sequence": 1000, "response_tokens": 128, "token_chunk": 32}
    ]
    saved = {"rows": rows}
    from songe.sleep.prompts import paired_prompts

    def teacher_need(sample):
        student, teacher = paired_prompts(Tokenizer(), sample, config)
        return max(len(student), len(teacher)) + config.response_tokens

    # The gradient window adapts to the slice; the full-context forward
    # (the measured teacher envelope) is what defers a long example.
    sizing = dict(
        max_sequence=900,
        response_tokens=128,
        max_teacher_sequence=teacher_need(long) - 1,
    )
    admission = SampleAdmission(config, sizing, saved, store, store.slice())
    admitted = admission.admit([short, long], Tokenizer())
    assert admitted == [short]
    selected, _ = select(admitted)
    store.add_pool(selected, scored=admitted)
    store.mark_scored(admitted)
    assert store.unscored([short, long]) == [long]
    assert admission.record["deferred_samples"][0]["id"] == long["id"]
    larger = SampleAdmission(
        config,
        sizing | {"max_teacher_sequence": teacher_need(long)},
        saved,
        store,
        store.slice(),
    )
    admitted = larger.admit(store.unscored([short, long]), Tokenizer())
    selected, _ = select(admitted)
    store.add_pool(selected, scored=admitted)
    store.mark_scored(admitted)
    assert not store.unscored([short, long])
    assert {s["id"] for s in store.pool()} == {short["id"], long["id"]}


def test_probe_response_budget_cannot_start_scoring(config, store, monkeypatch):
    config.calibration_available = True
    monkeypatch.setattr(
        worker, "preview", lambda *a: ({"reason": "Ready"}, [step()], [])
    )
    monkeypatch.setattr(worker, "choose", lambda *a: pytest.fail("Probe cannot score"))
    assert (
        worker.run(config, store, sizing={"max_sequence": 2048, "response_tokens": 1})[
            "reason"
        ]
        == "Waiting for calibration growth"
    )
    assert not store.nights() and store.unscored([step()])


def test_migration_preserves_valid_scores_and_trained_revisions(config, store):
    train, _ = legacy_state(config, store, overwritten_policy=False)
    saved = store.get("calibration")
    saved["samples"][1]["required_sequence"] = 300
    store.set("calibration", saved)
    store.set("last_slice_size", dict(max_sequence=400, response_tokens=128))
    store.set("experience_watermark", 1)
    # This fixture isolates length recovery at a usable response budget.
    with store.connect() as db:
        db.execute("DELETE FROM nights")
    # A different learner's scores and pool are independent.
    other = store.scoped("other")
    other.mark_scored(train)
    recovered = Store(store.root)
    assert not recovered.unscored(train[:2])
    assert len(recovered.unscored(train[2:])) == 443
    assert not recovered.scoped("other").unscored(train)


@pytest.mark.parametrize("known_lengths", [False, True])
def test_larger_measurement_does_not_validate_earlier_tiny_scores(
    config, store, known_lengths
):
    legacy_state(config, store, overwritten_policy=False)
    if not known_lengths:
        saved = store.get("calibration")
        saved.pop("samples")
        store.set("calibration", saved)
    store.set("last_slice_size", dict(max_sequence=2048, response_tokens=128))
    assert Store(store.root).pending_learnable() == 445


def test_legacy_complete_flag_cannot_override_missing_measured_coverage(config, store):
    store.sync_steps([step()])
    samples, target, key = measure_pool(config, store, Tokenizer())
    saved = {"rows": original_rows(), "samples": samples}
    saved["progress"] = progress(config, saved, target, key) | {"complete": True}
    store.set("calibration", saved)
    store.mark_scored([step()])
    assert store.pending_learnable() == 0
    assert needs_growth(config, store)


def test_selected_pool_is_preparation_even_when_every_revision_is_scored(config, store):
    train, _ = legacy_state(config, store, overwritten_policy=False)
    store.set("calibration", {})
    store.add_pool([s | {"surprise": 1.0} for s in train])
    assert store.experience_counts() == (445, 33, 0)
    assert experience_wait(config, store, registry.summaries(config, store)) == (
        "preparing examples · 445 examples waiting to learn"
    )


def test_examples_longer_than_the_gradient_window_still_train(config, store):
    """Agent contexts are long; backprop covers only the tail (DESIGN 3.3), so
    a context beyond max_sequence is admitted, never marked unfit for length."""
    train, _ = legacy_state(config, store)
    store = Store(store.root)
    policy = replace(config, model_key="base", max_sequence=800)
    registry.save_config(policy, store)
    samples, target, key = measure_pool(policy, store, Tokenizer())
    assert any(s["required_sequence"] > 800 for s in samples)
    saved = {
        "rows": [
            original_rows()[0]
            | {"sequence": 800, "response_tokens": 128, "token_chunk": 32}
        ],
        "samples": samples,
    }
    saved["progress"] = progress(policy, saved, target, key)
    store.set("calibration", saved)
    admitted = SampleAdmission(policy, None, saved, store, None).admit(
        train, Tokenizer()
    )
    assert len(admitted) == len(train)
    assert not store.get("unfit_samples")


async def test_next_idle_gaps_measure_select_pool_and_train(
    config, store, measured, monkeypatch
):
    calibration_backend, machine = measured
    train, _ = legacy_state(config, store)
    store = Store(store.root)
    assert store.experience_counts() == (445, 33, 445)
    scheduler = schedule.ModelScheduler(config, store, SimpleNamespace(), machine)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(scheduler, "discover_models", lambda: asyncio.sleep(0))
    monkeypatch.setattr(schedule, "guard", lambda *a, **kw: None)
    monkeypatch.setattr(
        schedule, "capture_baseline", lambda *a, **kw: Snapshot(24 * GB, 20 * GB)
    )
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker, "current", lambda: machine)
    monkeypatch.setattr(worker, "probe", lambda: schedule.probe(machine))

    class TrainingBackend(Backend):
        scored, trained_units, initial_adapters = [], [], []
        comparisons = 0
        interrupt_training = interrupt_evaluation = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = Tokenizer()

    monkeypatch.setattr(worker, "choose", lambda *a: TrainingBackend)
    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)
        if "songe.sleep.calibrate" in args:
            calibrate._calibrate(scheduler.config, store.root, (256, 512, 1024, 2048))
        else:
            sizing = json.loads(args[args.index("--sizing") + 1])
            worker.run(scheduler.config, scheduler.store, sizing=sizing)
        return SimpleNamespace(pid=12345, returncode=0, wait=lambda: asyncio.sleep(0))

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    await scheduler.tick()
    await scheduler.guard_task
    assert "songe.sleep.calibrate" in launched[0]
    assert (900, 128, 32, 4) in calibration_backend.calls
    assert not needs_growth(scheduler.config, scheduler.store)
    waiting = experience_wait(config, store, registry.summaries(config, store))
    assert waiting == "preparing examples · 445 examples waiting to learn"
    assert status_copy({"activity": {"now": "waiting · " + waiting}}) == (
        "Preparing training data",
        "445 to prepare",
    )
    await scheduler.tick()  # Collect the completed calibration child.
    store.set("last_attempt", 0)
    await scheduler.tick()
    await scheduler.guard_task
    assert "songe.sleep.worker" in launched[1]
    # Every recorded example is selected without a scoring pass (DESIGN 3.4).
    assert TrainingBackend.scored == []
    assert sorted(s["id"] for s in store.pool()) == sorted(s["id"] for s in train)
    assert TrainingBackend.trained_units
    assert store.nights()[-1]["record"]["steps_learned"] > 0
    assert store.nights()[-1]["record"]["consolidate_started"]
    assert store.nights()[-1]["status"] in {"ready", "rejected"}
