"""Before/after examples: what changed, on concrete moments
(docs/CONTRACTS.md, Learning cycle, "Before/after examples").

Sets, evaluated in display order so a short idle gap still yields the
examples most worth showing:
- original moments: for each signal with a lesson, the logged request at
  its responsible step. Real tasks never enter training under dreams.
- held-out practice: scenes of signals split off from training.
- trained practice: a few scenes the cycle trained on (never counted in
  the headline).

Before = the cycle's starting version, after = the candidate; same prompt,
greedy decoding, same token limit. The cycle's teacher, shown the same
context plus a hint (the real feedback for an original moment, the teacher
note for practice), scores both: teacher_preference is the mean per-token
log-probability of the after answer minus the before answer's.
"""

import contextlib
import json
import re

from ..parts import request_of
from .evaluate import health
from .prompts import UnusableStep, paired_prompts, render

LESSON_TYPES = ("correction", "retry", "tool_error", "failed_result", "outcome")


JUDGE_TOKENS = 24  # {"follows": "B"} and a little room


def clip(text, n=300):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def original_moments(store):
    """Steps for the original moments, user signals first, then tool ones."""
    from ..reading.store import signals

    out = []
    with store.connect() as db:
        for s in signals(store, types=LESSON_TYPES):
            if not s.get("lesson") or not s.get("exchanges"):
                continue
            # The user's feedback reacts to the step just before it (the README
            # with the table, not the first step of the task); a tool's to
            # the step that went wrong first.
            pick = max if s.get("source") == "user" else min
            step = pick(int(n) for n in s["exchanges"])
            exchange = s["exchanges"][str(step)]
            found = request_of(db, exchange, "*")
            if not found:
                continue
            request = found["request"]
            at = found.get("recorded_at")
            row = (None, found["created"] if at is None else at)
            messages = request.get("messages") or []
            asked = next(
                (m for m in reversed(messages) if m.get("role") == "user"), {}
            ).get("content")
            out.append(
                {
                    "id": f"original:{s['id']}",
                    "conversation": "session:" + s["session"],
                    "split": "original",
                    "context": messages,
                    "tools": request.get("tools") or [],
                    # The teacher's hint: what really followed.
                    "hindsight": [{"role": "user", "content": s.get("text") or ""}],
                    "action": {},
                    "meta": {
                        "source": "original",
                        "split": "original",
                        "signal": {"id": s["id"], "type": s["type"], "source": s["source"]},
                        "context": {
                            "request": clip(asked if isinstance(asked, str) else json.dumps(asked)),
                            "conversation": s["session"],
                            "at": row[1],
                        },
                        "feedback": clip(s.get("text")),
                        "lesson": s.get("lesson"),
                    },
                }
            )
    order = {"user": 0, "tool": 1}
    return sorted(out, key=lambda e: order.get(e["meta"]["signal"]["source"], 2))


def practice_moments(store, steps, split):
    """Practice steps as example candidates (split: held_out or trained)."""
    from ..practice.store import practice

    rows = {r["id"]: r for r in practice(store, kept=True)}
    out = []
    for s in steps:
        row = rows.get(s["id"].removeprefix("practice:"))
        if not row:
            continue
        scene = row.get("scene") or {}
        signal = row.get("signal") or {}
        out.append(
            s
            | {
                "meta": {
                    "source": "practice",
                    "split": split,
                    "signal": {
                        "id": row["signal_id"],
                        "type": signal.get("type"),
                        "source": signal.get("source"),
                    },
                    "context": {"request": clip(scene.get("request")), "conversation": None, "at": None},
                    "feedback": None,
                    "lesson": signal.get("lesson"),
                }
            }
        )
    return out


def answer(text, tokenizer, tools):
    """The text, parsed tool calls, and whether it is well-formed: any tool
    call parses and names a known tool (the gate's own health check)."""
    marked = bool(re.search(r"<tool_call>|<\|tool_call|<function=", text or ""))
    checked = health(text or "", tokenizer, tools, marked)
    calls = [
        {"name": c.get("name"), "arguments": c.get("arguments")}
        for c in checked.get("calls") or []
        if isinstance(c, dict)
    ]
    valid = bool(checked["tool_valid"]) if marked else not checked["empty"]
    return {"text": text or "", "tool_calls": calls, "valid": valid}


def teacher_logprob(backend, teacher_prompt, y):
    """Mean per-token log-probability of y under the cycle's teacher."""
    h = backend.teacher_hidden(teacher_prompt, y, backend.teacher())
    with backend.base():
        return backend.token_logprob(h, y)


def plain_logprob(backend, student_prompt, y):
    """Mean per-token log-probability of y under the loaded version, shown
    only the context (no hint)."""
    quiet = backend.torch.no_grad() if backend.kind == "torch" else contextlib.nullcontext()
    with quiet:
        h = backend.teacher_hidden(student_prompt, y, contextlib.nullcontext())
        return backend.token_logprob(h, y)


def side(backend, step, seed=None):
    """One version's greedy answer (or a sampled one, with a seed): its
    tokens, the teacher's score, the loaded version's own score and the
    teacher-student gap."""
    student, teacher = paired_prompts(backend.tokenizer, step, backend.config)
    if seed is None:
        y, text, _ = backend.sample(student, deterministic=True)
    else:
        y, text, _ = backend.sample(student, seed=seed)
    return {
        "text": text,
        "tokens": list(y),
        "logprob": teacher_logprob(backend, teacher, y),
        "plain": plain_logprob(backend, student, y),
        "gap": backend.compare(student, teacher, y),
    }


def contrast(a, b):
    """How much more the teacher prefers a over b than the starting version
    without the hint does. What both prefer anyway (fluency, a likelier
    phrasing) cancels; what is left is what the hint explains."""
    return (a["logprob"] - b["logprob"]) - (a["plain"] - b["plain"])


def run(backend, journal, sets, *, baseline, trained, after_scale, config, check,
        progress=lambda done, total: None):
    """Evaluate the example sets; returns the record fields (examples,
    improved, example_checks, example_margin). Resumable: every answer is
    a journal unit. progress(done, total) counts answers compared."""
    order = [(name, step) for name, steps in sets for step in steps[: config.example_limit]]
    usable = []
    for name, step in order:
        try:
            paired_prompts(backend.tokenizer, step, backend.config, check_length=False)
            usable.append((name, step))
        except UnusableStep:
            continue
    answers = {}
    # Before and after answers, the after answers rescored, the noise pairs.
    total = 3 * len(usable) + len(usable[: config.example_noise_samples])
    done = [0]

    def advanced():
        done[0] += 1
        progress(done[0], total)

    if total:
        progress(0, total)
    for which, scale in (("before", 0.0), ("after", after_scale)):
        backend.scale_delta(baseline, trained, scale)
        for _, step in usable:
            check()
            answers[(which, step["id"])] = journal.unit(
                f"example-{which}-v2-{step['id']}", lambda step=step: side(backend, step)
            )
            advanced()
    # Both answers are scored without the hint by the starting version, so
    # the after answer is rescored with the starting weights.
    backend.scale_delta(baseline, trained, 0.0)
    for _, step in usable:
        check()
        after = answers[("after", step["id"])]
        student, _ = paired_prompts(backend.tokenizer, step, backend.config)
        after["plain"] = journal.unit(
            f"example-plain-{step['id']}",
            lambda student=student, after=after: plain_logprob(backend, student, after["tokens"]),
        )
        advanced()
    # The noise: the same contrast between two sampled answers of the
    # starting version. An improvement must beat it.
    noise = []
    for _, step in usable[: config.example_noise_samples]:
        check()
        pair = journal.unit(
            f"example-noise-v2-{step['id']}",
            lambda step=step: [side(backend, step, seed=7613 + k) for k in (1, 2)],
        )
        noise.append(abs(contrast(pair[0], pair[1])))
        advanced()
    margin = config.example_margin or (max(noise) if noise else 0.0)
    from ..quick import differs, judge_prompt, verdict_of

    examples = []
    for _, step in usable:
        before, after = answers[("before", step["id"])], answers[("after", step["id"])]
        b = answer(before["text"], backend.tokenizer, step["tools"])
        a = answer(after["text"], backend.tokenizer, step["tools"])
        lesson = contrast(after, before)
        verdict = None
        if a["valid"] and lesson > margin and differs(b, a) and step["meta"].get("lesson"):
            # Scored better and acting differently; the starting version (no
            # self-preference) says which answer follows the lesson (live:
            # a "read the file" answer shown as better than the actual fix).
            prompt, swap = judge_prompt(step["meta"]["lesson"], b, a)
            check()
            text = journal.unit(f"example-judge-{step['id']}", lambda prompt=prompt: backend.sample(
                render(backend.tokenizer, [{"role": "user", "content": prompt}], None),
                deterministic=True, max_tokens=JUDGE_TOKENS)[1])
            verdict = verdict_of(text, swap)
        examples.append(
            {"id": step["id"], **step["meta"], "before": b, "after": a,
             "teacher_preference": after["logprob"] - before["logprob"],
             "lesson_preference": lesson, "verdict": verdict,
             "improved": verdict == "after",
             "gap": {"before": before["gap"], "after": after["gap"]}}
        )
    backend.scale_delta(baseline, trained, after_scale)
    counted = [e for e in examples if e["split"] in ("original", "held_out")]
    available = sum(len(steps) for name, steps in sets if name in ("original", "held_out"))

    def gap_check(split):
        rows = [e for e in examples if e["split"] == split]
        total = sum(len(steps) for name, steps in sets if name == split)
        if not rows:
            return {"state": "done", "count": {"done": 0, "total": total}}
        return {
            "state": "done",
            "before": sum(e["gap"]["before"] for e in rows) / len(rows),
            "after": sum(e["gap"]["after"] for e in rows) / len(rows),
            "count": {"done": len(rows), "total": total},
        }

    originals = [e for e in examples if e["split"] == "original"]
    total_originals = sum(len(steps) for name, steps in sets if name == "original")
    return {
        "examples": examples,
        "example_margin": margin,
        "example_noise": noise,
        "improved": {
            "count": sum(e["improved"] for e in counted),
            "checked": len(counted),
            "available": available,
        },
        "example_checks": {
            "new_practice": gap_check("held_out"),
            "trained_practice": gap_check("trained"),
            "original_moments": {
                "state": "done",
                "after": sum(e["improved"] for e in originals) / len(originals) if originals else None,
                "count": {"done": len(originals), "total": total_originals},
            },
        },
    }
