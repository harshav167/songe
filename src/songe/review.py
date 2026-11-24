"""Idle review: the local model reads how each user reply answered the model.

A phrase list cannot tell "这也太丑了" from a new request. In idle gaps the
model that is already loaded reads each step's short window (the request, what
the model did, what the tools returned, and the user's next message) and says
what that reply was. The verdict only labels evidence for lessons, practice
and the dashboard; training always sees the user's own words.
"""

import asyncio
import json
import logging
import re
import time

import httpx

from . import parts
from .steps import message_key, reply_key

logger = logging.getLogger(__name__)

KINDS = {
    "correction": "user_correction",
    "retry": "retry",
    "approval": "approval",
    "new_task": None,
    "question": None,
    "other": None,
}

SYSTEM = (
    "You review one moment from a coding assistant's work. Read what the user "
    "asked, what the assistant did, what the tools returned, and the user's "
    "next message. Classify that next message:\n"
    "- correction: the user says the result or approach was wrong or not good enough\n"
    "- retry: the user asks to try again\n"
    "- approval: the user accepts or thanks\n"
    "- new_task: the user moves on to something unrelated\n"
    "- question: the user asks something without judging the result\n"
    "- other\n"
    'Reply with JSON only: {"reply": "<kind>", "issue": "<what was wrong, '
    'under 12 words, or empty>", "lesson": "<a reusable rule for next time, '
    'under 16 words, or empty>"}'
)


def clip(value, limit):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def window(step):
    """The short evidence around one reply; never a whole conversation."""
    asked = next((m for m in reversed(step["context"]) if m.get("role") == "user"), {})
    action = step["action"]
    calls = ", ".join(
        f"{(c.get('function') or {}).get('name')}({clip((c.get('function') or {}).get('arguments', ''), 120)})"
        for c in action.get("tool_calls") or []
    )
    lines = [
        f"User asked: {clip(asked.get('content', ''), 500)}",
        f"Assistant did: {clip(action.get('content') or '', 300)} {calls}".strip(),
    ]
    for message in step["hindsight"][:-1][-4:]:
        role = "Tool returned" if message.get("role") == "tool" else "Assistant then"
        lines.append(f"{role}: {clip(message.get('content') or '', 240)}")
    lines.append(f"User replied: {clip(step['hindsight'][-1].get('content', ''), 600)}")
    return "\n".join(lines)


def pending(store, limit):
    """Unreviewed user replies that end a step, newest first."""
    found = {}
    with store.connect() as db, parts.snapshot(db):
        # The last hindsight message stays in the stored payload (parts.py).
        cache = {}
        rows = [
            (row["id"], parts.payload(db, row, cache))
            for row in db.execute(
                "SELECT * FROM steps WHERE retired_at IS NULL "
                "AND json_extract(payload,'$.hindsight[#-1].role')='user' "
                "ORDER BY rowid DESC LIMIT ?",
                (limit * 20,),
            ).fetchall()
        ]
    for step_id, step in rows:
        key = reply_key(step)
        found.setdefault(key, (step, []))[1].append(step_id)
    done = store.reviewed(list(found)) if found else set()
    return [(key, *found[key]) for key in found if key not in done][:limit]


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match[0])
    except ValueError:
        return None
    reply = str(value.get("reply", "")).strip().lower().replace(" ", "_")
    if reply not in KINDS:
        return None
    return {
        "reply": reply,
        "issue": clip(value.get("issue") or "", 120),
        "lesson": clip(value.get("lesson") or "", 160),
    }


async def review(store, url, model, *, budget_seconds=40, limit=8, still_idle=None):
    """Review up to `limit` replies within the budget; stop when the user returns."""
    started, begun, done = time.monotonic(), time.time(), 0
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        # The scan reads step payloads; off the event loop so answers being
        # relayed never stall behind it.
        for key, step, step_ids in await asyncio.to_thread(
            lambda: list(pending(store, limit))
        ):
            if time.monotonic() - started > budget_seconds:
                break
            if still_idle is not None and not still_idle():
                break
            response = await client.post(
                url.rstrip("/") + "/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        # Qwen-style models honour /no_think; others ignore it.
                        {"role": "user", "content": window(step) + "\n/no_think"},
                    ],
                    "temperature": 0,
                    "max_tokens": 800,
                    "stream": False,
                },
            )
            response.raise_for_status()
            choice = (response.json().get("choices") or [{}])[0]
            verdict = parse((choice.get("message") or {}).get("content") or "")
            # An unreadable verdict is recorded too, so it is not retried forever;
            # its reply keeps the phrase-list fallback.
            kind = KINDS.get((verdict or {}).get("reply"))
            payload = (verdict or {"reply": "unknown"}) | {
                "model": model,
                "text_key": message_key(step["hindsight"][-1]),
            }
            if verdict is None:
                from .steps import user_signal

                kind = user_signal(step["hindsight"][-1])
            store.add_review(key, kind, payload, step_ids, since=begun)
            done += 1
    return done
