"""The dashboard's Flow and Raw data: what Songe just did, and what it recorded.

Flow is the events table, newest first. Raw data is the latest recorded steps
as the learner sees them: what you asked, what the model did, what came back,
your reply, the review's reading of it, and whether cleaning keeps the step.
"""

import json

from . import parts
from .sleep.clean import verdict
from .steps import reply_key
from .wording import flow_line


def clip(value, limit):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def calls(action):
    return ", ".join(
        f"{(c.get('function') or {}).get('name')}"
        f"({clip((c.get('function') or {}).get('arguments') or '', 80)})"
        for c in action.get("tool_calls") or []
    )


def raw_step(step, reviews):
    asked = next(
        (m for m in reversed(step.get("context") or []) if m.get("role") == "user"),
        {},
    )
    action = step.get("action") or {}
    hindsight = step.get("hindsight") or []
    reply = hindsight[-1] if hindsight and hindsight[-1].get("role") == "user" else None
    returned = next((m for m in hindsight if m.get("role") == "tool"), None)
    review = reviews.get(reply_key(step)) if reply else None
    dropped = verdict(step)
    return {
        "id": step.get("id"),
        "conversation": str(step.get("conversation") or "")[:8],
        "split": "test" if step.get("split") == "heldout" else "train",
        "asked": clip(asked.get("content") or "", 240),
        "did": clip(
            " ".join(filter(None, [action.get("content") or "", calls(action)])), 240
        ),
        "returned": clip(returned.get("content") or "", 200) if returned else "",
        "reply": clip(reply.get("content") or "", 200) if reply else "",
        "review": review,
        "signal": (step.get("signal") or "").replace("_", " "),
        "kept": dropped is None,
        "dropped": dropped,
    }


def flow(store, events=60, steps=12, *, every_model=False):
    from . import steps as signals
    from .models.library import model_name

    store.load_reviews()
    with store.connect() as db, parts.snapshot(db):
        cache = {}
        rows = [
            parts.payload(db, row, cache)
            for row in db.execute(
                "SELECT * FROM steps WHERE retired_at IS NULL "
                "ORDER BY rowid DESC LIMIT ?",
                (steps,),
            ).fetchall()
        ]
    reviews = {
        key: " · ".join(
            filter(
                None,
                [str(note.get("reply") or "").replace("_", " "), note.get("issue")],
            )
        )
        for key, note in signals.REVIEW_NOTES.items()
    }
    return {
        "events": [
            {
                "created": e["created"],
                "model": model_name(e["model_id"]) if e["model_id"] else "",
                "stage": e["stage"],
                # Older lines read in today's wording too.
                "text": line,
            }
            for e in store.events(events, every_model=every_model)
            # Unlabelled lines are installation-wide recording and reading,
            # or the primary model's older lines: not another model's.
            if (
                every_model
                or e["model_id"]
                or e["stage"] in ("record", "review")
                or not store.model_id
            )
            and (line := flow_line(e["stage"], e["text"]))
        ],
        "steps": [raw_step(step, reviews) for step in rows],
    }
