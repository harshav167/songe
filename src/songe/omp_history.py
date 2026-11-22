"""Import saved OMP turns without replaying a request against the model server."""

import hashlib
import json
import time
from collections import Counter
from datetime import datetime

from .experience import refresh
from .log import Store


def message(raw):
    role = raw.get("role")
    if role not in {"user", "assistant", "developer", "toolResult"}:
        return None
    if (
        raw.get("stopReason") in {"error", "aborted"}
        or raw.get("attribution") == "agent"
    ):
        return None
    blocks = raw.get("content")
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    if not isinstance(blocks, list):
        return None
    texts, calls = [], []
    for block in blocks:
        if not isinstance(block, dict):
            return None
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif role == "assistant" and kind == "toolCall":
            if not all(
                isinstance(block.get(k), str) for k in ("id", "name")
            ) or not isinstance(block.get("arguments"), dict):
                return None
            calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["arguments"], ensure_ascii=False),
                    },
                }
            )
        elif kind != "thinking":
            return None
    msg = {"role": role, "content": "\n".join(texts)}
    if calls:
        msg["tool_calls"] = calls
    if role == "toolResult":
        if not isinstance(raw.get("toolCallId"), str):
            return None
        msg.update(role="tool", tool_call_id=raw["toolCallId"])
    return msg if msg["content"] or calls else None


def turns(path, provider, model, stats):
    # Non-message entries point at the last message in their own branch.
    links = {}
    session = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected an OMP entry")
            kind, key = row.get("type"), row.get("id")
            if kind == "session":
                session = key
                continue
            if not session or not isinstance(key, str):
                continue
            parent = links.get(row.get("parentId"))
            links[key] = parent
            if kind in {"reset_boundary", "compaction", "branch_summary"}:
                links[key] = None
            if kind != "message":
                continue
            raw = row.get("message") or {}
            if raw.get("role") == "assistant" and (
                raw.get("provider"),
                raw.get("model"),
            ) != (provider, model):
                stats["other_model"] += 1
                links[key] = None
                continue
            msg = message(raw)
            if not msg or (not parent and msg["role"] not in {"user", "developer"}):
                links[key] = None
                stats["unsupported_or_missing_context"] += 1
                continue
            links[key] = (parent, msg)
            if msg["role"] != "assistant":
                continue
            history = []
            while parent:
                parent, prior = parent
                history.append(prior)
            history.reverse()
            when = datetime.fromisoformat(row["timestamp"]).timestamp()
            identity = hashlib.sha256(
                json.dumps([session, key, provider, model]).encode()
            ).hexdigest()
            yield identity, when, history, msg


def backfill(args, root, config):
    directory = args.sessions.expanduser()
    if not directory.is_dir():
        raise ValueError("Expected a directory of OMP session JSONL files")
    if args.apply and (not config.model or not (root / "experience.sqlite3").is_file()):
        raise ValueError("Initialize the target Songe model before importing")
    store = Store(root) if args.apply else None
    stats = Counter()
    if store and store.get("primary_model") != args.target:
        raise ValueError(
            "--target must match the primary model of this Songe data directory"
        )
    if store:
        with store.connect() as db:
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS omp_seed_binding ON exchanges(binding) WHERE binding GLOB 'omp-seed:*'"
            )
    for path in sorted(directory.rglob("*.jsonl")):
        stats["files"] += 1
        try:
            for identity, when, history, action in turns(
                path, args.provider, args.model, stats
            ):
                stats["matched"] += 1
                if not store:
                    continue
                with store.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    cutoff = db.execute(
                        "SELECT value FROM state WHERE key='forgotten_at'"
                    ).fetchone()
                    if cutoff and when < json.loads(cutoff[0]):
                        stats["forgotten"] += 1
                        continue
                    binding = "omp-seed:" + identity
                    if db.execute(
                        "SELECT 1 FROM exchanges WHERE binding GLOB 'omp-seed:*' AND binding=?",
                        (binding,),
                    ).fetchone():
                        stats["duplicate"] += 1
                        continue
                    db.execute(
                        "INSERT INTO exchanges(created,path,request,response,status,complete,binding,recorded_at) VALUES (?,?,?,?,200,1,?,?)",
                        (
                            when,
                            "/v1/chat/completions",
                            json.dumps({"model": args.target, "messages": history}),
                            json.dumps({"choices": [{"message": action}]}),
                            binding,
                            time.time(),
                        ),
                    )
                    stats["imported"] += 1
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            raise ValueError(f"Cannot read OMP session {path}: {exc}") from exc
    if store:
        refresh(config, store)
    print(
        json.dumps({"mode": "import" if store else "preview", **stats}, sort_keys=True)
    )
