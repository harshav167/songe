import json

from songe.cli import main
from songe.config import Config
from songe.log import Store


def test_preview_and_import_only_matching_branches(tmp_path, capsys):
    root, sessions = tmp_path / "songe", tmp_path / "sessions"
    sessions.mkdir()
    rows = [{"type": "title"}, {"type": "session", "id": "session"}]

    def append(key, parent, role, content, **kwargs):
        rows.append(
            {
                "type": "message",
                "id": key,
                "parentId": parent,
                "timestamp": "2026-08-03T12:00:00Z",
                "message": {"role": role, "content": content, **kwargs},
            }
        )

    append("1", None, "user", "Which version is active?")
    append(
        "2",
        "1",
        "assistant",
        [
            {
                "type": "toolCall",
                "id": "call",
                "name": "read",
                "arguments": {"path": "version"},
            }
        ],
        provider="local",
        model="base",
    )
    append("3", "2", "toolResult", "2.0", toolCallId="call")
    append("4", "3", "assistant", "Version 2.0.", provider="local", model="base")
    append("5", "1", "assistant", "foreign branch", provider="other", model="base")
    (sessions / "example.jsonl").write_text(
        "\n".join(map(json.dumps, rows)), encoding="utf-8"
    )
    args = [
        "--data-dir",
        str(root),
        "omp-backfill",
        str(sessions),
        "--model",
        "base",
        "--provider",
        "local",
        "--target",
        "base",
    ]
    assert main(args) == 0 and not root.exists()
    assert json.loads(capsys.readouterr().out)["matched"] == 2
    config = Config(model=str(tmp_path), base_identifier="base")
    config.save(root)
    store = Store(root)
    store.set("primary_model", "base")
    assert main(args + ["--apply"]) == 0
    assert main(args + ["--apply"]) == 0
    first, second = store.exchanges()
    assert (
        first["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "read"
    )
    assert second["request"]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call",
        "content": "2.0",
    }
    assert second["response"]["choices"][0]["message"]["content"] == "Version 2.0."
    assert {s["action"]["role"] for s in store.active_steps()} == {"assistant"}
