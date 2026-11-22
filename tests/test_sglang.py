import json

import httpx
import pytest

from songe.config import Config
from songe.upstream.sglang import SGLang


async def test_sglang_routes_and_releases_only_its_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_ADMIN_API_KEY", "test-key")
    cards = [
        {"id": "base", "root": str(tmp_path)},
        {"id": "external", "root": "/external", "parent": "base"},
    ]

    def server(request):
        if request.url.path == "/v1/models":
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"data": cards})
        assert request.headers["authorization"] == "Bearer test-key"
        data = json.loads(request.content)
        if request.url.path == "/load_lora_adapter":
            cards.append(
                {"id": data["lora_name"], "parent": "base", "root": data["lora_path"]}
            )
        else:
            assert request.url.path == "/unload_lora_adapter"
            cards.remove(next(c for c in cards if c["id"] == data["lora_name"]))
        return httpx.Response(200, json={"success": True})

    config = Config(upstream="sglang", base_identifier="base")
    upstream = SGLang(config, transport=httpx.MockTransport(server))
    version = {
        "id": "night-1",
        "identifier": "songe/demo/night-1",
        "kind": "peft-adapter",
        "artifact": str(tmp_path),
        "serving": True,
    }
    await upstream.ensure(version)
    assert await upstream.resident(version)
    assert upstream.request_model(version) == "base:songe/demo/night-1"
    with pytest.raises(RuntimeError, match="other weights"):
        await upstream.ensure(version | {"artifact": str(tmp_path / "wrong")})
    await upstream.close()
    assert [card["id"] for card in cards] == ["base", "external"]
