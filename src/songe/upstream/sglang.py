"""Dynamic PEFT adapters on an externally managed SGLang runtime."""

import os

from ..models import discovery
from .vllm import VLLM


class SGLang(VLLM):
    name = "sglang"
    admin_prefix = ""

    def admin_headers(self):
        key = os.environ.get("SGLANG_ADMIN_API_KEY")
        if not key:
            raise ValueError(
                "Set SGLANG_ADMIN_API_KEY in the Songe service environment"
            )
        return {"Authorization": f"Bearer {key}"}

    def discover(self):
        return discovery.sglang(self.url, self.transport)

    def request_model(self, version):
        if version.get("kind") == "peft-adapter":
            return f"{self.config.base_identifier}:{version['identifier']}"
        return super().request_model(version)
