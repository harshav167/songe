from typing import Protocol


class Upstream(Protocol):
    async def load(self, version: dict) -> None: ...
    async def unload(self, version: dict) -> None: ...
    async def close(self) -> None: ...


def create(config, root):
    if config.upstream == "lmstudio":
        from ..log import Store
        from .lmstudio import LMStudio

        return LMStudio(config, store=Store(root))
    if config.upstream == "ollama":
        from ..log import Store
        from .ollama import Ollama

        return Ollama(config, store=Store(root))
    if config.upstream == "llamacpp":
        from .llamacpp import LlamaCPP
        from .supervisor import LlamaServer

        if not getattr(config, "model", None):
            return LlamaCPP(config)
        supervisor = LlamaServer(config, root)
        upstream = LlamaCPP(
            config,
            restart=supervisor.restart,
            stop=supervisor.stop,
            memory_probe=supervisor.memory_bytes,
        )
        upstream.supervisor = supervisor
        return upstream
    if config.upstream == "sglang":
        from .sglang import SGLang

        return SGLang(config)
    if config.upstream == "vllm":
        from .vllm import VLLM

        if not getattr(config, "server_command", None):
            return VLLM(config)
        from .supervisor import VLLMServer

        supervisor = VLLMServer(config, root)
        upstream = VLLM(config, memory_probe=supervisor.memory_bytes)
        upstream.supervisor = supervisor
        return upstream
    if config.upstream == "mlx":
        from .mlx import MLX

        return MLX(config, root)
    raise ValueError(f"Unknown upstream: {config.upstream}")
