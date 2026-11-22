# SGLang upstream

Run Songe on the SGLang host. Both processes need access to the same base checkpoint and adapter paths. No particular coding agent is required. Configure your existing Songe clients as usual.

1. Set `BASE` to the absolute path to the trainable safetensors checkpoint. Export `SGLANG_ADMIN_API_KEY` to **both** SGLang and the Songe service. Do not distribute this key to clients. Leave inference loopback-only; the integration authenticates adapter management, not general inference.
2. In the SGLang environment, start the model with LoRA enabled. Leave its served name at the model path so Songe can match the checkpoint. The configured rank and PEFT target modules must fit the model and server.

   ```sh
   python -m sglang.launch_server --model-path "$BASE" --host 127.0.0.1 --port 30000 \
     --enable-lora --max-lora-rank 64 --lora-target-modules all
   ```

3. In the Songe environment with accelerator-appropriate Torch installed:

   ```sh
   uv run songe init --upstream sglang --upstream-url http://127.0.0.1:30000/v1 --model "$BASE" --keep-default
   ```

Songe exports accepted candidates as PEFT adapters. It loads them through SGLang's admin API, then routes chat as `base:adapter-name`. It leaves the external base and other users' adapters resident. Reserve GPU memory for training (for example, reduce SGLang's `--mem-fraction-static`), or provide separate training capacity. Use `songe status` for cycle progress and `songe undo` to return to a previous version.
