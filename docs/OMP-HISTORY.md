# Seed a learner from OMP session files

The importer works with existing OMP JSONL sessions; it needs neither an OMP plugin nor a particular inference engine. Use a dedicated single-model Songe data directory. Confirm the historical model ID really referred to the checkpoint you plan to train; a matching label is not proof of identical weights.

```sh
uv run songe omp-backfill ~/.omp/agent/sessions \
  --provider "$OLD_PROVIDER" --model "$OLD_MODEL_ID" --target "$SONGE_PRIMARY_MODEL"
```

Preview is the default and writes nothing. Use the exact `provider` and `model` from historical assistant messages. For named profiles point to the profile's sessions directory. To import on a different host, privately copy only files you are willing to disclose and run the command there with `--apply`. The importer filters locally, not during transfer; session files may contain code, credentials and tool results.

The importer follows message parents across branches, keeps matching assistant text and tool calls with their preceding user/tool context, and breaks continuity at another model, compaction, or unsupported content. It does not reconstruct missing system prompts, tool schemas, images, or private reasoning. Import is repeatable and respects the forget cutoff. Errors identify the affected file; fix or exclude it and rerun. No historical tool call is executed.

Once imported, `songe sleep --now` requests the normal gated training cycle on the initialized service. `learning = "dreams"` learns from practice examples derived from feedback; `learning = "hindsight"` uses reconstructed real steps. Neither mode guarantees enough evidence or an accepted LoRA. Check `songe status` for the result.
