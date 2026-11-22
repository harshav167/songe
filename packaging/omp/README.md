# Songe provider for OMP

From the Songe source checkout on the machine running OMP:

```sh
omp plugin install ./packaging/omp
omp models songe
```

Select the model listed under the `songe` provider. The extension reads Songe's `/v1/models` catalog and uses OpenAI chat completions. It does not change the default model, read past sessions, or manage the inference server. It works with any configured Songe upstream.

If Songe runs on another machine, open an SSH tunnel first and keep it open:

```sh
ssh -N -L 127.0.0.1:7613:127.0.0.1:7613 user@songe-host
```

`SONGE_PORT` overrides the local port (default `7613`). Context limits come from the catalog when available; older Songe catalogs default to 8192 tokens. Run `omp plugin uninstall omp-songe` to remove the plugin.
