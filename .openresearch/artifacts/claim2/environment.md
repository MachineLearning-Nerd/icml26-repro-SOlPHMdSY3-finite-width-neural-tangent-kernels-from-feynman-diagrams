# Environment and command

Fixed command:

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

Python is constrained to 3.11 and all packages are resolved in `uv.lock`. `jax[cpu]` excludes CUDA extras. The selected Hugging Face flavor is `cpu-upgrade` (8 vCPU, 32 GB RAM, no accelerator). The run prints actual CPU affinity, `os.cpu_count()`, runtime, Git SHA, and accelerator-presence checks.
