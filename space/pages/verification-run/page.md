# Current verification run

The current verifier is [`code/run_all.py`](../../code/run_all.py), executed by:

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

See the [canonical index](../index.md). The previous weak run is preserved as
[Historical rejected baseline](../historical-rejected-baseline/index.md).
