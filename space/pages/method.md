# Method, environment, and deviations

## Source and implementation

Paper v4 was retrieved 2026-08-01 with an explicit browser User-Agent. HTML
SHA-256 is `62cc33fa20ebd0d07a64a0386b96318a3ae6341880c031689c139d6c8ffe409a`;
source archive SHA-256 is `5237b9fec2f128b23266771acbc8e44837619d5309cd32840ce537071b64c47f`.
The implementation was reconstructed independently from the equations and audited
against author code commit `22f40289` where useful.

The fixed command for every experiment node is:

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

One repository `.venv`, Python 3.11, `pyproject.toml`, and `uv.lock` are reused.
All scientific compute ran on Hugging Face `cpu-upgrade`; estimated and selected
core count was 8, actual cgroup quota was 8 CPUs, RAM contract 32 GB, and no GPU
was present. Git SHA, seeds, runtime, allocation, complete outputs, checker output,
and controls are in [raw JSON](../data/cumulative_run.json).

## Non-circularity

Claim 3's 1% equivalence margin, 4-SE diagonal limit, and off-diagonal 5-SE/1%
control were committed before sampling. Claim 4 compares fresh samples against
immutable PDF-extracted curves and rejects an infinite-width substitution. Claim 5
uses the analytical critical slope as a prediction, while low/high controls must
fail for their intended exponential reason. No sample size or prediction was fit
to the reproduction output.

## Deviations

- Claim 4 uses four representative Figure 2 widths rather than every plotted width,
  but retains the paper's 100,000 samples at every tested width.
- Claim 3 includes both the proof-level symbolic reconstruction and a finite
  source-scale experiment; the experiment alone would only corroborate the theorem.
- Trace averages use two unbiased Rademacher Hutchinson probes per network, as
  independently calibrated against exact Jacobians.
