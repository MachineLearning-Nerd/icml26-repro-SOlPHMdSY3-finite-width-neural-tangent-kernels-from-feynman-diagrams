# Finite-Width Neural Tangent Kernels — reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/blob/main/notebooks/finite_width_ntk_reproduction.py)

We tested all five judged claims in arXiv:2508.11522v4. The cumulative result is
**VERIFIED evidence status for Claims 1–5**—a research assessment, not a new live
judge score. The strongest paper-scale check sampled 5,000,000 bias-free
four-layer networks for each width (20, 80), jointly evaluating ReLU and
LeakyReLU: diagonal shifts were 0.012–0.057%, with every 99% bound below a
precommitted 1% equivalence margin; off-diagonal controls shifted 1.34–8.30%.

All scientific compute ran on Hugging Face `cpu-upgrade` (estimated/selected 8
cores; actual cgroup quota 8 CPUs, 32 GB RAM contract, no GPU). Claim 4 uses four
representative Figure 2 GeLU widths rather than every plotted marker. Claim 3's
universal theorem is supported by an independent symbolic cancellation as well as
finite source-scale sampling.

Read the [illustrated technical report](reports/reproduction/report.md) or the
[self-contained marimo tutorial](notebooks/finite_width_ntk_reproduction.py).

## Experiment log

Every experiment used the exact command shown below; variants were committed code,
not command-line knobs.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Publication surface | Not run as an experiment (publication surface) | Reader-facing report and notebook | none |
| [audited-five-diagram-baseline](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/audited-five-diagram-baseline) | Exact five-diagram baseline | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claim 2 VERIFIED | HF cpu-upgrade, 8 CPU quota, no GPU |
| [symbolic-rules-and-scale-invariance-proofs](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/symbolic-rules-and-scale-invariance-proofs) | Rules and symbolic cancellation | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claims 1–3 symbolic VERIFIED | HF cpu-upgrade, 8 CPU quota, no GPU |
| [paper-scale-critical-depth-stability](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/paper-scale-critical-depth-stability) | Width 200, depth 30, 1,000 networks/regime | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claim 5 VERIFIED | HF cpu-upgrade, 8 CPU quota, no GPU |
| [source-faithful-gelu-finite-width-correction](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/source-faithful-gelu-finite-width-correction) | Figure 2 GeLU, 100k/width | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claim 4 VERIFIED | HF cpu-upgrade, 8 CPU quota, no GPU |
| [five-million-scale-invariant-cancellation](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/five-million-scale-invariant-cancellation) | Exact ReLU/LeakyReLU cancellation | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claim 3 empirical VERIFIED; cumulative Claims 1–5 pass | HF cpu-upgrade, 8 CPU quota, no GPU; 5,225.662 s scientific runtime |
| [evaluator-visible-cumulative-release](https://github.com/MachineLearning-Nerd/icml26-repro-SOlPHMdSY3-finite-width-neural-tangent-kernels-from-feynman-diagrams/tree/orx/evaluator-visible-cumulative-release) | Independent rerun and HF-generated figures | `uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all` | Claims 1–5 VERIFIED; exact rerun match | HF cpu-upgrade, 8 CPU quota, no GPU; 5,363.686 s scientific runtime |

## Reproduce

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

The locked command is intentionally full-scale and CPU-heavy. Inspect the embedded
results in the report or notebook before deciding to rerun it.
