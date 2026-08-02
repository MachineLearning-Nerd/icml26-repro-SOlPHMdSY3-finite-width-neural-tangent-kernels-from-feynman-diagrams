---
title: "Finite-width NTK: claim-by-claim reproduction"
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-SOlPHMdSY3
---

# Finite-width NTK: claim-by-claim reproduction

**Current verification supersedes the Historical rejected baseline.** Start with the
[canonical evidence index](pages/index.md). Every claim below is backed by executable
source, inline numbers, raw JSON, an independent checker, and a negative control.

Paper: [Finite-Width Neural Tangent Kernels from Feynman Diagrams](https://arxiv.org/abs/2508.11522), v4. Source archive retrieved 2026-08-01 with an explicit User-Agent; SHA-256 `5237b9fec2f128b23266771acbc8e44837619d5309cd32840ce537071b64c47f`.

| Claim | Verdict | Canonical evidence |
|---|---|---|
| 1. Section 4 graphical rules determine the order-`1/n` layer recursions | VERIFIED | [Claim 1](pages/claim-1.md) |
| 2. The first NTK-mean correction contains exactly five diagrams | VERIFIED | [Claim 2](pages/claim-2.md) |
| 3. Scale-invariant ReLU/LeakyReLU diagonal means have no finite-width correction | VERIFIED | [Claim 3](pages/claim-3.md) |
| 4. Source Figure 2 GeLU data agree with the first-order correction | VERIFIED | [Claim 4](pages/claim-4.md) |
| 5. Critical `C_W=2` NTKs scale linearly through depth 30 | VERIFIED | [Claim 5](pages/claim-5.md) |

Fixed command:

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

The cumulative evidence run used Hugging Face `cpu-upgrade`: estimated 8 cores;
actual cgroup quota 8 CPUs, 32 GB flavor contract, no GPU; scientific runtime
5,225.662 s. Its exact raw output is [downloadable JSON](data/cumulative_run.json).
The independent release rerun at Git SHA `0955d05f3d38f60adc356c3ea8fcf4fce3adb2b2`
used the same allocation and command, took 5,363.686 s of scientific runtime, and
matched Claims 3–5 exactly (`max combined-SE z = 0.0`, threshold 5.0).
[Download its terminal summary](data/release_rerun_summary.json).

[Method, environment, and limitations](pages/method.md) ·
[Release report and visibility matrix](pages/release.md) ·
[Historical rejected baseline](pages/historical-rejected-baseline/index.md)
