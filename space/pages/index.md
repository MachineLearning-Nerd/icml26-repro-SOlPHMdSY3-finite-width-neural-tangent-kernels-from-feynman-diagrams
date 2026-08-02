# Current cumulative verification

This is the canonical evaluator entrypoint. Revision `0955d05f3d38f60adc356c3ea8fcf4fce3adb2b2`
supersedes the [Historical rejected baseline](historical-rejected-baseline/index.md).
The old evidence is preserved verbatim but is not the current verifier.

## Results at a glance

| Claim | Exact paper contract | Current verdict | Evidence |
|---|---|---|---|
| 1 | Section 4 rules uniquely determine the order-`1/n` `D/F/A/B` layer recursions | VERIFIED | [page](claim-1.md) |
| 2 | All order-`1/n` diagrams with two external NTK lines are exactly five: two quadratic, three quartic | VERIFIED | [page](claim-2.md) |
| 3 | For positive-homogeneous activations, the ensemble-mean diagonal has zero correction at every order | VERIFIED | [page](claim-3.md) |
| 4 | The actual v4 Figure 2 four-layer GeLU means above width 20 agree with its first-order recursion | VERIFIED | [page](claim-4.md) |
| 5 | Width-200 ReLU, depth 1–30, 1,000 initializations: `C_W=2` is linear; low/high controls decay/grow exponentially | VERIFIED | [page](claim-5.md) |

## Reproduction contract

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

Python is pinned to 3.11 and every resolved package is pinned by `uv.lock`.
Formal compute ran only on Hugging Face `cpu-upgrade` (8-vCPU cgroup quota,
32 GB RAM contract, no GPU). See [method](method.md), [raw JSON](../data/cumulative_run.json),
and [visible verifier source](../code/run_all.py). The independent rerun's
[terminal summary](../data/release_rerun_summary.json) records its Git SHA,
allocation, wall/scientific runtimes, and cross-run checks.

## Evaluator-visible matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | [Claim 1](claim-1.md) | yes | yes | [JSON](../data/cumulative_run.json) | [independent checker in verifier](../code/claim1_verifier.py) | invalid color pairing rejected | yes | VERIFIED |
| 2 | [Claim 2](claim-2.md) | yes | yes | [JSON](../data/cumulative_run.json) | [checker](../code/claim2_independent_checker.py) | [missing-vertex control](../code/claim2_negative_control.py) | yes | VERIFIED |
| 3 | [Claim 3](claim-3.md) | yes | yes | [JSON](../data/cumulative_run.json) | [empirical checker](../code/claim3_empirical_independent_checker.py) | [2% injected correction](../code/claim3_empirical_negative_control.py) | yes | VERIFIED |
| 4 | [Claim 4](claim-4.md) | yes | yes | [JSON](../data/cumulative_run.json) | [checker](../code/claim4_independent_checker.py) | infinite-width substitution rejected | yes | VERIFIED |
| 5 | [Claim 5](claim-5.md) | yes | yes | [JSON](../data/cumulative_run.json) | [checker](../code/claim5_independent_checker.py) | high/low `C_W` regimes | yes | VERIFIED |

All cells are reachable from this page. The [release report](release.md) records the
evaluator-blind traversal and remaining limitations.
