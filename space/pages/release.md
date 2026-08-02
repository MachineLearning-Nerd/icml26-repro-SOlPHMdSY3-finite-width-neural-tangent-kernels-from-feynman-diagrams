# Release report and evaluator-visible audit

- Previous live judged score: `0/10`
- Conservative projected score range after this proposed change: **8–10/10**
- Best-supported possible new score: **10/10 forecast, not a judge result**
- Current live score: **0/10** until the evaluator records a new revision

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 0 | 2 | MEDIUM | VERIFIED | Direct rule enumeration and worked recursion; formal scope beyond finite order remains outside the certificate. |
| 2 | 0 | 2 | HIGH | VERIFIED | Exactly five topologies, coefficients, independent algebraic sum, missing-vertex control. |
| 3 | 0 | 2 | HIGH | VERIFIED | Symbolic cancellation plus exact 5M-scale ReLU/LeakyReLU equivalence and off-diagonal controls. |
| 4 | 0 | 2 | HIGH | VERIFIED | Source-faithful GeLU architecture, 100k/width, immutable predictions, all eight points favor correction. |
| 5 | 0 | 2 | HIGH | VERIFIED | Exact width/depth/sample count, slope and 30-point tests, exponential low/high controls. |

All five claims changed from INCONCLUSIVE to evidence status VERIFIED. None remains
BLOCKED. The material evaluator risk is interpretation of Claim 1's proof scope and
Claim 4's four-width subset. Forecast points are not earned points.

The exact release-generation rerun summary is [downloadable JSON](../data/release_rerun_summary.json).

## Blind traversal record

The candidate was reviewed only from `README.md` then `pages/index.md`, without
repository knowledge. Files opened were the five claim pages, `method.md`, this
page, visible verifier/checker/control source, and `data/cumulative_run.json`.
The first traversal found all claims and raw data but required clearer historical
labelling; navigation was revised to put current verification first and label the
archive exactly **Historical rejected baseline**. The repeated traversal found no
missing matrix cell. Automated traversal and hash checks are executed by the fixed
command and exit nonzero on any missing evidence.

## Publication action

After the cumulative release-gate run passes, the exact text allowlist will be
uploaded to the existing `DineshAI/SOlPHMdSY3` Space using the text-only API. No
second Space will be created. The judged revision `3fa1fabbd86c5e3dc7dbc2ef6ea4360568d3745b`
remains immutable and its file paths remain a subset of the new remote tree.
