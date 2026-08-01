# Finite-Width NTK reproduction

This branch is the immutable audited baseline for arXiv:2508.11522v4.

The fixed reproduction command is:

```bash
uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all
```

The baseline implements the exact Section 5.1 contract: the order-`1/n`
correction to the NTK mean is the sum of five admissible Feynman diagrams.
See [the canonical baseline page](pages/baseline.md) and the durable evidence
under `.openresearch/artifacts/claim2/`.

This is not the publication branch. The judged Hugging Face evidence remains
protected and is not replaced by this baseline.
