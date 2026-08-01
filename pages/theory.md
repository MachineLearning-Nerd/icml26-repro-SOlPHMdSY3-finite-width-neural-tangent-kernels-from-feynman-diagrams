# Current theoretical verification

Claims 1–3 are exposed here with exact scope.

- Claim 1 code: `reproduction/claim1_verifier.py`; contract and assumptions: `.openresearch/artifacts/claim1/claim_contract.json`; scope limitation: the full `F` derivation plus `D/F/A/B` signature coverage, not an unclaimed regeneration of every appendix expression.
- Claim 2 code: `reproduction/claim2_verifier.py`; raw baseline result: `.openresearch/artifacts/claim2/raw/baseline_run.json`; required failing control: `reproduction/claim2_negative_control.py`.
- Claim 3 code: `reproduction/claim3_verifier.py`; independent quadrature: `reproduction/claim3_independent_checker.py`; required failing GeLU control: `reproduction/claim3_negative_control.py`; exact quantifier and assumptions: `.openresearch/artifacts/claim3/claim_contract.json`.

The completed symbolic parent run VERIFIED Claims 1–3 under their exact theorem scopes. This cumulative child adds the source's exact five-million-initialization ReLU and LeakyReLU experiment at widths 20 and 80. Its current empirical reviewer verdict is **BLOCKED pending execution** on Hugging Face `cpu-upgrade`. The historical judged Space verifier is not the current verifier.
