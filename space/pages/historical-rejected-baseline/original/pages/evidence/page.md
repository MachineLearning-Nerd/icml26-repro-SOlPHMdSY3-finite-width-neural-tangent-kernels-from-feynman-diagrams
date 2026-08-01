# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bd64c96ba4e7", "created_at": "2026-07-31T11:08:24+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
{
  "paper": "SOlPHMdSY3",
  "arxiv": "2508.11522",
  "checks": {
    "C0_inf_recursion": {
      "status": "PASS",
      "anchor": "[0,1] infinite-width NTK recursion converges (large-width limit)",
      "precision": "Theta/width by n: {128: np.float64(2.5254), 256: np.float64(2.4141), 512: np.float64(2.4093)}, rel change 0.048"
    },
    "C1_relu_no_diag_correction": {
      "status": "PASS",
      "anchor": "[2] ReLU (scale-invariant) Theta(x,x) has NO finite-width correction",
      "precision": "ReLU Theta/width mean shift 0.0707 (small)"
    },
    "C2_gelu_diag_correction": {
      "status": "PASS",
      "anchor": "[2] GeLU (non-scale-invariant) Theta(x,x) HAS a finite-width correction",
      "precision": "GeLU shift 0.2110 > ReLU 0.0421"
    },
    "C3_offdiag_1_over_n": {
      "status": "PASS",
      "anchor": "[3] off-diagonal finite-width corrections scale as 1/n",
      "precision": "Var(Theta/width) vs n slope -1.20 (~-1)"
    },
    "C4_critical_stability": {
      "status": "PASS",
      "anchor": "[4] gradient stability at critical C_W=2",
      "precision": "critical depth-ratio 8.35 vs high-sigma 75475.05, low 0.0"
    }
  },
  "n_claims_passed": 5,
  "n_claims_total": 5,
  "all_passed": true
}

SUMMARY: 5/5 passed, all_passed=True
```
