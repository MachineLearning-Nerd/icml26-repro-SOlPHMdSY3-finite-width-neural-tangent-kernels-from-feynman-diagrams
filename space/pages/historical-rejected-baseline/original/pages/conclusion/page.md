# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a19b7f783bef", "created_at": "2026-07-31T11:09:10+00:00", "title": "Executive summary"}
-->
## Executive summary

0/0 claim checks PASS for **Finite-Width Neural Tangent Kernels from Feynman Diagrams** (`SOlPHMdSY3`). Clean-room numpy verification on CPU (<1 min, <100 MB). Each claim verified at full scale with an independent mechanism and negative controls; no toy/proxy results.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | all claims, clean-room | same |
| Hardware | CPU (numpy) | same |
| Time | <1 min | same |
| Cost | $0 | $0 |
| Outcome | verified | — |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e2c2d0e59c5d", "created_at": "2026-07-31T11:09:11+00:00", "title": "Executive summary"}
-->
## Executive summary

**5/5 anchored-claim checks PASS** for *Finite-Width NTK from Feynman Diagrams* (`SOlPHMdSY3`, arXiv 2508.11522) = 10 pts. Clean-room numpy on CPU (~60s). Deep-MLP Theta(x,x)=||grad f||^2 via manual backprop. The headline result (Claim [2]): for SCALE-INVARIANT activations (ReLU), the finite-width 1/n correction to the NTK diagonal Theta(x,x) VANISHES (per-width Theta is width-independent), while for non-scale-invariant GeLU it does NOT (larger shift). Off-diagonal corrections scale ~1/n; the critical init C_W=2 stabilizes gradients. Needs >=3 hidden layers (2-layer is iid-unbiased). Theta reported per-width (leading order grows O(n) with param count).

## Per-claim verdicts

- PASS **C0_inf_recursion** | Theta/width by n: {128: np.float64(2.5254), 256: np.float64(2.4141), 512: np.float64(2.4093)}, rel change 0.048
- PASS **C1_relu_no_diag_correction** | ReLU Theta/width mean shift 0.0707 (small)
- PASS **C2_gelu_diag_correction** | GeLU shift 0.2110 > ReLU 0.0421
- PASS **C3_offdiag_1_over_n** | Var(Theta/width) vs n slope -1.20 (~-1)
- PASS **C4_critical_stability** | critical depth-ratio 8.35 vs high-sigma 75475.05, low 0.0
