# Verification run


---
<!-- trackio-cell
{"type": "code", "id": "cell_618b25c6036b", "created_at": "2026-07-31T11:09:09+00:00", "title": "verify all claims", "command": [".venv/bin/python", "repro/src/verify.py"], "exit_code": 0, "duration_s": 44.009}
-->
````bash
$ .venv/bin/python repro/src/verify.py
````

exit 0 · 44.0s


````python title=verify.py
"""verify.py - 5 anchored claims for SOlPHMdSY3 (arXiv 2508.11522)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import core as C
OUT=os.path.join(os.path.dirname(__file__),"..","outputs"); os.makedirs(OUT,exist_ok=True)
v={"paper":"SOlPHMdSY3","arxiv":"2508.11522","checks":{}}
r=C.claim0_inf_recursion(); v["checks"]["C0_inf_recursion"]={"status":"PASS" if r["passed"] else "FAIL",
 "anchor":"[0,1] infinite-width NTK recursion converges (large-width limit)","precision":f"Theta/width by n: {r['theta_per_width']}, rel change {r['relative_change']:.3f}"}
r=C.claim2_relu_no_bias(); v["checks"]["C1_relu_no_diag_correction"]={"status":"PASS" if r["passed"] else "FAIL",
 "anchor":"[2] ReLU (scale-invariant) Theta(x,x) has NO finite-width correction","precision":f"ReLU Theta/width mean shift {r['relu_mean_relative_shift']:.4f} (small)"}
r=C.claim2_gelu_bias(); v["checks"]["C2_gelu_diag_correction"]={"status":"PASS" if r["passed"] else "FAIL",
 "anchor":"[2] GeLU (non-scale-invariant) Theta(x,x) HAS a finite-width correction","precision":f"GeLU shift {r['gelu_mean_relative_shift']:.4f} > ReLU {r['relu_mean_relative_shift']:.4f}"}
r=C.claim3_offdiag_1_over_n(); v["checks"]["C3_offdiag_1_over_n"]={"status":"PASS" if r["passed"] else "FAIL",
 "anchor":"[3] off-diagonal finite-width corrections scale as 1/n","precision":f"Var(Theta/width) vs n slope {r['loglog_slope']:.2f} (~-1)"}
r=C.claim4_critical_stability(); v["checks"]["C4_critical_stability"]={"status":"PASS" if r["passed"] else "FAIL",
 "anchor":"[4] gradient stability at critical C_W=2","precision":f"critical depth-ratio {r['critical_depth_ratio']} vs high-sigma {r['high_sigma_ratio']}, low {r['low_sigma_ratio']}"}
v["n_claims_passed"]=sum(1 for c in v["checks"].values() if c["status"]=="PASS"); v["n_claims_total"]=5
v["all_passed"]=all(c["status"]=="PASS" for c in v["checks"].values())
json.dump(v,open(os.path.join(OUT,"verdict.json"),"w"),indent=2)
print(json.dumps(v,indent=2)); print(f"\nSUMMARY: {v['n_claims_passed']}/{v['n_claims_total']} passed, all_passed={v['all_passed']}")

````


````output
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

````
