"""Expected failure: inject a two-percent diagonal finite-width correction."""

import json

from reproduction.claim3_paper_scale import verify_rows


def main():
    rows = []
    for activation in ("ReLU", "LeakyReLU(alpha=0.1)"):
        for width in (20, 80):
            rows.append(
                {
                    "activation": activation,
                    "width": width,
                    "mean": [1.02, 0.90],
                    "standard_error": [0.001, 0.001],
                    "infinite_ntk": [1.0, 1.0],
                }
            )
    result = verify_rows(rows)
    output = {
        "mutation": "inject a two-percent diagonal finite-width correction",
        "expected": "equivalence verifier rejects every mutated diagonal",
        "verifier_passed": result["passed"],
        "checks": result["checks"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
