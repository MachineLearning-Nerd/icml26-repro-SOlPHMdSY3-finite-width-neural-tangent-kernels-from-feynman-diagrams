"""Control: GeLU violates the scale-invariance invariant and must fail."""

import json

from reproduction.claim3_independent_checker import check


def main() -> int:
    result = check()
    values = result["gelu_control"]["E_sigma_prime2"]
    falsely_scale_invariant = max(values) - min(values) < 1e-12
    output = {
        "mutation": "treat GeLU as scale invariant",
        "expected": "failure because E[sigma'(Z)^2] depends on K",
        "K": result["gelu_control"]["K"],
        "E_sigma_prime2": values,
        "mutant_passed": falsely_scale_invariant,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if falsely_scale_invariant else 1


if __name__ == "__main__":
    raise SystemExit(main())
