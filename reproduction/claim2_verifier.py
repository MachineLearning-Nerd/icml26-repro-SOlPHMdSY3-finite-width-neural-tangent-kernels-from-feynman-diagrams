"""Executable verifier for the five-diagram NTK-mean recursion claim."""

import hashlib
import json

from reproduction.diagram_rules import enumerate_diagrams


EXPECTED_IDS = {
    "delta_omega:K1",
    "delta_omega:V",
    "sigma_prime_pair:D",
    "sigma_prime_pair:F",
    "sigma_prime_pair:Theta1",
}
EXPECTED_COEFFICIENTS = {
    "delta_omega:K1": "1/2",
    "delta_omega:V": "1/8",
    "sigma_prime_pair:D": "1/2",
    "sigma_prime_pair:F": "1",
    "sigma_prime_pair:Theta1": "1",
}


def verify(drop_vertex: str | None = None) -> dict:
    diagrams = enumerate_diagrams(drop_vertex=drop_vertex)
    ids = {diagram["id"] for diagram in diagrams}
    coefficients = {
        diagram["id"]: diagram["correction_vertex"]["coefficient"]
        for diagram in diagrams
    }
    checks = {
        "exactly_five_diagrams": len(diagrams) == 5,
        "complete_unique_topologies": ids == EXPECTED_IDS,
        "paper_coefficients_match": coefficients == EXPECTED_COEFFICIENTS,
        "two_quadratic_vertices": sum(
            name.endswith(("Theta1", "K1")) for name in ids
        ) == 2,
        "three_quartic_vertices": sum(
            name.endswith(("V", "D", "F")) for name in ids
        ) == 3,
    }
    certificate = {
        "claim": "Section 5.1 order-1/n NTK-mean recursion has exactly five diagrams",
        "source": {
            "arxiv": "2508.11522v4",
            "html_anchor": "S5.SS1",
            "equation_anchor": "S5.E12",
            "source_sha256": "5237b9fec2f128b23266771acbc8e44837619d5309cd32840ce537071b64c47f",
        },
        "diagrams": diagrams,
        "checks": checks,
        "passed": all(checks.values()),
    }
    canonical = json.dumps(certificate, sort_keys=True, separators=(",", ":"))
    certificate["certificate_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return certificate


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
