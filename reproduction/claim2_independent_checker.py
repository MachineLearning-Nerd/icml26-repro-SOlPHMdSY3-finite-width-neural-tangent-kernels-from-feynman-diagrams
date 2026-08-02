"""Independent structural checker for a claim-2 certificate.

Unlike the producer, this checker does not import the diagram-rule module.  It
audits only the serialized certificate against the source-level invariants.
"""

from collections import Counter


def check(certificate: dict) -> dict:
    diagrams = certificate.get("diagrams", [])
    names = [item["correction_vertex"]["name"] for item in diagrams]
    fields = Counter(item["correction_vertex"]["field"] for item in diagrams)
    derivative_orders = Counter(
        item["correction_vertex"]["derivative_order"] for item in diagrams
    )
    checks = {
        "five_serialized_diagrams": len(diagrams) == 5,
        "source_vertex_names": sorted(names) == ["D", "F", "K1", "Theta1", "V"],
        "two_preactivation_three_ntk_insertions": fields
        == Counter({"ntk": 3, "preactivation": 2}),
        "derivative_profile_0_2_2_2_4": derivative_orders
        == Counter({2: 3, 0: 1, 4: 1}),
        "unique_ids": len({item["id"] for item in diagrams}) == len(diagrams),
    }
    return {"checks": checks, "passed": all(checks.values())}
