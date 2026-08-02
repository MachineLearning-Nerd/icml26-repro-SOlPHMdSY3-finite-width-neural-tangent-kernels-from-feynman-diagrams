"""Primitive order-1/n rules for the NTK-mean two-point function.

The data model is deliberately small: an outer cubic interaction exposes an
internal insertion slot, and an order-1/n correction vertex can fill it only
when its field, derivative order, and external NTK-line count match.  These are
the rules stated in Section 4 and applied in Section 5.1 of arXiv:2508.11522v4.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction


@dataclass(frozen=True)
class OuterVertex:
    name: str
    insertion_field: str
    derivative_orders: tuple[int, ...]


@dataclass(frozen=True)
class CorrectionVertex:
    name: str
    field: str
    derivative_order: int
    external_ntk_lines: int
    coefficient: Fraction
    algebraic_term: str


OUTER_VERTICES = (
    OuterVertex("sigma_prime_pair", "ntk", (0, 2)),
    OuterVertex("delta_omega", "preactivation", (2, 4)),
)

CORRECTION_VERTICES = (
    CorrectionVertex(
        "Theta1", "ntk", 0, 2, Fraction(1),
        "Cw*Theta1*E[sigma1_prime*sigma2_prime]",
    ),
    CorrectionVertex(
        "K1", "preactivation", 2, 0, Fraction(1, 2),
        "sum(K1[b1,b2]*E[d_b1_b2 DeltaOmega12])",
    ),
    CorrectionVertex(
        "V", "preactivation", 4, 0, Fraction(1, 8),
        "sum(V[(b1,b2),(b3,b4)]*E[d_b1_b2_b3_b4 DeltaOmega12])",
    ),
    CorrectionVertex(
        "D", "ntk", 2, 2, Fraction(1, 2),
        "Cw*sum(E[d_b1_b2 sigma1_prime*sigma2_prime]*D[b1,b2,1,2])",
    ),
    CorrectionVertex(
        "F", "ntk", 2, 2, Fraction(1),
        "Cw*sum(E[d_b1_b2 sigma1_prime*sigma2_prime]*F[b1,1,b2,2])",
    ),
)


def enumerate_diagrams(drop_vertex: str | None = None) -> list[dict]:
    """Enumerate every compatible diagram from primitive slot constraints."""
    diagrams = []
    for outer in OUTER_VERTICES:
        for correction in CORRECTION_VERTICES:
            if correction.name == drop_vertex:
                continue
            if correction.field != outer.insertion_field:
                continue
            if correction.derivative_order not in outer.derivative_orders:
                continue
            if correction.field == "ntk" and correction.external_ntk_lines != 2:
                continue
            if correction.field == "preactivation" and correction.external_ntk_lines != 0:
                continue
            diagrams.append(
                {
                    "id": f"{outer.name}:{correction.name}",
                    "outer_vertex": asdict(outer),
                    "correction_vertex": {
                        **asdict(correction),
                        "coefficient": str(correction.coefficient),
                    },
                }
            )
    return sorted(diagrams, key=lambda item: item["id"])
