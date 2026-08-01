"""Mutation control: dropping the F quartic vertex must fail verification."""

import json

from reproduction.claim2_verifier import verify


def main() -> int:
    result = verify(drop_vertex="F")
    output = {
        "mutation": "drop quartic F vertex",
        "expected": "verifier exits nonzero because only four diagrams remain",
        "observed_diagram_count": len(result["diagrams"]),
        "verifier_passed": result["passed"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
