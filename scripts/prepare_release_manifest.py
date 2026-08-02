"""Create the exact text-only Space upload allowlist."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    rows = []
    for path in sorted(Path("space").rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "source": str(path),
                "destination": str(path.relative_to("space")),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    output = {
        "schema_version": 1,
        "target": "DineshAI/SOlPHMdSY3",
        "base_judged_revision": "3fa1fabbd86c5e3dc7dbc2ef6ea4360568d3745b",
        "policy": "text-only additive upload; unspecified existing files are preserved",
        "files": rows,
    }
    Path("release").mkdir(exist_ok=True)
    Path("release/space_upload_allowlist.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
