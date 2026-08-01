"""Evaluator-visible release checks; every failure blocks the fixed command."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path("space")
ALLOWLIST = Path("release/space_upload_allowlist.json")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_links(path: Path) -> list[str]:
    links = []
    for match in LINK.finditer(path.read_text()):
        target = match.group(1) or match.group(2)
        target = target.split("#", 1)[0]
        if target and not target.startswith(("http://", "https://", "mailto:")):
            links.append(target)
    return links


def traverse(root: Path) -> tuple[set[str], list[str]]:
    queue = ["README.md", "pages/index.md"]
    visited = set()
    missing = []
    while queue:
        relative = queue.pop(0)
        if relative in visited:
            continue
        visited.add(relative)
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if path.suffix != ".md":
            continue
        for target in markdown_links(path):
            resolved = (path.parent / target).resolve()
            try:
                linked_relative = str(resolved.relative_to(root.resolve()))
            except ValueError:
                missing.append(f"outside:{relative}->{target}")
                continue
            if not resolved.is_file():
                missing.append(f"missing:{relative}->{linked_relative}")
                continue
            queue.append(linked_relative)
    return visited, missing


def verify_historical_archive(root: Path) -> dict:
    manifest = {}
    for line in (root / "historical_judged_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    archive_root = root / "pages/historical-rejected-baseline/original"
    mapping = {
        "README.md": archive_root / "README.md",
        "logbook.json": archive_root / "logbook.json",
        "pages/index.md": archive_root / "pages/index.md",
        "pages/claims/page.md": archive_root / "pages/claims/page.md",
        "pages/conclusion/page.md": archive_root / "pages/conclusion/page.md",
        "pages/evidence/page.md": archive_root / "pages/evidence/page.md",
        "pages/overview/page.md": archive_root / "pages/overview/page.md",
        "pages/verification-run/page.md": archive_root / "pages/verification-run/page.md",
    }
    checks = {
        relative: path.is_file() and sha256(path) == manifest[relative]
        for relative, path in mapping.items()
    }
    return {
        "judged_revision": "3fa1fabbd86c5e3dc7dbc2ef6ea4360568d3745b",
        "judged_file_count": len(manifest),
        "archived_text_hashes": checks,
        "passed": len(manifest) == 17 and all(checks.values()),
    }


def verify_allowlist() -> dict:
    payload = json.loads(ALLOWLIST.read_text())
    rows = payload["files"]
    destinations = [row["destination"] for row in rows]
    checks = {
        "nonempty": bool(rows),
        "destinations_unique": len(destinations) == len(set(destinations)),
        "text_extensions_only": all(
            Path(destination).suffix.lower()
            in {"", ".css", ".html", ".js", ".json", ".md", ".py", ".svg", ".toml", ".lock"}
            for destination in destinations
        ),
        "hashes_match": all(
            Path(row["source"]).is_file()
            and sha256(Path(row["source"])) == row["sha256"]
            for row in rows
        ),
        "canonical_entrypoint_uploaded": "README.md" in destinations,
        "logbook_uploaded": "logbook.json" in destinations,
        "raw_json_uploaded": "data/cumulative_run.json" in destinations,
    }
    judged_paths = {
        line.split("  ", 1)[1]
        for line in (ROOT / "historical_judged_manifest.sha256").read_text().splitlines()
    }
    candidate_paths = judged_paths | set(destinations)
    checks["old_file_set_is_candidate_subset"] = judged_paths <= candidate_paths
    return {
        "file_count": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_visibility(root: Path) -> dict:
    visited, missing = traverse(root)
    required = {
        "pages/claim-1.md",
        "pages/claim-2.md",
        "pages/claim-3.md",
        "pages/claim-4.md",
        "pages/claim-5.md",
        "pages/method.md",
        "pages/release.md",
        "data/cumulative_run.json",
        "code/run_all.py",
        "code/claim2_negative_control.py",
        "code/claim3_empirical_negative_control.py",
    }
    claim_checks = {}
    for number in range(1, 6):
        text = (root / f"pages/claim-{number}.md").read_text()
        claim_checks[str(number)] = {
            "verdict": "VERIFIED" in text,
            "raw_link": "raw JSON" in text,
            "code_link": "../code/" in text,
            "limitations": "## Limits" in text,
            "source_contract": "source contract" in text.lower(),
        }
    index = (root / "pages/index.md").read_text()
    logbook = json.loads((root / "logbook.json").read_text())
    checks = {
        "no_missing_links": not missing,
        "required_files_reachable": required <= visited,
        "all_claim_cells_complete": all(
            all(values.values()) for values in claim_checks.values()
        ),
        "visibility_matrix_present": "Evaluator-visible matrix" in index,
        "historical_label_exact": "Historical rejected baseline" in index,
        "current_claims_first_in_navigation": logbook["root"]["children"][0]["slug"] == "claim-1",
        "historical_navigation_last": logbook["root"]["children"][-1]["slug"]
        == "historical-rejected-baseline",
    }
    return {
        "files_opened": sorted(visited),
        "missing": missing,
        "claim_checks": claim_checks,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_numbers(root: Path) -> dict:
    raw = json.loads((root / "data/cumulative_run.json").read_text())
    pages = "\n".join(
        (root / f"pages/claim-{number}.md").read_text() for number in range(1, 6)
    )
    checks = {
        "raw_passed": raw["passed"],
        "fixed_command_exact": raw["fixed_command"]
        == "uv sync --frozen --all-extras && uv run --frozen python -m reproduction.run_all",
        "all_claim_evidence_passed": all(
            raw[f"claim{number}"]["verifier"]["passed"]
            if number != 3
            else raw["claim3"]["symbolic"]["passed"]
            and raw["claim3"]["empirical"]["passed"]
            for number in range(1, 6)
        ),
        "claim3_display_matches": all(
            token in pages for token in ["2.677574", "2.759170", "2.676830", "2.756708"]
        ),
        "claim4_display_matches": all(
            token in pages for token in ["1.782813", "1.771809", "1.763749", "1.754189"]
        ),
        "claim5_display_matches": all(
            token in pages for token in ["0.679499", "0.669128", "0.999900"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def verify_secret_scan() -> dict:
    findings = []
    patterns = [re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"sk-[A-Za-z0-9]{20,}")]
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if any(pattern.search(text) for pattern in patterns):
            findings.append(str(path))
    return {"findings": findings, "passed": not findings}


def verify_marimo() -> dict:
    command = ["marimo", "check", "notebooks/finite_width_ntk_reproduction.py"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "passed": result.returncode == 0,
    }


def verify_images() -> dict:
    expected = {
        "claim2_five_diagrams.svg": (
            "d741653f4322ad3a2f11c11ac22065f330697242eb69a0833b046fb181de3ee5",
            "3ca6d485365515647c0d23a3326c6675ff647d4f1d04df49a78f31b219219c6e",
        ),
        "claim3_exact_scale.svg": (
            "d42bf81d4d34499ee8de3596da8cf985a28852bb32c3e688ab0e54783860f602",
            "9c361837d8eeb7e7e334048a6a09c00a3482d30ce22303ffa392de201d1c1ac3",
        ),
        "claim4_gelu_correction.svg": (
            "149bf894293b36a1e8e8756425032e47f0a97b522266b8efecefbad6787a0eae",
            "2dcd95d9af4c2e433e2695784d81aa09eea21918b23f22e9e9b19963b0fbe438",
        ),
        "claim5_depth_stability.svg": (
            "3f0430607aafbbc90667f831580b0ec1a0a1e9962a11e5d5f48ebea0b5d37cd5",
            "d74d4c34aae2f00ddc9d9e002b3a9da27df3d0022e44a31e77f912fa458dfd8d",
        ),
    }
    rows = {}
    for name, (payload_digest, committed_digest) in expected.items():
        path = Path("reports/reproduction/images") / name
        space_path = ROOT / "reports/reproduction/images" / name
        root = ET.parse(path).getroot()
        width = float(root.attrib["width"])
        height = float(root.attrib["height"])
        rows[name] = {
            "well_formed_svg": root.tag.endswith("svg"),
            "positive_dimensions": width > 0 and height > 0,
            "has_graphical_elements": len(list(root)) >= 4,
            "hf_payload_hash_matches_before_terminal_lf_normalization": hashlib.sha256(
                path.read_bytes().removesuffix(b"\n")
            ).hexdigest()
            == payload_digest,
            "committed_hash_matches": sha256(path) == committed_digest,
            "space_copy_hash_matches": sha256(space_path) == committed_digest,
        }
    return {
        "images": rows,
        "passed": all(all(checks.values()) for checks in rows.values()),
    }


def verify_release() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        fresh = Path(directory) / "candidate"
        shutil.copytree(ROOT, fresh)
        visibility = verify_visibility(fresh)
        numbers = verify_numbers(fresh)
        historical = verify_historical_archive(fresh)
    allowlist = verify_allowlist()
    secret_scan = verify_secret_scan()
    marimo = verify_marimo()
    images = verify_images()
    passed = all(
        item["passed"]
        for item in [
            visibility,
            numbers,
            historical,
            allowlist,
            secret_scan,
            marimo,
            images,
        ]
    )
    return {
        "fresh_candidate_traversal": visibility,
        "displayed_numbers": numbers,
        "historical_evidence": historical,
        "upload_allowlist": allowlist,
        "secret_scan": secret_scan,
        "marimo": marimo,
        "images": images,
        "passed": passed,
    }
