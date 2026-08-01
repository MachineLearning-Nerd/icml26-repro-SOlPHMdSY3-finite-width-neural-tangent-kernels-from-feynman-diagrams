"""Fixed cumulative campaign entrypoint."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from reproduction.claim2_independent_checker import check
from reproduction.claim2_verifier import verify


def cpu_metadata() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "core_estimate_before_run": 8,
        "estimate_basis": "symbolic enumeration is single-core; cpu-upgrade is mandated",
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "flavor_contract": {"vcpus": 8, "ram_gb": 32, "gpu": False},
        "actual_os_cpu_count": os.cpu_count(),
        "actual_affinity_cpu_count": affinity,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "cuda_visible_devices_name_present": "CUDA_VISIBLE_DEVICES" in os.environ,
    }


def main() -> int:
    started = time.perf_counter()
    certificate = verify()
    independent = check(certificate)
    control = subprocess.run(
        [sys.executable, "-m", "reproduction.claim2_negative_control"],
        check=False,
        capture_output=True,
        text=True,
    )
    negative_control = {
        "command": "python -m reproduction.claim2_negative_control",
        "expected_exit": "nonzero",
        "actual_exit": control.returncode,
        "passed": control.returncode != 0,
        "stdout": control.stdout.strip(),
        "stderr": control.stderr.strip(),
    }
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fixed_command": (
            "uv sync --frozen --all-extras && "
            "uv run --frozen python -m reproduction.run_all"
        ),
        "seeds": [],
        "cpu": cpu_metadata(),
        "runtime_seconds": elapsed,
        "claim2_verifier": certificate,
        "claim2_independent_checker": independent,
        "claim2_negative_control": negative_control,
    }
    result["passed"] = (
        certificate["passed"] and independent["passed"] and negative_control["passed"]
    )
    Path(".openresearch/runtime").mkdir(parents=True, exist_ok=True)
    Path(".openresearch/runtime/latest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("OPENRESEARCH_EVIDENCE_BEGIN")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OPENRESEARCH_EVIDENCE_END")
    print(f"SUMMARY claim2={'VERIFIED' if result['passed'] else 'BLOCKED'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
