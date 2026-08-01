"""Fixed cumulative campaign entrypoint."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from reproduction.claim1_verifier import independent_check as check_claim1
from reproduction.claim1_verifier import verify as verify_claim1
from reproduction.claim2_independent_checker import check
from reproduction.claim2_verifier import verify
from reproduction.claim3_independent_checker import check as check_claim3
from reproduction.claim3_verifier import symbolic_certificate


def cgroup_cpu_metadata() -> dict:
    result = {}
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota, period = cpu_max.read_text().strip().split()
        result["cpu_max_raw"] = f"{quota} {period}"
        result["quota_cpu_count"] = None if quota == "max" else int(quota) / int(period)
    cpuset = Path("/sys/fs/cgroup/cpuset.cpus.effective")
    if cpuset.exists():
        value = cpuset.read_text().strip()
        result["cpuset_cpus_effective"] = value
    return result


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
        "cgroup": cgroup_cpu_metadata(),
    }


def run_expected_failure(module: str) -> dict:
    command = [sys.executable, "-m", module]
    control = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": f"python -m {module}",
        "expected_exit": "nonzero",
        "actual_exit": control.returncode,
        "passed": control.returncode != 0,
        "stdout": control.stdout.strip(),
        "stderr": control.stderr.strip(),
    }


def main() -> int:
    started = time.perf_counter()
    claim1 = verify_claim1()
    claim1_independent = check_claim1(claim1)
    certificate = verify()
    independent = check(certificate)
    negative_control = run_expected_failure("reproduction.claim2_negative_control")
    claim3 = symbolic_certificate()
    claim3_independent = check_claim3()
    claim3_negative = run_expected_failure("reproduction.claim3_negative_control")
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
        "claim1_verifier": claim1,
        "claim1_independent_checker": claim1_independent,
        "claim2_verifier": certificate,
        "claim2_independent_checker": independent,
        "claim2_negative_control": negative_control,
        "claim3_verifier": claim3,
        "claim3_independent_checker": claim3_independent,
        "claim3_negative_control": claim3_negative,
    }
    result["passed"] = (
        claim1["passed"]
        and claim1_independent["passed"]
        and certificate["passed"]
        and independent["passed"]
        and negative_control["passed"]
        and claim3["passed"]
        and claim3_independent["passed"]
        and claim3_negative["passed"]
    )
    Path(".openresearch/runtime").mkdir(parents=True, exist_ok=True)
    Path(".openresearch/runtime/latest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("OPENRESEARCH_EVIDENCE_BEGIN")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OPENRESEARCH_EVIDENCE_END")
    status = "VERIFIED" if result["passed"] else "BLOCKED"
    print(f"SUMMARY claim1={status} claim2={status} claim3={status}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
