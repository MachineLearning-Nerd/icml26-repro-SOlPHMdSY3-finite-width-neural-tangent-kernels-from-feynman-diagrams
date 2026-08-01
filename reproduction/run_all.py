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
from reproduction.claim3_empirical_independent_checker import check as check_claim3_empirical
from reproduction.claim3_paper_scale import run_five_million_scale
from reproduction.claim3_verifier import symbolic_certificate
from reproduction.claim4_gelu import run_gelu_correction
from reproduction.claim4_independent_checker import check as check_claim4
from reproduction.claim5_independent_checker import check as check_claim5
from reproduction.claim5_stability import run_paper_scale


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
        "estimate_basis": (
            "five-million-sample Claim 3 plus paper-scale Claims 4 and 5 use all "
            "8 vCPUs for batched JAX CPU matrix products; estimated peak memory "
            "below 8 GB"
        ),
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
    claim5_independent = check_claim5()
    claim5 = run_paper_scale()
    claim4_independent = check_claim4()
    claim4 = run_gelu_correction()
    claim3_empirical_independent = check_claim3_empirical()
    claim3_empirical_negative = run_expected_failure(
        "reproduction.claim3_empirical_negative_control"
    )
    claim3_empirical = run_five_million_scale()
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
        "claim3_empirical_independent_checker": claim3_empirical_independent,
        "claim3_empirical_negative_control": claim3_empirical_negative,
        "claim3_empirical_verifier": claim3_empirical,
        "claim4_independent_checker": claim4_independent,
        "claim4_verifier": claim4,
        "claim5_independent_checker": claim5_independent,
        "claim5_verifier": claim5,
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
        and claim3_empirical_independent["passed"]
        and claim3_empirical_negative["passed"]
        and claim3_empirical["passed"]
        and claim4_independent["passed"]
        and claim4["passed"]
        and claim5_independent["passed"]
        and claim5["passed"]
    )
    Path(".openresearch/runtime").mkdir(parents=True, exist_ok=True)
    Path(".openresearch/runtime/latest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("OPENRESEARCH_EVIDENCE_BEGIN")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OPENRESEARCH_EVIDENCE_END")
    claim1_status = "VERIFIED" if claim1["passed"] and claim1_independent["passed"] else "BLOCKED"
    claim2_status = (
        "VERIFIED"
        if certificate["passed"] and independent["passed"] and negative_control["passed"]
        else "BLOCKED"
    )
    claim3_status = (
        "VERIFIED"
        if (
            claim3["passed"]
            and claim3_independent["passed"]
            and claim3_negative["passed"]
            and claim3_empirical_independent["passed"]
            and claim3_empirical_negative["passed"]
            and claim3_empirical["passed"]
        )
        else "BLOCKED"
    )
    claim5_status = (
        "VERIFIED" if claim5_independent["passed"] and claim5["passed"] else "BLOCKED"
    )
    claim4_status = (
        "VERIFIED" if claim4_independent["passed"] and claim4["passed"] else "BLOCKED"
    )
    print(
        f"SUMMARY claim1={claim1_status} claim2={claim2_status} "
        f"claim3={claim3_status} claim4={claim4_status} claim5={claim5_status}"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
