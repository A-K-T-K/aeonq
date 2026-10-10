"""Audit saved trials against their manifest and local lock file."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from aeonq.benchmarks.runner import build_problem
from aeonq.benchmarks.evidence import trial_seed
from aeonq.benchmarks.validation import (
    validate_manifest,
    validate_extended,
    validate_sampling,
    read_records,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PROFILES = (
    "publication_reference",
    "publication_cross_sdk",
    "publication_scheduler_qiskit",
    "publication_case_study",
    "publication_case_study_sdk",
    "publication_case_study_qsharp",
)


def audit() -> dict[str, object]:
    manifest_bytes = (ROOT / "experiment_manifest.yaml").read_bytes()
    manifest = json.loads(manifest_bytes)
    lock = (WORKSPACE / "requirements-lock.txt").read_bytes()
    lock_hash = sha256(lock).hexdigest()
    locked_versions = {
        package: pinned
        for line in lock.decode("utf-8-sig").splitlines()
        if line and not line.startswith(("#", "-e "))
        for package, pinned in (line.split("==", 1),)
    }
    installed_aeonq = version("aeonq")
    errors: list[str] = []
    counts: Counter[str] = Counter()
    raw_sha256: dict[str, str] = {}
    missing_commits = 0
    for name in PROFILES:
        profile = manifest["profiles"][name]
        raw_path = ROOT / "results" / "raw" / f"{name}.jsonl"
        raw_bytes = raw_path.read_bytes()
        raw_sha256[name] = sha256(raw_bytes).hexdigest()
        validate_manifest(read_records(raw_path), profile)
        expected = {
            (case.case_id, label, backend, trial): (
                case,
                algorithm_spec.get("method", label),
                algorithm_spec.get("scheduler") or "default",
                algorithm_spec.get("options", algorithm_spec),
            )
            for problem_spec in profile["problems"]
            for case in (build_problem(problem_spec),)
            for label, algorithm_spec in profile["algorithms"].items()
            for backend in profile["backends"]
            for trial in range(profile["repetitions"])
        }
        seen: set[tuple[str, str, str, int]] = set()
        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
            row = json.loads(line)
            key = (row["case_id"], row["method_label"], row["backend"], row["trial"])
            if key in seen:
                errors.append(f"{name}:{line_number}: duplicate trial identity")
                continue
            seen.add(key)
            if key not in expected:
                errors.append(f"{name}:{line_number}: trial absent from manifest")
                continue
            case, algorithm, scheduler, options = expected[key]
            environment = row["environment"]
            if (
                row["profile"] != name
                or row["algorithm"] != algorithm
                or row["scheduler"] != scheduler
                or row["configuration"] != options
                or row["seed"]
                != trial_seed(
                    name,
                    row["case_id"],
                    row["method_label"],
                    row["backend"],
                    row["trial"],
                    profile["base_seed"],
                )
                or abs(row["a_true"] - case.a_true) > 1e-12
                or row["reference_method"] != case.reference_method
                or environment.get("requirements_lock_sha256") != lock_hash
                or not environment.get("python")
                or not environment.get("platform")
                or not environment.get("processor")
                or not environment.get("dependencies")
                or any(
                    environment["dependencies"].get(package) != pinned
                    for package, pinned in locked_versions.items()
                    if package in environment["dependencies"]
                )
                or environment["dependencies"].get("aeonq") != installed_aeonq
                or row["result"] is None
                or row["error"] is not None
            ):
                errors.append(
                    f"{name}:{line_number}: metadata, truth, or result mismatch"
                )
            try:
                trial_time = datetime.fromisoformat(row["timestamp_utc"])
                environment_time = datetime.fromisoformat(environment["timestamp_utc"])
                if trial_time.utcoffset() != timezone.utc.utcoffset(
                    trial_time
                ) or environment_time.utcoffset() != timezone.utc.utcoffset(
                    environment_time
                ):
                    raise ValueError("timestamp is not UTC")
            except ValueError:
                errors.append(f"{name}:{line_number}: invalid UTC timestamp")
            missing_commits += environment.get("git_commit") is None
            counts[name] += 1
        if seen != set(expected):
            errors.append(
                f"{name}: missing {len(set(expected) - seen)} manifest trials"
            )

    extended_counts = {}
    for study in ("native_baselines", "noise_study", "equivalence_estimators"):
        path = ROOT / "results" / "raw" / f"{study}.jsonl"
        metadata = json.loads(
            path.with_suffix(".environment.json").read_text(encoding="utf-8")
        )
        rows = read_records(path)
        validate_extended(rows, study, metadata["repetitions"])
        extended_counts[study] = len(rows)
        raw_sha256[study] = sha256(path.read_bytes()).hexdigest()
    from aeonq.benchmarks.validation import (
        validate_mutation_impact,
        validate_mutation_battery,
    )

    impact_path = ROOT / "results/raw/adapter_mutation_impact.jsonl"
    import csv

    with impact_path.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
        validate_mutation_impact(
            read_records(impact_path), list(csv.DictReader(stream))
        )
    mutation_path = ROOT / "results/raw/adapter_mutation.csv"
    with mutation_path.open(encoding="utf-8", newline="") as stream:
        validate_mutation_battery(list(csv.DictReader(stream)))
    from aeonq.benchmarks.evidence import verify_source_archive

    verify_source_archive(
        json.loads(
            mutation_path.with_suffix(".environment.json").read_text(encoding="utf-8")
        )
    )
    raw_sha256["adapter_mutation"] = sha256(mutation_path.read_bytes()).hexdigest()
    extended_counts["adapter_mutation_impact"] = len(read_records(impact_path))
    raw_sha256["adapter_mutation_impact"] = sha256(impact_path.read_bytes()).hexdigest()
    sampling_path = ROOT / "results/raw/equivalence_sampling.json"
    validate_sampling(json.loads(sampling_path.read_text(encoding="utf-8")))
    raw_sha256["equivalence_sampling"] = sha256(sampling_path.read_bytes()).hexdigest()
    installed_mismatches = []
    for package, pinned in locked_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "missing"
        if installed != pinned:
            installed_mismatches.append(
                f"{package}: lock {pinned}, installed {installed}"
            )
    errors.extend(installed_mismatches)
    report: dict[str, object] = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "requirements_lock_sha256": lock_hash,
        "raw_sha256": raw_sha256,
        "trial_counts": dict(counts),
        "total_trials": sum(counts.values()),
        "extended_trial_counts": extended_counts,
        "trials_without_git_commit": missing_commits,
        "installed_dependency_mismatches": installed_mismatches,
        "errors": errors,
        "passed": not errors,
    }
    destination = ROOT / "results" / "processed" / "reproducibility_audit.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"reproducibility audit failed: {errors[:5]}")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
