"""Fail-closed evidence validation before resuming or analyzing experiments."""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

from aeonq.benchmarks.evidence import (
    ROOT,
    SEED_SCHEME,
    trial_seed,
    verify_source_archive,
)

# Direct benchmark CLIs put benchmarks/ on sys.path. The editable package
# does not install the research scripts used to define expected populations.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


@lru_cache(maxsize=64)
def _check_source(digest: str, archive: str) -> None:
    verify_source_archive({"source_snapshot_sha256": digest, "source_archive": archive})


def validate_records(
    rows: list[dict],
    keys: tuple[str, ...],
    expected: dict,
    *,
    complete: bool = True,
    allow_execution_failures: bool = False,
) -> None:
    seen = set()
    sources = set()
    lock_hash = sha256((ROOT / "requirements-lock.txt").read_bytes()).hexdigest()
    for number, row in enumerate(rows, 1):
        try:
            identity = tuple(row[k] for k in keys)
            if identity in seen:
                raise ValueError("duplicate trial identity")
            seen.add(identity)
            if identity not in expected:
                raise ValueError("unexpected trial identity")
            spec = expected[identity]
            if not allow_execution_failures and (
                row.get("error") is not None
                or ("result" in row and row["result"] is None)
            ):
                raise ValueError("execution failure")
            for key, value in spec.items():
                if key == "a_true":
                    if abs(float(row[key]) - value) > 1e-12:
                        raise ValueError("ground truth mismatch")
                elif json_value(row.get(key)) != json_value(value):
                    raise ValueError(f"{key} mismatch")
            env = row["environment"]
            if env.get("requirements_lock_sha256") != lock_hash:
                raise ValueError("dependency lock hash mismatch")
            if env.get("seed_scheme") != SEED_SCHEME:
                raise ValueError("seed scheme mismatch")
            if not all(
                env.get(k)
                for k in (
                    "python",
                    "platform",
                    "dependencies",
                    "requirements_lock_sha256",
                )
            ):
                raise ValueError("incomplete environment metadata")
            _check_source(env["source_snapshot_sha256"], env["source_archive"])
            sources.add(env["source_snapshot_sha256"])
            json_value(row)  # reject NaN / infinity
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trial {number}: {exc}") from exc
    if complete and seen != set(expected):
        raise ValueError(f"missing {len(set(expected) - seen)} expected trials")
    if len(sources) > 1:
        raise ValueError("mixed source snapshots in one experiment")


def validate_mutation_battery(rows: list[dict]) -> None:
    from aeonq.conformance.mutation import MUTANTS

    expected = {
        (sdk, mutant)
        for sdk in ("qiskit", "cirq", "pennylane", "qsharp")
        for mutant in ("none", *MUTANTS)
    }
    seen = set()
    for row in rows:
        key = (row["sdk"], row["mutant"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected mutation battery record")
        seen.add(key)
        failures = [
            int(row[f"failed_{level}"])
            for level in ("operator", "measurement", "sampling")
        ]
        if any(
            not 0 <= count <= limit
            for count, limit in zip(failures, (43, 5, 18), strict=True)
        ):
            raise ValueError("mutation failure counts out of range")
        failed, unsupported = int(row["failed_checks"]), int(row["unsupported_checks"])
        killed = str(row["killed"]) == "True"
        if (
            int(row["checks"]) != 66
            or failed != sum(failures)
            or not 0 <= unsupported <= 66 - failed
            or killed != (failed > 0)
        ):
            raise ValueError("mutation check totals/status disagree")
    if seen != expected:
        raise ValueError("missing mutation battery records")


def validate_mutation_impact(
    rows: list[dict], summaries: list[dict], repetitions: int = 20
) -> None:
    """Fault-induced rejection is an outcome; never summarize only survivors."""
    from aeonq.benchmarks.problems import bernoulli, lognormal_tail, strike_for_index
    from aeonq.conformance.mutation import MUTANTS
    from statistics import mean
    from math import isclose

    cases = (
        bernoulli(0.3),
        lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True),
    )
    methods = {
        "iqae": {"epsilon": 0.01, "max_power": 8},
        "canonical_qae": {"evaluation_qubits": 4, "shots": 256},
    }
    expected = {
        (mutant, case.case_id, method, trial): {
            "sdk": "cirq",
            "a_true": case.a_true,
            "configuration": configuration,
            "seed": trial_seed(
                "mutation_impact", case.case_id, method, "cirq", trial, 4001
            ),
        }
        for mutant in ("none", *MUTANTS)
        for case in cases
        for method, configuration in methods.items()
        for trial in range(repetitions)
    }
    validate_records(
        rows,
        ("mutant", "case_id", "method", "trial"),
        expected,
        allow_execution_failures=True,
    )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mutant"], row["case_id"], row["method"]), []).append(
            row
        )
        if row["error"] is None:
            if not isclose(
                row["absolute_error"],
                abs(row["estimate"] - row["a_true"]),
                abs_tol=1e-12,
            ):
                raise ValueError("mutation impact estimate/error mismatch")
        elif (
            row["mutant"] == "none"
            or row["estimate"] is not None
            or row["absolute_error"] is not None
        ):
            raise ValueError("invalid mutation execution failure record")
    seen = set()
    for summary in summaries:
        key = (summary["mutant"], summary["case_id"], summary["method"])
        if key in seen or key not in groups:
            raise ValueError("duplicate or unexpected mutation impact summary")
        seen.add(key)
        observations = groups[key]
        crashes = sum(row["error"] is not None for row in observations)
        if (
            int(summary["repetitions"]) != repetitions
            or int(summary["crashes"]) != crashes
        ):
            raise ValueError("mutation impact population mismatch")
        for column, statistic in (("mean_abs_error", mean), ("max_abs_error", max)):
            value = summary[column]
            if crashes:
                if value not in (None, ""):
                    raise ValueError("mutation impact analyzes only successful runs")
            elif not isclose(
                float(value),
                statistic(r["absolute_error"] for r in observations),
                abs_tol=1e-12,
            ):
                raise ValueError("mutation impact summary mismatch")
    if seen != set(groups):
        raise ValueError("missing mutation impact summaries")


def read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resume_records(path: Path) -> list[dict]:
    """Recover only a final interrupted write, never corrupt interior records."""
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("malformed interior record") from None
    return rows


def manifest_expected(profile: dict) -> dict:
    from aeonq.benchmarks.runner import build_problem

    return {
        (case.case_id, label, backend, trial): {
            "profile": profile["profile"],
            "configuration": spec.get("options", spec),
            "algorithm": spec.get("method", label),
            "scheduler": spec.get("scheduler") or "default",
            "a_true": case.a_true,
            "reference_method": case.reference_method,
            "seed": trial_seed(
                profile["profile"],
                case.case_id,
                label,
                backend,
                trial,
                profile["base_seed"],
            ),
        }
        for problem in profile["problems"]
        for case in [build_problem(problem)]
        for label, spec in profile["algorithms"].items()
        for backend in profile["backends"]
        for trial in range(profile["repetitions"])
    }


def validate_manifest(
    rows: list[dict], profile: dict, *, complete: bool = True
) -> None:
    validate_records(
        rows,
        ("case_id", "method_label", "backend", "trial"),
        manifest_expected(profile),
        complete=complete,
    )


def extended_expected(study: str, repetitions: int) -> tuple[tuple, dict]:
    expected: dict[tuple, dict]
    if study == "native_baselines":
        from benchmarks import native_baselines as native

        keys = ("case_id", "method", "library", "trial")
        expected = {
            (case.case_id, method, library, trial): {
                "a_true": case.a_true,
                "configuration": native.configuration(method),
                "seed": native.seed_for(case.case_id, method, library, trial),
            }
            for case in native.cases()
            for method in native.METHODS
            for library in native.LIBRARIES
            if method in native.LIBRARY_METHODS.get(library, native.METHODS)
            for trial in range(repetitions)
        }
    elif study == "equivalence_estimators":
        from benchmarks import equivalence_study as equiv

        keys = ("case_id", "method", "backend", "trial")
        expected = {
            (case.case_id, method, backend, trial): {
                "a_true": case.a_true,
                "configuration": equiv.METHODS[method],
                "seed": trial_seed(
                    study, case.case_id, method, backend, trial, 1_300_001
                ),
            }
            for case in equiv.cases()
            for method in equiv.METHODS
            for backend in equiv.BACKENDS
            for trial in range(repetitions)
        }
    elif study == "noise_study":
        from benchmarks import noise_study as noise

        keys = ("one_qubit_error", "case_id", "method", "trial")
        expected = {
            (rate, case.case_id, method, trial): {
                "a_true": case.a_true,
                "configuration": noise.METHODS[method],
                "two_qubit_error": min(1.0, 10 * rate),
                "seed": trial_seed(
                    study, case.case_id, method, "qiskit", trial, 920_001, rate
                ),
            }
            for rate in noise.ONE_QUBIT_RATES
            for case in noise.cases()
            for method in noise.METHODS
            for trial in range(repetitions)
        }
    else:
        raise ValueError(f"unknown study: {study}")
    return keys, expected


def validate_extended(
    rows: list[dict], study: str, repetitions: int, *, complete: bool = True
) -> None:
    keys, expected = extended_expected(study, repetitions)
    validate_records(rows, keys, expected, complete=complete)


def validate_sampling(document: dict) -> None:
    from benchmarks.equivalence_study import BACKENDS, SAMPLING_SHOTS, cases
    import numpy as np
    from aeonq.ir import GroverPower, Sequence
    from aeonq.reference import grover_for, statevector

    env = document["environment"]
    if env.get("seed_scheme") != SEED_SCHEME:
        raise ValueError("sampling seed scheme mismatch")
    _check_source(env["source_snapshot_sha256"], env["source_archive"])
    expected = {}
    for case in cases():
        q = grover_for(case.problem)
        for power in (0, 1, 2, 4):
            exact = (
                np.abs(
                    statevector(
                        Sequence(
                            (
                                case.problem.state_preparation,
                                GroverPower(q, power, q.n_qubits),
                            ),
                            q.n_qubits,
                        )
                    )
                )
                ** 2
            )
            for backend in BACKENDS:
                expected[(case.case_id, power, backend)] = exact
    seen = set()
    for row in document["rows"]:
        key = (row["case_id"], row["power"], row["backend"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected sampling identity")
        seen.add(key)
        counts = row["counts"]
        if (
            row["shots"] != SAMPLING_SHOTS
            or sum(counts) != SAMPLING_SHOTS
            or any(not isinstance(c, int) or c < 0 for c in counts)
            or len(counts) != len(expected[key])
            or not np.allclose(
                row["exact_probabilities"], expected[key], rtol=0, atol=1e-12
            )
            or row["seed"]
            != trial_seed(
                "equivalence_sampling", key[0], "sampling", key[2], key[1], 333
            )
        ):
            raise ValueError("sampling configuration, counts, seed or truth mismatch")
    if seen != set(expected):
        raise ValueError("missing sampling comparisons")
