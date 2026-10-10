"""Mutation analysis of the conformance battery with seeded adapter faults.

Every mutant in ``aeonq.conformance.mutation.MUTANTS`` is injected into each
SDK adapter, and the full battery is run. A mutant is *killed* when at least
one check fails. Rows for the unmutated adapters ("none") are the false-alarm
control: a correct adapter must pass every check. Per-level failure counts
show which kind of check exposes which kind of bug.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.benchmarks.runner import environment_record
from aeonq.benchmarks.evidence import trial_seed
from aeonq.conformance.battery import run_battery
from aeonq.conformance.mutation import MUTANTS, MutantAdapter

LEVELS = ("operator", "measurement", "sampling")


def run(output: Path) -> list[dict]:
    adapters = {
        "qiskit": QiskitAdapter,
        "cirq": CirqAdapter,
        "pennylane": PennyLaneAdapter,
        "qsharp": QSharpAdapter,
    }
    rows = []
    for sdk, factory in adapters.items():
        for mutant in ("none", *MUTANTS):
            adapter = (
                factory() if mutant == "none" else MutantAdapter(factory(), mutant)
            )
            started = perf_counter()
            results = run_battery(adapter)
            failures = {
                level: sum(r.passed is False for r in results if r.level == level)
                for level in LEVELS
            }
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "sdk": sdk,
                    "mutant": mutant,
                    "description": MUTANTS.get(mutant, "unmutated adapter (control)"),
                    "checks": len(results),
                    "failed_checks": sum(failures.values()),
                    "unsupported_checks": sum(r.passed is None for r in results),
                    **{f"failed_{level}": failures[level] for level in LEVELS},
                    "killed": sum(failures.values()) > 0,
                    "first_failure": next(
                        (
                            f"{r.level}:{r.case}:{r.check}"
                            for r in results
                            if r.passed is False
                        ),
                        "",
                    ),
                    "seconds": perf_counter() - started,
                }
            )
            print(sdk, mutant, rows[-1]["killed"], failures, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".environment.json").write_text(
        json.dumps(environment_record(), indent=2), encoding="utf-8"
    )
    return rows


def estimator_impact(output: Path, *, repetitions: int = 20) -> list[dict]:
    """Measure how each fault changes the estimates the estimators report.

    Faults invisible to sampling can still matter: canonical QAE interferes
    controlled Grover powers, so a relative phase error shifts its estimate
    even when A·Q^k sampling is unchanged. The study runs on Cirq; the battery
    shows that every SDK is affected identically.
    """
    from aeonq import estimate
    from aeonq.benchmarks.problems import bernoulli, lognormal_tail, strike_for_index

    cases = (
        bernoulli(0.3),
        lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True),
    )
    methods = {
        "iqae": {"epsilon": 0.01, "max_power": 8},
        "canonical_qae": {"evaluation_qubits": 4, "shots": 256},
    }
    environment = environment_record()
    observations = []
    rows = []
    for mutant in ("none", *MUTANTS):
        adapter = (
            CirqAdapter() if mutant == "none" else MutantAdapter(CirqAdapter(), mutant)
        )
        for case in cases:
            for method, options in methods.items():
                errors, crashes = [], 0
                for trial in range(repetitions):
                    # Matched streams across faults isolate their effect; cases
                    # and estimators have distinct streams. This is descriptive
                    # fault analysis, not independent-sample equivalence testing.
                    seed = trial_seed(
                        "mutation_impact", case.case_id, method, "cirq", trial, 4001
                    )
                    observation = {
                        "sdk": "cirq",
                        "mutant": mutant,
                        "case_id": case.case_id,
                        "a_true": case.a_true,
                        "method": method,
                        "trial": trial,
                        "seed": seed,
                        "configuration": options,
                        "environment": environment,
                    }
                    try:
                        result = estimate(
                            case.problem,
                            algorithm=method,
                            backend=adapter,
                            seed=seed,
                            **options,
                        )
                        errors.append(abs(result.estimate - case.a_true))
                        observation.update(
                            estimate=result.estimate,
                            absolute_error=errors[-1],
                            error=None,
                        )
                    except Exception as exc:
                        crashes += 1
                        observation.update(
                            estimate=None,
                            absolute_error=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    observations.append(observation)
                rows.append(
                    {
                        "sdk": "cirq",
                        "mutant": mutant,
                        "case_id": case.case_id,
                        "a_true": case.a_true,
                        "method": method,
                        "repetitions": repetitions,
                        "crashes": crashes,
                        "mean_abs_error": sum(errors) / len(errors)
                        if errors and not crashes
                        else None,
                        "max_abs_error": max(errors)
                        if errors and not crashes
                        else None,
                    }
                )
                print(
                    mutant, case.case_id, method, rows[-1]["mean_abs_error"], flush=True
                )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8"
    )
    output.with_suffix(".environment.json").write_text(
        json.dumps(
            {
                **environment,
                "repetitions": repetitions,
                "seed_pairing": "matched across mutants only",
                "methods": methods,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    from aeonq.benchmarks.validation import validate_mutation_impact

    validate_mutation_impact(observations, rows, repetitions)
    return rows


if __name__ == "__main__":
    run(Path("benchmarks/results/raw/adapter_mutation.csv"))
    estimator_impact(Path("benchmarks/results/raw/adapter_mutation_impact.csv"))
