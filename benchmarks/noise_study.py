"""Coverage of ideal-model intervals under depolarizing gate noise.

The frequentist guarantees of IQAE and MLQAE assume ideal independent
Bernoulli shots. This study adds depolarizing noise through an Aer noise model
and measures coverage and bias as the error rate grows. Two-qubit gates get
ten times the one-qubit rate. Results record the guarantee status that aeonq
reports, which is withheld whenever the backend declares non-ideal sampling.
"""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

from qiskit_aer.noise import NoiseModel, depolarizing_error

from aeonq import estimate
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.benchmarks.problems import (
    BenchmarkProblem,
    bernoulli,
    lognormal_tail,
    strike_for_index,
)
from aeonq.benchmarks.runner import environment_record, map_trials, open_trial_log
from aeonq.benchmarks.evidence import trial_seed

ONE_QUBIT_RATES = (0.0, 1e-4, 1e-3, 5e-3, 1e-2)
METHODS = {
    "iqae": {"epsilon": 0.01, "alpha": 0.05, "shots_per_round": 512, "max_power": 16},
    "mlqae": {
        "epsilon": 0.01,
        "alpha": 0.05,
        "powers": (0, 1, 2, 4, 8),
        "shots_per_power": 512,
    },
}


def noise_model(rate: float) -> NoiseModel | None:
    if rate == 0:
        return None
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(rate, 1),
        ["u", "u1", "u2", "u3", "rx", "ry", "rz", "sx", "x", "h"],
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(min(1.0, 10 * rate), 2), ["cx"]
    )
    return model


@lru_cache(maxsize=1)
def cases() -> tuple[BenchmarkProblem, ...]:
    # One qubit (Q^k transpiles to a single gate), then 3- and 4-qubit case
    # studies whose compiled depth grows with k.
    return (
        bernoulli(0.25),
        lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True),
        lognormal_tail(3, strike_for_index(3, 5), objective_ancilla=True),
    )


@lru_cache(maxsize=None)
def backend_for(rate: float) -> QiskitAdapter:
    return QiskitAdapter(noise_model=noise_model(rate))


def trial_row(task: tuple[float, int, str, int]) -> dict:
    """One trial, identified by (one-qubit rate, case index, method, trial)."""
    rate, case_index, method, trial = task
    case = cases()[case_index]
    seed = trial_seed(
        "noise_study", case.case_id, method, "qiskit", trial, 920_001, condition=rate
    )
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "one_qubit_error": rate,
        "two_qubit_error": min(1.0, 10 * rate),
        "case_id": case.case_id,
        "a_true": case.a_true,
        "method": method,
        "trial": trial,
        "seed": seed,
        "configuration": METHODS[method],
        "environment": environment_record(),
    }
    started = perf_counter()
    try:
        result = estimate(
            case.problem,
            algorithm=method,
            backend=backend_for(rate),
            seed=seed,
            **METHODS[method],
        )
        low, high = result.confidence_or_credible_interval
        row.update(
            estimate=result.estimate,
            interval=[low, high],
            covered=low <= case.a_true <= high,
            convergence_status=result.convergence_status,
            guarantee_status=result.guarantee_status,
            warnings=list(result.warnings),
            max_power=result.max_power,
            total_queries=result.total_oracle_queries,
            error=None,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = perf_counter() - started
    return row


def run(
    output: Path, repetitions: int, *, resume: bool = False, workers: int = 1
) -> None:
    from aeonq.benchmarks.validation import (
        validate_extended,
        read_records,
        resume_records,
    )

    if resume and output.exists():
        validate_extended(
            resume_records(output), "noise_study", repetitions, complete=False
        )
    stream, done = open_trial_log(
        output, ("one_qubit_error", "case_id", "method", "trial"), resume=resume
    )
    tasks = [
        (rate, case_index, method, trial)
        for rate in ONE_QUBIT_RATES
        for case_index, case in enumerate(cases())
        for method in METHODS
        for trial in range(repetitions)
        if (rate, case.case_id, method, trial) not in done
    ]
    with stream:
        for row in map_trials(trial_row, tasks, workers=workers):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if row["trial"] == repetitions - 1:
                print(row["one_qubit_error"], row["case_id"], row["method"], flush=True)
    output.with_suffix(".environment.json").write_text(
        json.dumps({**environment_record(), "repetitions": repetitions}, indent=2),
        encoding="utf-8",
    )
    validate_extended(read_records(output), "noise_study", repetitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", default="benchmarks/results/raw/noise_study.jsonl")
    parser.add_argument(
        "--resume", action="store_true", help="keep recorded trials and skip them"
    )
    parser.add_argument("--workers", type=int, default=1, help="trial worker processes")
    arguments = parser.parse_args()
    run(
        Path(arguments.output),
        arguments.repetitions,
        resume=arguments.resume,
        workers=arguments.workers,
    )
