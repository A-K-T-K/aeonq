"""Raw data for cross-SDK statistical equivalence.

Equivalence is tested, not inferred from a failure to reject a difference.
Margins are fixed before the data are seen:

* sampling: 200,000 shots of A·Q^k per SDK and circuit; a distribution-free
  95% upper bound on the total-variation distance between the SDK's outcome
  distribution and the exact one (empirical distance plus the Weissman et
  al. 2003 L1-deviation bound) must be below 0.01;
* estimators: 500 independent trials per SDK for IQAE and MLQAE; coverage
  must be within ±0.05 and mean absolute error within ±0.002 (20% of the
  requested epsilon) of the exact reference backend, by two one-sided tests
  at alpha = 0.05 (90% confidence intervals inside the margins).

This script only records raw counts and trials; statistics are computed from
these files by benchmarks/pipeline/process_extended.py.
"""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from aeonq import estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli, lognormal_tail, strike_for_index
from aeonq.benchmarks.runner import environment_record, map_trials, open_trial_log
from aeonq.benchmarks.evidence import trial_seed
from aeonq.ir import GroverPower, RegisterMap, Sequence
from aeonq.reference import grover_for, statevector

SAMPLING_SHOTS = 200_000
BACKENDS = {
    "reference": ReferenceAdapter,
    "qiskit": QiskitAdapter,
    "cirq": CirqAdapter,
    "pennylane": PennyLaneAdapter,
    "qsharp": QSharpAdapter,
}
METHODS = {
    "iqae": {"epsilon": 0.01, "alpha": 0.05, "shots_per_round": 256, "max_power": 16},
    "mlqae": {
        "epsilon": 0.01,
        "alpha": 0.05,
        "powers": (0, 1, 2, 4),
        "shots_per_power": 256,
    },
}


@lru_cache(maxsize=1)
def cases():
    return (
        bernoulli(0.25),
        lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True),
    )


def sampling(output: Path) -> None:
    rows = []
    for case in cases():
        q = grover_for(case.problem)
        n = q.n_qubits
        for power in (0, 1, 2, 4):
            ir = Sequence((case.problem.state_preparation, GroverPower(q, power, n)), n)
            exact = (np.abs(statevector(ir)) ** 2).tolist()
            for name, factory in BACKENDS.items():
                adapter = factory()
                compiled = adapter.compile(adapter.lower(ir))
                seed = trial_seed(
                    "equivalence_sampling", case.case_id, "sampling", name, power, 333
                )
                raw = adapter.execute(compiled, SAMPLING_SHOTS, seed=seed)
                result = adapter.normalize_result(
                    raw, RegisterMap(tuple(range(n)), tuple(range(n)))
                )
                counts = [0] * 2**n
                for bits, count in result.counts.items():
                    counts[sum(bit << i for i, bit in enumerate(bits))] += count
                rows.append(
                    {
                        "case_id": case.case_id,
                        "power": power,
                        "backend": name,
                        "shots": result.shots,
                        "seed": seed,
                        "counts": counts,
                        "exact_probabilities": exact,
                    }
                )
                print("sampling", case.case_id, power, name, flush=True)
    output.write_text(
        json.dumps({"environment": environment_record(), "rows": rows}),
        encoding="utf-8",
    )


def sampling_complete(output: Path) -> bool:
    """True when a previous run wrote the whole sampling file."""
    try:
        from aeonq.benchmarks.validation import validate_sampling

        validate_sampling(json.loads(output.read_text(encoding="utf-8")))
        return True
    except (OSError, ValueError, KeyError):
        return False


@lru_cache(maxsize=None)
def adapter_for(name: str):
    return BACKENDS[name]()


def trial_row(task: tuple[int, str, str, int]) -> dict:
    """One trial, identified by (case index, method, backend, trial)."""
    case_index, method, name, trial = task
    case = cases()[case_index]
    seed = trial_seed(
        "equivalence_estimators", case.case_id, method, name, trial, 1_300_001
    )
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id,
        "a_true": case.a_true,
        "method": method,
        "backend": name,
        "trial": trial,
        "seed": seed,
        "configuration": METHODS[method],
        "environment": environment_record(),
    }
    try:
        result = estimate(
            case.problem,
            algorithm=method,
            backend=adapter_for(name),
            seed=seed,
            **METHODS[method],
        )
        low, high = result.confidence_or_credible_interval
        row.update(
            estimate=result.estimate,
            interval=[low, high],
            covered=low <= case.a_true <= high,
            queries=result.total_oracle_queries,
            error=None,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def estimators(
    output: Path, repetitions: int, *, resume: bool = False, workers: int = 1
) -> None:
    from aeonq.benchmarks.validation import (
        validate_extended,
        read_records,
        resume_records,
    )

    if resume and output.exists():
        validate_extended(
            resume_records(output),
            "equivalence_estimators",
            repetitions,
            complete=False,
        )
    stream, done = open_trial_log(
        output, ("case_id", "method", "backend", "trial"), resume=resume
    )
    tasks = [
        (case_index, method, name, trial)
        for case_index, case in enumerate(cases())
        for method in METHODS
        for name in BACKENDS
        for trial in range(repetitions)
        if (case.case_id, method, name, trial) not in done
    ]
    with stream:
        for row in map_trials(trial_row, tasks, workers=workers):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if row["trial"] == repetitions - 1:
                print(
                    "estimators",
                    row["case_id"],
                    row["method"],
                    row["backend"],
                    flush=True,
                )
    validate_extended(read_records(output), "equivalence_estimators", repetitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--directory", default="benchmarks/results/raw")
    parser.add_argument(
        "--resume", action="store_true", help="keep recorded results and skip them"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="estimator-trial worker processes"
    )
    arguments = parser.parse_args()
    directory = Path(arguments.directory)
    directory.mkdir(parents=True, exist_ok=True)
    sampling_output = directory / "equivalence_sampling.json"
    if not (arguments.resume and sampling_complete(sampling_output)):
        sampling(sampling_output)
    estimators(
        directory / "equivalence_estimators.jsonl",
        arguments.repetitions,
        resume=arguments.resume,
        workers=arguments.workers,
    )
    (directory / "equivalence_estimators.environment.json").write_text(
        json.dumps(
            {**environment_record(), "repetitions": arguments.repetitions}, indent=2
        ),
        encoding="utf-8",
    )
