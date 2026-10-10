"""aeonq estimators versus the native qiskit-algorithms estimators.

Both libraries estimate the same A (the aeonq IR lowered once to a Qiskit
circuit), with each library's documented parameters at the same nominal 95%
confidence. The native Grover operator is qiskit-algorithms' own
construction, so only the estimator, its inference and its bookkeeping differ.

Sampling is exact-statevector on both sides: Qiskit's ``StatevectorSampler``
for the native estimators and aeonq's reference backend for aeonq. The native
MLAE runs without a transpiler because qiskit-algorithms 0.4.0 MLAE transpiles
only the first circuit when one is supplied (mlae.py, construct_circuits).

Query counts use one convention, calls to A or A†: a shot of A·Q^k costs 2k+1.
qiskit-algorithms counts k per shot for IAE, MLAE and canonical AE (iae.py
``num_oracle_queries += shots * k``; mlae.py ``shots * sum(k)``; ae.py
``shots * (M - 1)``), so those are converted as 2·reported + total shots. FAE
already reports (2k+1)·shots (fae.py ``(2 * k + 1) * shots``).
"""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)

from aeonq import estimate
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import (
    BenchmarkProblem,
    bernoulli,
    lognormal_tail,
    strike_for_index,
)
from aeonq.benchmarks.runner import environment_record, map_trials, open_trial_log
from aeonq.benchmarks.evidence import trial_seed

SHOTS = 512
ALPHA = 0.05
EPSILON = 0.01
POWERS = [0, 1, 2, 4, 8]
FAE_ITERATIONS = 4
EVALUATION_QUBITS = 4
AMPLITUDES = (
    0.0,
    1e-6,
    1e-3,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.99,
    0.999,
    1 - 1e-6,
    1.0,
)


class CountingSampler:
    """Delegate to StatevectorSampler and count every shot it executes."""

    def __init__(self, seed: int) -> None:
        # A Generator, not the int: StatevectorSampler reseeds every circuit
        # with an int seed, so all circuits of one estimator run (five MLAE
        # powers, every IAE/FAE round) would draw identical random streams and
        # their sampling errors would be perfectly correlated. A shared
        # Generator keeps the stream running across circuits and calls.
        self._inner = StatevectorSampler(
            default_shots=SHOTS, seed=np.random.default_rng(seed)
        )
        self.total_shots = 0

    @property
    def default_shots(self) -> int:
        return self._inner.default_shots

    def run(self, pubs: Any, *, shots: int | None = None) -> Any:
        pubs = list(pubs)
        for pub in pubs:
            pub_shots = pub[2] if isinstance(pub, tuple) and len(pub) > 2 else None
            self.total_shots += int(pub_shots or shots or SHOTS)
        return self._inner.run(pubs, shots=shots)


@lru_cache(maxsize=1)
def cases() -> list[BenchmarkProblem]:
    return [bernoulli(a) for a in AMPLITUDES] + [
        lognormal_tail(3, strike_for_index(3, 5), objective_ancilla=True),
        lognormal_tail(4, strike_for_index(4, 10), objective_ancilla=True),
    ]


def _native_problem(case: BenchmarkProblem) -> EstimationProblem:
    problem = case.problem
    circuit = QiskitAdapter().lower(problem.state_preparation)
    # Bernoulli objective is qubit 0; the case study flags qubit n.
    objective = problem.objective.qubits[0]
    return EstimationProblem(circuit, objective_qubits=[objective])


def native_trial(
    case: BenchmarkProblem, method: str, seed: int, *, mlae_interval: str = "fisher"
) -> dict:
    sampler = CountingSampler(seed)
    if method == "iqae":
        algorithm: Any = IterativeAmplitudeEstimation(
            EPSILON, ALPHA, confint_method="beta", sampler=sampler
        )
    elif method == "mlqae":
        algorithm = MaximumLikelihoodAmplitudeEstimation(POWERS, sampler=sampler)
    elif method == "faster_ae":
        algorithm = FasterAmplitudeEstimation(ALPHA, FAE_ITERATIONS, sampler=sampler)
    else:
        algorithm = AmplitudeEstimation(EVALUATION_QUBITS, sampler=sampler)
    problem = _native_problem(case)
    started = perf_counter()
    result = algorithm.estimate(problem)
    seconds = perf_counter() - started
    reported = int(result.num_oracle_queries)
    queries = reported if method == "faster_ae" else 2 * reported + sampler.total_shots
    if method == "mlqae" and mlae_interval != "fisher":
        # Same samples and estimate; only the interval theory differs.
        interval = [
            float(x)
            for x in MaximumLikelihoodAmplitudeEstimation.compute_confidence_interval(
                result, ALPHA, kind=mlae_interval
            )
        ]
        interval_kind = f"{mlae_interval} (sensitivity check, not the default)"
    else:
        interval = [float(x) for x in result.confidence_interval]
        interval_kind = {
            "iqae": "Clopper-Pearson per round (confint_method='beta')",
            "mlqae": "asymptotic Fisher information (library default)",
            "faster_ae": "Chernoff-based (library default)",
            "canonical_qae": "likelihood ratio around the MLE (library default)",
        }[method]
    return {
        "estimate": float(result.estimation),
        "mle": float(result.mle) if method == "canonical_qae" else None,
        "interval": interval,
        "interval_kind": interval_kind,
        "shots": sampler.total_shots,
        "queries": queries,
        "reported_queries": reported,
        "seconds": seconds,
        "status": "native",
    }


def aeonq_trial(case: BenchmarkProblem, method: str, seed: int) -> dict:
    options: dict[str, Any] = {
        "iqae": {"epsilon": EPSILON, "shots_per_round": SHOTS},
        "mlqae": {
            "epsilon": EPSILON,
            "powers": tuple(POWERS),
            "shots_per_power": SHOTS,
        },
        "faster_ae": {"epsilon": EPSILON, "max_iterations": FAE_ITERATIONS},
        "canonical_qae": {"evaluation_qubits": EVALUATION_QUBITS, "shots": SHOTS},
    }[method]
    started = perf_counter()
    result = estimate(
        case.problem,
        algorithm=method,
        backend=ReferenceAdapter(),
        alpha=ALPHA,
        seed=seed,
        **options,
    )
    seconds = perf_counter() - started
    has_interval = result.interval_type != "no inferential interval"
    return {
        "estimate": result.estimate,
        "mle": None,
        "interval": list(result.confidence_or_credible_interval)
        if has_interval
        else None,
        "interval_kind": result.interval_type,
        "shots": result.total_shots,
        "queries": result.total_oracle_queries,
        "reported_queries": result.total_oracle_queries,
        "seconds": seconds,
        "status": result.convergence_status,
    }


def native_trial_likelihood_ratio(
    case: BenchmarkProblem, method: str, seed: int
) -> dict:
    return native_trial(case, method, seed, mlae_interval="likelihood_ratio")


METHODS = ("iqae", "mlqae", "faster_ae", "canonical_qae")
LIBRARIES = {
    "aeonq": aeonq_trial,
    "qiskit_algorithms": native_trial,
    # Sensitivity check: the same MLAE runs with the likelihood-ratio interval
    # separate the effect of the default Fisher interval from the estimator.
    "qiskit_algorithms_lr": native_trial_likelihood_ratio,
}
LIBRARY_METHODS = {"qiskit_algorithms_lr": ("mlqae",)}


def configuration(method: str) -> dict:
    return {
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "default_shots": SHOTS,
        "powers": POWERS if method == "mlqae" else None,
        "max_iterations": FAE_ITERATIONS if method == "faster_ae" else None,
        "evaluation_qubits": EVALUATION_QUBITS if method == "canonical_qae" else None,
    }


def seed_for(case_id: str, method: str, library: str, trial: int) -> int:
    # LR is a sensitivity reanalysis of the *same* native observations, not
    # an independent library comparison. All primary libraries are separate.
    primary = "qiskit_algorithms" if library == "qiskit_algorithms_lr" else library
    return trial_seed("native_baselines", case_id, method, primary, trial, 810_001)


def trial_row(task: tuple[int, str, str, int]) -> dict[str, Any]:
    """One trial, identified by (case index, method, library, trial)."""
    case_index, method, library, trial = task
    case = cases()[case_index]
    seed = seed_for(case.case_id, method, library, trial)
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id,
        "a_true": case.a_true,
        "method": method,
        "library": library,
        "trial": trial,
        "seed": seed,
        "configuration": configuration(method),
        "environment": environment_record(),
    }
    try:
        row.update(LIBRARIES[library](case, method, seed))
        interval = row["interval"]
        row["covered"] = (
            None if interval is None else interval[0] <= case.a_true <= interval[1]
        )
        row["error"] = None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run(
    output: Path, repetitions: int, *, resume: bool = False, workers: int = 1
) -> None:
    from aeonq.benchmarks.validation import validate_extended, read_records

    if resume and output.exists():
        # open_trial_log recovers only a truncated final record; validate the
        # complete prefix before skipping any recorded identities.
        prefix = output.read_text(encoding="utf-8").splitlines()
        rows = []
        for i, line in enumerate(prefix):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(prefix) - 1:
                    raise
        validate_extended(rows, "native_baselines", repetitions, complete=False)
    environment = environment_record()
    stream, done = open_trial_log(
        output, ("case_id", "method", "library", "trial"), resume=resume
    )
    tasks = [
        (case_index, method, library, trial)
        for case_index, case in enumerate(cases())
        for method in METHODS
        for library in LIBRARIES
        if method in LIBRARY_METHODS.get(library, METHODS)
        for trial in range(repetitions)
        if (case.case_id, method, library, trial) not in done
    ]
    with stream:
        for row in map_trials(trial_row, tasks, workers=workers):
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if row["trial"] == repetitions - 1:
                print(row["case_id"], row["method"], row["library"], flush=True)
    output.with_suffix(".environment.json").write_text(
        json.dumps({**environment, "repetitions": repetitions}, indent=2),
        encoding="utf-8",
    )
    validate_extended(read_records(output), "native_baselines", repetitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument(
        "--output", default="benchmarks/results/raw/native_baselines.jsonl"
    )
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
