"""Measure reference caching, likelihood vectorization, and optional threading."""

from math import sin
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from aeonq.adapters.reference import ReferenceAdapter, _probability_vector
from aeonq.benchmarks.problems import independent_threshold
from aeonq.benchmarks.runner import environment_record, run_manifest
from aeonq.inference.likelihood import (
    amplified_log_likelihood_grid,
    binomial_log_likelihood,
)
from aeonq.ir import GroverPower, Sequence
from aeonq.reference import grover_for, matrix


def _time(call, repetitions: int = 3) -> float:
    observations = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        observations.append(perf_counter() - start)
    return median(observations)


def run() -> dict:
    case = independent_threshold(3, 3)
    problem = case.problem
    q = grover_for(problem)
    ir = Sequence((problem.state_preparation, GroverPower(q, 8, 3)), 3)
    backend = ReferenceAdapter()

    def original_sample() -> None:
        amplitudes = matrix(ir)[:, 0]
        probabilities = np.abs(amplitudes) ** 2
        probabilities /= probabilities.sum()
        np.random.default_rng(31).multinomial(128, probabilities)

    _probability_vector.cache_clear()
    backend.execute(ir, 128, seed=31)
    uncached_seconds = _time(lambda: [original_sample() for _ in range(500)])
    cached_seconds = _time(
        lambda: [backend.execute(ir, 128, seed=31) for _ in range(500)]
    )

    grid = np.linspace(0, np.pi / 2, 1025)
    powers = np.array([0, 1, 2, 4, 8], dtype=np.int64)
    successes = np.array([61, 73, 28, 55, 39], dtype=np.int64)
    shots = 128

    def scalar_grid():
        return np.array(
            [
                sum(
                    binomial_log_likelihood(
                        int(y), shots, sin((2 * int(k) + 1) * float(theta)) ** 2
                    )
                    for k, y in zip(powers, successes)
                )
                for theta in grid
            ]
        )

    def vector_grid():
        return amplified_log_likelihood_grid(grid, powers, successes, shots)

    np.testing.assert_allclose(scalar_grid(), vector_grid(), atol=2e-11, rtol=0)
    scalar_seconds = _time(lambda: [scalar_grid() for _ in range(25)])
    vector_seconds = _time(lambda: [vector_grid() for _ in range(25)])

    manifest = {
        "schema_version": 1,
        "profile": "performance_reference",
        "base_seed": 90171,
        "repetitions": 200,
        "backends": ["reference"],
        "problems": [{"kind": "bernoulli", "amplitude": 0.25}],
        "algorithms": {
            "mlqae": {
                "method": "mlqae",
                "options": {
                    "epsilon": 0.1,
                    "alpha": 0.05,
                    "powers": [0, 1, 2],
                    "shots_per_power": 64,
                },
            }
        },
    }
    with TemporaryDirectory() as directory:
        root = Path(directory)
        timings: dict[int, list[float]] = {1: [], 4: []}
        results: dict[int, list[dict]] = {}
        for repetition in range(3):
            for workers in (1, 4) if repetition % 2 == 0 else (4, 1):
                start = perf_counter()
                measured = run_manifest(
                    manifest,
                    root / f"workers_{workers}_{repetition}.jsonl",
                    max_workers=workers,
                )
                timings[workers].append(perf_counter() - start)
                results[workers] = measured
        serial_seconds = median(timings[1])
        threaded_seconds = median(timings[4])
        serial, threaded = results[1], results[4]
    if [(row["seed"], row["result"]["estimate"]) for row in serial] != [
        (row["seed"], row["result"]["estimate"]) for row in threaded
    ]:
        raise AssertionError("threading changed seeded estimates")
    return {
        "environment": environment_record(),
        "cached_reference_500_seconds": cached_seconds,
        "uncached_reference_500_seconds": uncached_seconds,
        "reference_cache_speedup": uncached_seconds / cached_seconds,
        "vectorized_grid_25_seconds": vector_seconds,
        "scalar_grid_25_seconds": scalar_seconds,
        "grid_speedup": scalar_seconds / vector_seconds,
        "serial_manifest_200_seconds": serial_seconds,
        "threaded_manifest_200_seconds": threaded_seconds,
        "thread_speedup": serial_seconds / threaded_seconds,
        "threaded_backend_scope": "reference only",
    }


if __name__ == "__main__":
    output = Path("benchmarks/results/raw/performance_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = run()
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
