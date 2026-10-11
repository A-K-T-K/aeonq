"""Runtime and memory scaling of each backend with register width.

Uniform threshold problems (H on n qubits, objective x >= 2^(n-1) + 1, exact
a = (2^(n-1) - 1) / 2^n) are lowered, compiled and executed for 1000 shots at
Grover powers 1, 4 and 16 as the width grows, repeated with a distinct
execution seed per repetition. The truth is analytic, so no
exponential reference computation limits the width. A backend is not tried at
larger widths once a configuration exceeds its declared limit, fails, or
takes longer than the per-configuration budget. Peak resident memory is the
process RSS increase measured with psutil.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter

import psutil
from qdk import qsharp

from aeonq import AEProblem
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter, clear_evaluation_cache
from aeonq.adapters.reference import ReferenceAdapter, _probability_vector
from aeonq.benchmarks.runner import environment_record
from aeonq.ir import (
    Gate,
    GroverPower,
    ObjectivePredicate,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.reference import _objective_signs, grover_for
from aeonq.synthesis.reflection import (
    synthesize_objective_reflection,
    synthesize_zero_reflection,
)

WIDTHS = (2, 4, 6, 8, 10, 12)
POWERS = (1, 4, 16)
BUDGET_SECONDS = 120.0
SHOTS = 1000
REPETITIONS = 10
SEED_BASE = 17


def threshold_problem(n: int) -> tuple[AEProblem, float]:
    prep = StatePreparation(Sequence(tuple(Gate("H", (q,), n) for q in range(n)), n), n)
    value = 2 ** (n - 1) + 1
    objective = ObjectivePredicate("ge", qubits=tuple(range(n)), value=value)
    return AEProblem(prep, objective, tuple(range(n))), (2**n - value) / 2**n


def _clear_caches() -> None:
    """Make every repetition pay for synthesis, evaluation and simulation again."""
    _probability_vector.cache_clear()
    _objective_signs.cache_clear()
    synthesize_objective_reflection.cache_clear()
    synthesize_zero_reflection.cache_clear()
    clear_evaluation_cache()


def measure(adapter, n: int, power: int, seed: int) -> dict:
    problem, truth = threshold_problem(n)
    q = grover_for(problem)
    ir = Sequence((problem.state_preparation, GroverPower(q, power, n)), n)
    process = psutil.Process()
    _clear_caches()
    gc.collect()
    baseline = process.memory_info().rss
    started = perf_counter()
    compiled = adapter.compile(adapter.lower(ir))
    compile_seconds = perf_counter() - started
    started = perf_counter()
    raw = adapter.execute(compiled, SHOTS, seed=seed)
    execute_seconds = perf_counter() - started
    result = adapter.normalize_result(
        raw, RegisterMap(tuple(range(n)), tuple(range(n)))
    )
    peak = process.memory_info().rss
    metrics = adapter.resource_metrics(compiled)
    return {
        "a_true": truth,
        "compile_seconds": compile_seconds,
        "execute_seconds": execute_seconds,
        "rss_increase_bytes": max(0, peak - baseline),
        "shots_returned": result.shots,
        "two_qubit_gates": metrics.get("two_qubit_gates"),
        "multi_qubit_gates": metrics.get("multi_qubit_gates"),
        "depth": metrics.get("depth"),
    }


def run(output: Path, repetitions: int) -> None:
    backends = {
        "reference": ReferenceAdapter,
        "qiskit": QiskitAdapter,
        "cirq": CirqAdapter,
        "pennylane": PennyLaneAdapter,
        "qsharp": QSharpAdapter,
    }
    adapters = {name: factory() for name, factory in backends.items()}
    rows = []
    # Repetitions are the outer loop so that slow drift in machine load is
    # spread over every backend instead of concentrating on one.
    for repetition in range(repetitions):
        seed = SEED_BASE + repetition
        for name, adapter in adapters.items():
            if name == "qsharp":
                # A fresh interpreter per repetition, as in a new process;
                # re-evaluated programs otherwise accumulate in one session.
                qsharp.init()
            stopped = None
            for n in WIDTHS:
                for power in POWERS:
                    row = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "backend": name,
                        "qubits": n,
                        "power": power,
                        "repetition": repetition,
                        "seed": seed,
                    }
                    if stopped:
                        row.update(status="not attempted", reason=stopped)
                        rows.append(row)
                        continue
                    started = perf_counter()
                    try:
                        row.update(measure(adapter, n, power, seed))
                        row["status"] = "ok"
                    except Exception as exc:
                        row.update(
                            status="failed", reason=f"{type(exc).__name__}: {exc}"
                        )
                        stopped = f"failed at {n} qubits, power {power}"
                    elapsed = perf_counter() - started
                    row["total_seconds"] = elapsed
                    if row["status"] == "ok" and elapsed > BUDGET_SECONDS:
                        stopped = (
                            f"exceeded {BUDGET_SECONDS:.0f}s at {n} qubits, "
                            f"power {power}"
                        )
                    rows.append(row)
                    print(
                        repetition,
                        name,
                        n,
                        power,
                        row["status"],
                        f"{elapsed:.2f}s",
                        flush=True,
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"environment": environment_record(), "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/raw/scaling_study.json")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    arguments = parser.parse_args()
    run(Path(arguments.output), arguments.repetitions)
