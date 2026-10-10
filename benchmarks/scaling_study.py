"""Runtime and memory scaling of each backend with register width.

Uniform threshold problems (H on n qubits, objective x >= 2^(n-1) + 1, exact
a = (2^(n-1) - 1) / 2^n) are lowered, compiled and executed for 1000 shots at
Grover powers 1, 4 and 16 as the width grows. The truth is analytic, so no
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

from aeonq import AEProblem
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.runner import environment_record
from aeonq.ir import (
    Gate,
    GroverPower,
    ObjectivePredicate,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.reference import grover_for

WIDTHS = (2, 4, 6, 8, 10, 12)
POWERS = (1, 4, 16)
BUDGET_SECONDS = 120.0
SHOTS = 1000


def threshold_problem(n: int) -> tuple[AEProblem, float]:
    prep = StatePreparation(Sequence(tuple(Gate("H", (q,), n) for q in range(n)), n), n)
    value = 2 ** (n - 1) + 1
    objective = ObjectivePredicate("ge", qubits=tuple(range(n)), value=value)
    return AEProblem(prep, objective, tuple(range(n))), (2**n - value) / 2**n


def measure(adapter, n: int, power: int) -> dict:
    problem, truth = threshold_problem(n)
    q = grover_for(problem)
    ir = Sequence((problem.state_preparation, GroverPower(q, power, n)), n)
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    started = perf_counter()
    compiled = adapter.compile(adapter.lower(ir))
    compile_seconds = perf_counter() - started
    started = perf_counter()
    raw = adapter.execute(compiled, SHOTS, seed=17)
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


def run(output: Path) -> None:
    backends = {
        "reference": ReferenceAdapter,
        "qiskit": QiskitAdapter,
        "cirq": CirqAdapter,
        "pennylane": PennyLaneAdapter,
        "qsharp": QSharpAdapter,
    }
    rows = []
    for name, factory in backends.items():
        adapter = factory()
        stopped = None
        for n in WIDTHS:
            for power in POWERS:
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "backend": name,
                    "qubits": n,
                    "power": power,
                }
                if stopped:
                    row.update(status="not attempted", reason=stopped)
                    rows.append(row)
                    continue
                started = perf_counter()
                try:
                    row.update(measure(adapter, n, power))
                    row["status"] = "ok"
                except Exception as exc:
                    row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    stopped = f"failed at {n} qubits, power {power}"
                elapsed = perf_counter() - started
                row["total_seconds"] = elapsed
                if row["status"] == "ok" and elapsed > BUDGET_SECONDS:
                    stopped = (
                        f"exceeded {BUDGET_SECONDS:.0f}s at {n} qubits, power {power}"
                    )
                rows.append(row)
                print(name, n, power, row["status"], f"{elapsed:.2f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"environment": environment_record(), "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results/raw/scaling_study.json")
    run(Path(parser.parse_args().output))
