"""Measure naive, generic and AE-aware Qiskit compilation of Grover powers."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import tracemalloc

from qiskit.quantum_info import Operator
import numpy as np

from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.compiler import AECompiler
from aeonq.ir import GroverPower, Sequence
from aeonq.benchmarks.problems import bernoulli, independent_threshold
from aeonq.reference import grover_for


def run(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = QiskitAdapter()
    compiler = AECompiler()
    rows = []
    for case in (bernoulli(0.25), independent_threshold(2, 2)):
        prep = case.problem.state_preparation
        n = prep.n_qubits
        q = grover_for(case.problem)
        for power in (0, 1, 2, 4, 8, 16, 32):
            naive_ir = Sequence((prep,) + (q,) * power, n)
            semantic_ir = Sequence((prep, GroverPower(q, power, n)), n)
            native = adapter.lower(naive_ir)
            expected = adapter.unitary(semantic_ir)
            for label, level in (("naive_materialization", 0), ("generic_sdk", 3)):
                tracemalloc.start()
                start = perf_counter()
                compiled = adapter.compile(
                    native,
                    {
                        "optimization_level": level,
                        "seed_transpiler": 0,
                        "basis_gates": ["u", "cx"],
                    },
                )
                elapsed = perf_counter() - start
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if not np.allclose(Operator(compiled).data, expected, atol=1e-10):
                    raise AssertionError(
                        f"semantic mismatch for {case.case_id}, {label}, k={power}"
                    )
                if sum(adapter.execute(compiled, 1, seed=0).values()) != 1:
                    raise AssertionError("compiled circuit did not execute one shot")
                rows.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "sdk": "qiskit",
                        "sdk_version": adapter.capabilities().metadata["qiskit"],
                        "case_id": case.case_id,
                        "amplitude": case.a_true,
                        "power": power,
                        "pipeline": label,
                        "compile_seconds": elapsed,
                        "peak_python_bytes": peak_bytes,
                        "executable_success": True,
                        **adapter.resource_metrics(compiled),
                    }
                )
            tracemalloc.start()
            selected = compiler.compile(semantic_ir, adapter)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if not np.allclose(
                Operator(selected.compiled_program).data, expected, atol=1e-10
            ):
                raise AssertionError(
                    f"semantic mismatch for {case.case_id}, AE-aware, k={power}"
                )
            if sum(adapter.execute(selected.compiled_program, 1, seed=0).values()) != 1:
                raise AssertionError("AE-aware circuit did not execute one shot")
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "sdk": "qiskit",
                    "sdk_version": adapter.capabilities().metadata["qiskit"],
                    "case_id": case.case_id,
                    "amplitude": case.a_true,
                    "power": power,
                    "pipeline": "ae_aware",
                    "compile_seconds": selected.compile_time,
                    "peak_python_bytes": peak_bytes,
                    "executable_success": True,
                    **selected.metrics,
                }
            )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run(Path("benchmarks/results/raw/phase10_compiler.csv"))
