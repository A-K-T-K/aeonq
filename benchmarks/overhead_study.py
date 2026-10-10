"""Abstraction overhead: aeonq lowering versus hand-written native code.

Part 1 builds A·Q^k twice per SDK: once from aeonq IR through the adapter,
once by hand in the SDK's own idiom (Qiskit uses its ``grover_operator``
library function). It records build and compile time (median over
repetitions, with aeonq's synthesis caches cleared for a cold build), native
gate counts, and the exact and phase-aligned unitary distance of each circuit
from the reference semantics.

Part 2 splits the wall time of complete estimator runs into SDK compilation,
SDK execution, and everything else (aeonq inference, bookkeeping, validation).
"""

import argparse
from datetime import datetime, timezone
import json
from math import asin, pi, sqrt
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable

import cirq
import numpy as np
import pennylane as qml
from qdk import qsharp
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator
from qiskit.quantum_info import Operator as QiskitOperator

from aeonq import AEProblem, estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter, clear_evaluation_cache
from aeonq.benchmarks.problems import lognormal_tail, strike_for_index
from aeonq.benchmarks.runner import environment_record
from aeonq.ir import Gate, GroverPower, ObjectivePredicate, Sequence, StatePreparation
from aeonq.reference import grover_for, matrix
from aeonq.synthesis import synthesize_objective_reflection, synthesize_zero_reflection
from aeonq.utils.numerics import aligned_operator_distance

BERNOULLI_ANGLE = 2 * asin(sqrt(0.3))
ANGLES_2Q = (0.8, 1.3)


def aeonq_problem(case: str) -> AEProblem:
    if case == "bernoulli_0.3":
        prep = StatePreparation(Gate("RY", (0,), 1, BERNOULLI_ANGLE), 1)
        return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    ops = tuple(Gate("RY", (q,), 2, a) for q, a in enumerate(ANGLES_2Q))
    prep = StatePreparation(Sequence(ops, 2), 2)
    # x >= 2 on two qubits, i.e. the high bit is set.
    return AEProblem(prep, ObjectivePredicate("ge", qubits=(0, 1), value=2), (0, 1))


# Hand-written native circuits: Q = -A S0 A† Sχ, applied k times after A.


def qiskit_native(case: str, k: int) -> QuantumCircuit:
    n = 1 if case == "bernoulli_0.3" else 2
    a = QuantumCircuit(n)
    oracle = QuantumCircuit(n)
    if n == 1:
        a.ry(BERNOULLI_ANGLE, 0)
        oracle.z(0)
    else:
        a.ry(ANGLES_2Q[0], 0)
        a.ry(ANGLES_2Q[1], 1)
        oracle.z(1)
    q = grover_operator(oracle, state_preparation=a)
    circuit = a.copy()
    for _ in range(k):
        circuit.compose(q, inplace=True)
    return circuit


def cirq_native(case: str, k: int) -> cirq.Circuit:
    n = 1 if case == "bernoulli_0.3" else 2
    qs = cirq.LineQubit.range(n)
    angles = (BERNOULLI_ANGLE,) if n == 1 else ANGLES_2Q
    prep = [cirq.ry(angle).on(q) for angle, q in zip(angles, qs)]
    s0 = [
        cirq.X.on_each(*qs),
        cirq.Z(qs[0]) if n == 1 else cirq.CZ(*qs),
        cirq.X.on_each(*qs),
    ]
    grover = [
        cirq.Z(qs[-1]),
        cirq.inverse(prep),
        s0,
        prep,
        cirq.global_phase_operation(-1),
    ]
    return cirq.Circuit(prep, [grover] * k)


def pennylane_native(case: str, k: int) -> qml.tape.QuantumScript:
    n = 1 if case == "bernoulli_0.3" else 2
    angles = (BERNOULLI_ANGLE,) if n == 1 else ANGLES_2Q

    def prep() -> None:
        for wire, angle in enumerate(angles):
            qml.RY(angle, wires=wire)

    with qml.queuing.AnnotatedQueue() as queue:
        prep()
        for _ in range(k):
            qml.PauliZ(wires=n - 1)
            qml.adjoint(prep)()
            for wire in range(n):
                qml.PauliX(wires=wire)
            qml.PauliZ(wires=0) if n == 1 else qml.CZ(wires=[0, 1])
            for wire in range(n):
                qml.PauliX(wires=wire)
            prep()
            qml.GlobalPhase(pi)
    return qml.tape.QuantumScript.from_queue(queue)


def qsharp_native_source(case: str, k: int, name: str) -> str:
    n = 1 if case == "bernoulli_0.3" else 2
    angles = (BERNOULLI_ANGLE,) if n == 1 else ANGLES_2Q
    prep = " ".join(f"Ry({a!r}, qs[{i}]);" for i, a in enumerate(angles))
    reflect = "Z(qs[0]);" if n == 1 else "CZ(qs[0], qs[1]);"
    return f"""
operation {name}Prep(qs : Qubit[]) : Unit is Adj + Ctl {{ {prep} }}
operation {name}Grover(qs : Qubit[]) : Unit is Adj + Ctl {{
    Z(qs[{n - 1}]);
    Adjoint {name}Prep(qs);
    within {{ ApplyToEachCA(X, qs); }} apply {{ {reflect} }}
    {name}Prep(qs);
    R(PauliI, {-2 * pi!r}, qs[0]);
}}
operation {name}(qs : Qubit[]) : Unit is Adj + Ctl {{
    {name}Prep(qs);
    for _ in 1..{k} {{ {name}Grover(qs); }}
}}"""


def _qsharp_unitary(entry: str, n: int) -> np.ndarray:
    native = np.asarray(qsharp.dump_operation(entry, n), dtype=complex)
    permutation = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
    return native[np.ix_(permutation, permutation)]


def _unitary(sdk: str, program: Any, n: int) -> np.ndarray:
    if sdk == "qiskit":
        return np.asarray(QiskitOperator(program).data)
    if sdk == "cirq":
        order = list(reversed(cirq.LineQubit.range(n)))
        return np.asarray(program.unitary(qubit_order=order))
    if sdk == "pennylane":
        return np.asarray(qml.matrix(program, wire_order=list(reversed(range(n)))))
    return _qsharp_unitary(program, n)


def _timed(
    call: Callable[[], Any], repetitions: int, reset: Callable[[], None]
) -> tuple[float, Any]:
    samples, value = [], None
    for _ in range(repetitions):
        reset()
        started = perf_counter()
        value = call()
        samples.append(perf_counter() - started)
    return median(samples), value


def _clear_caches() -> None:
    synthesize_objective_reflection.cache_clear()
    synthesize_zero_reflection.cache_clear()


def construction_rows(repetitions: int) -> list[dict]:
    adapters = {
        "qiskit": QiskitAdapter(),
        "cirq": CirqAdapter(),
        "pennylane": PennyLaneAdapter(),
        "qsharp": QSharpAdapter(),
    }
    rows = []
    for case in ("bernoulli_0.3", "uniform_ge2_n2"):
        problem = aeonq_problem(case)
        n = problem.state_preparation.n_qubits
        for k in (1, 4, 16):

            def build_ir():
                q = grover_for(problem)
                return Sequence((problem.state_preparation, GroverPower(q, k, n)), n)

            reference = matrix(build_ir())
            for sdk, adapter in adapters.items():
                counter = {"value": 0}

                def native_build():
                    if sdk == "qiskit":
                        return qiskit_native(case, k)
                    if sdk == "cirq":
                        return cirq_native(case, k)
                    if sdk == "pennylane":
                        return pennylane_native(case, k)
                    counter["value"] += 1
                    name = f"Native{case.replace('.', '_')}K{k}R{counter['value']}"
                    qsharp.eval(qsharp_native_source(case, k, name))
                    return name

                aeonq_seconds, lowered = _timed(
                    lambda: adapter.lower(build_ir()), repetitions, _clear_caches
                )
                native_seconds, native = _timed(native_build, repetitions, lambda: None)
                compile_aeonq, compiled_aeonq = _timed(
                    lambda: adapter.compile(lowered),
                    repetitions,
                    clear_evaluation_cache,
                )
                compile_native, compiled_native = _timed(
                    lambda: adapter.compile(native) if sdk != "qsharp" else native,
                    repetitions,
                    lambda: None,
                )
                aeonq_unitary = adapter.unitary(build_ir())
                native_unitary = _unitary(sdk, native, n)
                aeonq_metrics = adapter.resource_metrics(compiled_aeonq)
                native_metrics = (
                    adapter.resource_metrics(compiled_native) if sdk != "qsharp" else {}
                )
                rows.append(
                    {
                        "sdk": sdk,
                        "case_id": case,
                        "power": k,
                        "aeonq_lower_seconds": aeonq_seconds,
                        "native_build_seconds": native_seconds,
                        "aeonq_compile_seconds": compile_aeonq,
                        "native_compile_seconds": compile_native,
                        # Build plus compile; for Q# evaluation happens in either.
                        "total_overhead_seconds": (aeonq_seconds + compile_aeonq)
                        - (native_seconds + compile_native),
                        "aeonq_two_qubit_gates": aeonq_metrics.get("two_qubit_gates"),
                        "native_two_qubit_gates": native_metrics.get("two_qubit_gates"),
                        "aeonq_depth": aeonq_metrics.get("depth"),
                        "native_depth": native_metrics.get("depth"),
                        "aeonq_exact_distance": float(
                            np.max(np.abs(aeonq_unitary - reference))
                        ),
                        "native_exact_distance": float(
                            np.max(np.abs(native_unitary - reference))
                        ),
                        "native_phase_aligned_distance": aligned_operator_distance(
                            native_unitary, reference
                        ),
                    }
                )
                print(rows[-1], flush=True)
    return rows


def breakdown_rows(repetitions: int) -> list[dict]:
    cases = {
        "bernoulli_0.3": aeonq_problem("bernoulli_0.3"),
        "lognormal_tail_n3_flag": lognormal_tail(
            3, strike_for_index(3, 5), objective_ancilla=True
        ).problem,
    }
    adapters = {
        "qiskit": QiskitAdapter(),
        "cirq": CirqAdapter(),
        "pennylane": PennyLaneAdapter(),
        "qsharp": QSharpAdapter(),
    }
    rows = []
    for case, problem in cases.items():
        for sdk, adapter in adapters.items():
            for method, options in (
                ("iqae", {"epsilon": 0.01, "shots_per_round": 512, "max_power": 16}),
                (
                    "mlqae",
                    {
                        "epsilon": 0.01,
                        "powers": (0, 1, 2, 4, 8),
                        "shots_per_power": 512,
                    },
                ),
            ):
                for trial in range(repetitions):
                    # Each trial pays its own Q# evaluations, as every other
                    # SDK pays its own compilation per trial.
                    clear_evaluation_cache()
                    result = estimate(
                        problem,
                        algorithm=method,
                        backend=adapter,
                        seed=5003 + trial,
                        **options,
                    )
                    other = (
                        result.wall_clock_time
                        - result.compile_time
                        - result.execution_time
                    )
                    rows.append(
                        {
                            "sdk": sdk,
                            "case_id": case,
                            "method": method,
                            "trial": trial,
                            "wall_seconds": result.wall_clock_time,
                            "sdk_compile_seconds": result.compile_time,
                            "sdk_execute_seconds": result.execution_time,
                            "aeonq_other_seconds": other,
                            "aeonq_other_fraction": other / result.wall_clock_time,
                        }
                    )
                print(case, sdk, method, flush=True)
    return rows


def run(output: Path, repetitions: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_record(),
        "construction": construction_rows(repetitions),
        "estimator_breakdown": breakdown_rows(repetitions),
    }
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output", default="benchmarks/results/raw/overhead_study.json"
    )
    arguments = parser.parse_args()
    run(Path(arguments.output), arguments.repetitions)
