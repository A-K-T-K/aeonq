"""Generate exact small-system operator conformance rows."""

import numpy as np

from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.ir import Adjoint, ControlledOperator, GlobalPhase, GroverPower
from aeonq.ir.nodes import Operator
from aeonq.reference import grover_for, matrix
from aeonq.benchmarks.problems import bernoulli, independent_threshold
from aeonq.utils.numerics import aligned_operator_distance


def measure_operator_conformance() -> list[dict[str, object]]:
    backends = {
        "qiskit": QiskitAdapter(),
        "cirq": CirqAdapter(),
        "pennylane": PennyLaneAdapter(),
        "qsharp": QSharpAdapter(),
    }
    rows: list[dict[str, object]] = []
    for case in (bernoulli(0.25), independent_threshold(2, 2)):
        problem = case.problem
        q = grover_for(problem)
        n = q.n_qubits
        nodes: dict[str, Operator] = {
            "A": problem.state_preparation,
            "A_adjoint": Adjoint(problem.state_preparation, n),
            "S0": q.zero_reflection,
            "Schi": q.objective_reflection,
            "Q": q,
            "Q_power_2": GroverPower(q, 2, n),
        }
        if n == 1:
            nodes["controlled_phase"] = ControlledOperator(
                GlobalPhase(np.pi, 1), (1,), (1,), (0,), 2
            )
            nodes["controlled_Q"] = ControlledOperator(q, (1,), (1,), (0,), 2)
        for operator_name, node in nodes.items():
            reference = matrix(node)
            for backend_name, backend in backends.items():
                native = backend.unitary(node)
                exact_distance = float(np.linalg.norm(native - reference))
                phase_distance = aligned_operator_distance(native, reference)
                # QDK's dump_operation matrix output has about six decimal
                # digits on this local version; preserve the measured error.
                tolerance = 1e-6 if backend_name == "qsharp" else 1e-9
                rows.append(
                    {
                        "case_id": case.case_id,
                        "operator": operator_name,
                        "backend": backend_name,
                        "sdk_version": next(
                            iter(backend.capabilities().metadata.values())
                        ),
                        "qubits": node.n_qubits,
                        "exact_distance": exact_distance,
                        "phase_aligned_distance": phase_distance,
                        "exact_pass": exact_distance < tolerance,
                        "tolerance": tolerance,
                    }
                )
    return rows
