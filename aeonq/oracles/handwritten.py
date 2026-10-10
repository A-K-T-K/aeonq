"""A, Q, Q^2 and controlled Q written by hand in each SDK.

Written directly from the problem statements in ``cases`` with each SDK's own
idioms (Qiskit ``cry``/``mcx``, Cirq ``controlled_by``, PennyLane
``FlipSign``, Q# ``ApplyControlledOnInt`` and ``within``), and evaluated with
each SDK's own matrix facility. The constructions deliberately differ from
aeonq's synthesis where a different exact construction exists: the
distribution oracle is Z (x) Z, the threshold oracle is -S0, Q#'s -1 is
ZXZX. The global phase of Q is part of the operator throughout.

Every function returns little-endian matrices: index x has qubit q in bit q.
The control of controlled Q is qubit n, the most significant bit.
"""

from typing import Any

import numpy as np

from aeonq.oracles.cases import OracleCase


def _msb_to_little_endian(native: np.ndarray, n: int) -> np.ndarray:
    order = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
    return native[np.ix_(order, order)]


# ------------------------------------------------------------------ Qiskit


def qiskit_preparation(case: OracleCase) -> Any:
    from qiskit import QuantumCircuit

    a = case.angles
    qc = QuantumCircuit(case.n_qubits)
    if case.name == "bernoulli_0.25":
        qc.ry(a["ry"], 0)
    elif case.name == "uniform_threshold_n2_k1":
        qc.h([0, 1])
    elif case.name.startswith("distribution"):
        qc.ry(a["low"], 0)
        qc.cry(a["high_if_0"], 0, 1, ctrl_state=0)
        qc.cry(a["high_if_1"], 0, 1)
    elif case.name == "lognormal_tail_n2_k2_flag":
        qc.ry(a["high"], 1)
        qc.cry(a["low_if_0"], 1, 0, ctrl_state=0)
        qc.cry(a["low_if_1"], 1, 0)
        qc.cx(1, 2)
    else:
        raise KeyError(case.name)
    return qc


def _qiskit_zero_reflection(n: int) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    return qc


def qiskit_objective(case: OracleCase) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(case.n_qubits)
    if case.name == "bernoulli_0.25":
        qc.z(0)
    elif case.name == "uniform_threshold_n2_k1":
        # x >= 1 marks every state except |00>: the oracle is -S0.
        qc.compose(_qiskit_zero_reflection(2), inplace=True)
        qc.global_phase += np.pi
    elif case.name.startswith("distribution"):
        qc.z([0, 1])  # (-1) on |01> and |10>, i.e. x in {1, 2}
    else:
        qc.z(2)
    return qc


def qiskit_grover(case: OracleCase) -> Any:
    prep = qiskit_preparation(case)
    qc = qiskit_objective(case)
    qc.compose(prep.inverse(), inplace=True)
    qc.compose(_qiskit_zero_reflection(case.n_qubits), inplace=True)
    qc.compose(prep, inplace=True)
    qc.global_phase += np.pi
    return qc


def qiskit_matrices(case: OracleCase) -> dict[str, np.ndarray]:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    n = case.n_qubits
    grover = qiskit_grover(case)
    squared = grover.compose(grover)
    controlled = QuantumCircuit(n + 1)
    controlled.append(grover.to_gate().control(1), [n, *range(n)])
    return {
        "A": Operator(qiskit_preparation(case)).data,
        "Q": Operator(grover).data,
        "Q_power_2": Operator(squared).data,
        "controlled_Q": Operator(controlled).data,
    }


# -------------------------------------------------------------------- Cirq


def _cirq_preparation(case: OracleCase, q: list[Any]) -> list[Any]:
    import cirq

    a = case.angles
    if case.name == "bernoulli_0.25":
        return [cirq.ry(a["ry"]).on(q[0])]
    if case.name == "uniform_threshold_n2_k1":
        return [cirq.H(q[0]), cirq.H(q[1])]
    if case.name.startswith("distribution"):
        return [
            cirq.ry(a["low"]).on(q[0]),
            cirq.ry(a["high_if_0"]).on(q[1]).controlled_by(q[0], control_values=[0]),
            cirq.ry(a["high_if_1"]).on(q[1]).controlled_by(q[0]),
        ]
    if case.name == "lognormal_tail_n2_k2_flag":
        return [
            cirq.ry(a["high"]).on(q[1]),
            cirq.ry(a["low_if_0"]).on(q[0]).controlled_by(q[1], control_values=[0]),
            cirq.ry(a["low_if_1"]).on(q[0]).controlled_by(q[1]),
            cirq.CNOT(q[1], q[2]),
        ]
    raise KeyError(case.name)


def _cirq_zero_reflection(q: list[Any]) -> list[Any]:
    import cirq

    flips = [cirq.X(qubit) for qubit in q]
    middle = [cirq.Z(q[-1]).controlled_by(*q[:-1])] if len(q) > 1 else [cirq.Z(q[0])]
    return flips + middle + flips


def _cirq_objective(case: OracleCase, q: list[Any]) -> list[Any]:
    import cirq

    if case.name == "bernoulli_0.25":
        return [cirq.Z(q[0])]
    if case.name == "uniform_threshold_n2_k1":
        return _cirq_zero_reflection(q) + [cirq.global_phase_operation(-1)]
    if case.name.startswith("distribution"):
        return [cirq.Z(q[0]), cirq.Z(q[1])]
    return [cirq.Z(q[2])]


def cirq_grover(case: OracleCase) -> Any:
    import cirq

    q = cirq.LineQubit.range(case.n_qubits)
    prep = cirq.Circuit(_cirq_preparation(case, q))
    return (
        cirq.Circuit(_cirq_objective(case, q))
        + cirq.inverse(prep)
        + cirq.Circuit(_cirq_zero_reflection(q))
        + prep
        + cirq.Circuit(cirq.global_phase_operation(-1))
    )


def cirq_matrices(case: OracleCase) -> dict[str, np.ndarray]:
    import cirq

    n = case.n_qubits
    q = cirq.LineQubit.range(n + 1)
    order = list(reversed(q[:n]))
    grover = cirq_grover(case)

    def matrix(circuit: Any, qubits: list[Any]) -> np.ndarray:
        return np.asarray(
            circuit.unitary(qubit_order=qubits, qubits_that_should_be_present=qubits)
        )

    controlled = cirq.Circuit(
        cirq.CircuitOperation(grover.freeze()).controlled_by(q[n])
    )
    return {
        "A": matrix(cirq.Circuit(_cirq_preparation(case, q)), order),
        "Q": matrix(grover, order),
        "Q_power_2": matrix(grover + grover, order),
        "controlled_Q": matrix(controlled, [q[n], *order]),
    }


# --------------------------------------------------------------- PennyLane


def _pennylane_preparation(case: OracleCase) -> None:
    import pennylane as qml

    a = case.angles
    if case.name == "bernoulli_0.25":
        qml.RY(a["ry"], wires=0)
    elif case.name == "uniform_threshold_n2_k1":
        qml.Hadamard(0)
        qml.Hadamard(1)
    elif case.name.startswith("distribution"):
        qml.RY(a["low"], wires=0)
        qml.ctrl(qml.RY, control=0, control_values=[0])(a["high_if_0"], wires=1)
        qml.CRY(a["high_if_1"], wires=[0, 1])
    elif case.name == "lognormal_tail_n2_k2_flag":
        qml.RY(a["high"], wires=1)
        qml.ctrl(qml.RY, control=1, control_values=[0])(a["low_if_0"], wires=0)
        qml.CRY(a["low_if_1"], wires=[1, 0])
        qml.CNOT(wires=[1, 2])
    else:
        raise KeyError(case.name)


def _pennylane_grover(case: OracleCase) -> None:
    import pennylane as qml

    n = case.n_qubits
    # PennyLane documents GlobalPhase(phi) as exp(-i phi); phi = pi gives -1.
    if case.name == "bernoulli_0.25":
        qml.PauliZ(0)
    elif case.name == "uniform_threshold_n2_k1":
        qml.FlipSign([0, 0], wires=[0, 1])
        qml.GlobalPhase(np.pi)
    elif case.name.startswith("distribution"):
        qml.PauliZ(0)
        qml.PauliZ(1)
    else:
        qml.PauliZ(2)
    qml.adjoint(_pennylane_preparation)(case)
    qml.FlipSign([0] * n, wires=list(range(n)))
    _pennylane_preparation(case)
    qml.GlobalPhase(np.pi)


def pennylane_matrices(case: OracleCase) -> dict[str, np.ndarray]:
    import pennylane as qml

    n = case.n_qubits
    order = list(reversed(range(n)))

    def squared(c: OracleCase) -> None:
        _pennylane_grover(c)
        _pennylane_grover(c)

    return {
        "A": np.asarray(qml.matrix(_pennylane_preparation, wire_order=order)(case)),
        "Q": np.asarray(qml.matrix(_pennylane_grover, wire_order=order)(case)),
        "Q_power_2": np.asarray(qml.matrix(squared, wire_order=order)(case)),
        "controlled_Q": np.asarray(
            qml.matrix(qml.ctrl(_pennylane_grover, control=n), wire_order=[n, *order])(
                case
            )
        ),
    }


# ---------------------------------------------------------------------- Q#


def qsharp_source(case: OracleCase, prefix: str) -> str:
    a = {key: repr(value) for key, value in case.angles.items()}
    if case.name == "bernoulli_0.25":
        prep = f"Ry({a['ry']}, qs[0]);"
        objective = "Z(qs[0]);"
    elif case.name == "uniform_threshold_n2_k1":
        prep = "H(qs[0]); H(qs[1]);"
        objective = f"{prefix}Zero(qs); {prefix}MinusOne(qs[0]);"
    elif case.name.startswith("distribution"):
        prep = (
            f"Ry({a['low']}, qs[0]); "
            f"ApplyControlledOnInt(0, Ry({a['high_if_0']}, _), [qs[0]], qs[1]); "
            f"Controlled Ry([qs[0]], ({a['high_if_1']}, qs[1]));"
        )
        objective = "Z(qs[0]); Z(qs[1]);"
    elif case.name == "lognormal_tail_n2_k2_flag":
        prep = (
            f"Ry({a['high']}, qs[1]); "
            f"ApplyControlledOnInt(0, Ry({a['low_if_0']}, _), [qs[1]], qs[0]); "
            f"Controlled Ry([qs[1]], ({a['low_if_1']}, qs[0])); "
            "CNOT(qs[1], qs[2]);"
        )
        objective = "Z(qs[2]);"
    else:
        raise KeyError(case.name)
    n = case.n_qubits
    adjctl = "Unit is Adj + Ctl"
    return f"""
import Std.Arrays.*;
import Std.Canon.*;
operation {prefix}Prep(qs : Qubit[]) : {adjctl} {{ {prep} }}
operation {prefix}MinusOne(q : Qubit) : {adjctl} {{ Z(q); X(q); Z(q); X(q); }}
operation {prefix}Zero(qs : Qubit[]) : {adjctl} {{
    within {{ ApplyToEachCA(X, qs); }} apply {{ Controlled Z(Most(qs), Tail(qs)); }}
}}
operation {prefix}Objective(qs : Qubit[]) : {adjctl} {{ {objective} }}
operation {prefix}Grover(qs : Qubit[]) : {adjctl} {{
    {prefix}Objective(qs); Adjoint {prefix}Prep(qs); {prefix}Zero(qs);
    {prefix}Prep(qs); {prefix}MinusOne(qs[0]);
}}
operation {prefix}Grover2(qs : Qubit[]) : Unit {{ {prefix}Grover(qs); {prefix}Grover(qs); }}
operation {prefix}ControlledGrover(qs : Qubit[]) : Unit {{
    Controlled {prefix}Grover([qs[{n}]], qs[0..{n - 1}]);
}}
"""


_QSHARP_DEFINED: set[str] = set()


def qsharp_matrices(case: OracleCase) -> dict[str, np.ndarray]:
    from qdk import qsharp

    n = case.n_qubits
    prefix = "Oracle" + "".join(ch for ch in case.name.title() if ch.isalnum())
    if prefix not in _QSHARP_DEFINED:
        qsharp.eval(qsharp_source(case, prefix))
        _QSHARP_DEFINED.add(prefix)

    def matrix(operation: str, width: int) -> np.ndarray:
        native = np.asarray(qsharp.dump_operation(operation, width), dtype=complex)
        return _msb_to_little_endian(native, width)

    return {
        "A": matrix(f"{prefix}Prep", n),
        "Q": matrix(f"{prefix}Grover", n),
        "Q_power_2": matrix(f"{prefix}Grover2", n),
        "controlled_Q": matrix(f"{prefix}ControlledGrover", n + 1),
    }


HANDWRITTEN = {
    "qiskit": qiskit_matrices,
    "cirq": cirq_matrices,
    "pennylane": pennylane_matrices,
    "qsharp": qsharp_matrices,
}
# dump_operation prints about six decimals per entry.
PRECISION = {"qiskit": 1e-9, "cirq": 1e-9, "pennylane": 1e-9, "qsharp": 1e-6}
