"""Read native SDK programs into ``numpy_sim`` operations.

This lets the NumPy simulator evaluate the programs that aeonq's adapters
produce, without the SDK's own simulator or matrix code. Each SDK first
decomposes its program into a small documented gate set; every gate in that
set is then given its textbook matrix here:

* Qiskit: ``transpile`` to {u, cx} plus the circuit's global phase.
* Cirq: ``cirq.decompose`` to X/Y/Z/H power gates (including Rx, Ry, Rz),
  CX/CZ powers and global phases, each possibly controlled. An EigenGate
  P**t with global shift s is exp(i*pi*t*s) * ((1+w)/2 I + (1-w)/2 P),
  w = exp(i*pi*t), for the involutions P in {X, Y, Z, H}.
* PennyLane: ``qml.transforms.decompose`` to one-qubit rotations, Paulis,
  Hadamard, PhaseShift, CNOT, CZ and GlobalPhase (documented as
  exp(-i*phi)), each possibly controlled or adjoint.

A gate outside these sets raises rather than being guessed.
"""

from typing import Any

import numpy as np

from aeonq.oracles.numpy_sim import Op, gate_matrix

_INVOLUTIONS = {
    "X": gate_matrix("X", ()),
    "Y": gate_matrix("Y", ()),
    "Z": gate_matrix("Z", ()),
    "H": gate_matrix("H", ()),
}


def _matrix_op(
    u: np.ndarray, target: int, controls: tuple[int, ...], values: tuple[int, ...]
) -> Op:
    return Op("M", target, tuple(complex(v) for v in u.reshape(-1)), controls, values)


# ------------------------------------------------------------------ Qiskit


def from_qiskit(circuit: Any) -> list[Op]:
    from qiskit import transpile

    basic = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    ops: list[Op] = []
    if basic.global_phase:
        ops.append(Op("GPHASE", None, float(basic.global_phase)))
    for item in basic.data:
        name = item.operation.name
        qubits = [basic.find_bit(q).index for q in item.qubits]
        if name == "u":
            theta, phi, lam = (float(p) for p in item.operation.params)
            ops.append(Op("U", qubits[0], (theta, phi, lam)))
        elif name == "cx":
            ops.append(Op("X", qubits[1], controls=(qubits[0],)))
        elif name in ("barrier", "id"):
            continue
        else:
            raise ValueError(f"unexpected gate after transpiling: {name}")
    return ops


# -------------------------------------------------------------------- Cirq


def _cirq_kind(gate: Any) -> str | None:
    import cirq

    for kind, cls in (
        ("X", cirq.XPowGate),
        ("Y", cirq.YPowGate),
        ("Z", cirq.ZPowGate),
        ("H", cirq.HPowGate),
        ("CX", cirq.CXPowGate),
        ("CZ", cirq.CZPowGate),
    ):
        if isinstance(gate, cls):
            return kind
    if isinstance(gate, (cirq.GlobalPhaseGate, cirq.IdentityGate)):
        return type(gate).__name__
    return None


def _cirq_supported(operation: Any) -> bool:
    import cirq

    if isinstance(operation, cirq.ControlledOperation):
        return _cirq_supported(operation.sub_operation)
    gate = operation.gate
    kind = _cirq_kind(gate)
    if kind is None:
        return False
    return kind in ("GlobalPhaseGate", "IdentityGate") or isinstance(
        gate.exponent, (int, float)
    )


def _power(kind: str, exponent: float, shift: float) -> tuple[np.ndarray, float]:
    """(matrix of P**t without global shift, shift phase in radians)."""
    w = np.exp(1j * np.pi * exponent)
    p = _INVOLUTIONS[kind]
    return (1 + w) / 2 * np.eye(2) + (1 - w) / 2 * p, float(np.pi * exponent * shift)


def from_cirq(circuit: Any, qubits: list[Any]) -> list[Op]:
    """``qubits[i]`` is logical qubit i."""
    import cirq

    index = {qubit: i for i, qubit in enumerate(qubits)}
    flat = cirq.decompose(circuit, keep=_cirq_supported, on_stuck_raise=None)
    ops: list[Op] = []
    for operation in flat:
        controls: tuple[int, ...] = ()
        values: tuple[int, ...] = ()
        while isinstance(operation, cirq.ControlledOperation):
            for qubit, allowed in zip(operation.controls, operation.control_values):
                (value,) = tuple(allowed)
                controls += (index[qubit],)
                values += (int(value),)
            operation = operation.sub_operation
        if not _cirq_supported(operation):
            raise ValueError(f"cirq operation outside the gate set: {operation!r}")
        gate: Any = operation.gate
        kind = _cirq_kind(gate)
        targets = [index[q] for q in operation.qubits]
        if kind == "IdentityGate":
            continue
        if kind == "GlobalPhaseGate":
            phase = float(np.angle(complex(gate.coefficient)))
            ops.append(Op("GPHASE", None, phase, controls, values))
            continue
        assert kind is not None
        exponent, shift = float(gate.exponent), float(gate.global_shift)
        if kind in ("CX", "CZ"):
            u, phase = _power(kind[1], exponent, shift)
            controls += (targets[0],)
            values += (1,)
            # The shift multiplies the whole two-qubit gate, so it stays
            # outside the new control and inside any outer controls.
            if phase:
                ops.append(Op("GPHASE", None, phase, controls[:-1], values[:-1]))
            ops.append(_matrix_op(u, targets[1], controls, values))
        else:
            u, phase = _power(kind, exponent, shift)
            if phase:
                ops.append(Op("GPHASE", None, phase, controls, values))
            ops.append(_matrix_op(u, targets[0], controls, values))
    return ops


# --------------------------------------------------------------- PennyLane

_PENNYLANE_GATES = {
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "CNOT",
    "CZ",
    "GlobalPhase",
    "Identity",
}


def _pennylane_supported(op: Any) -> bool:
    import pennylane as qml

    if isinstance(op, (qml.ops.op_math.Controlled, qml.ops.op_math.Adjoint)):
        return _pennylane_supported(op.base)
    return op.name in _PENNYLANE_GATES


def _pennylane_matrix(name: str, parameters: list[float]) -> np.ndarray:
    names = {"Hadamard": "H", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}
    if name in names:
        return gate_matrix(names[name], ())
    if name == "PhaseShift":
        return gate_matrix("P", (parameters[0],))
    return gate_matrix(name, (parameters[0],))


def from_pennylane(tape: Any) -> list[Op]:
    """Wires must be the integers 0..n-1, wire i being logical qubit i."""
    import pennylane as qml

    [decomposed], _ = qml.transforms.decompose(
        tape, stopping_condition=_pennylane_supported
    )
    ops: list[Op] = []
    for op in decomposed.operations:
        controls: tuple[int, ...] = ()
        values: tuple[int, ...] = ()
        dagger = False
        while isinstance(op, (qml.ops.op_math.Controlled, qml.ops.op_math.Adjoint)):
            if isinstance(op, qml.ops.op_math.Adjoint):
                dagger = not dagger
            else:
                controls += tuple(int(w) for w in op.control_wires)
                values += tuple(int(bool(v)) for v in op.control_values)
            op = op.base
        if op.name not in _PENNYLANE_GATES:
            raise ValueError(f"PennyLane operation outside the gate set: {op.name}")
        if op.name == "Identity":
            continue
        parameters = [float(p) for p in op.parameters]
        wires = [int(w) for w in op.wires]
        if op.name == "GlobalPhase":
            ops.append(
                Op("GPHASE", None, -parameters[0], controls, values, None, dagger)
            )
        elif op.name in ("CNOT", "CZ"):
            base = "X" if op.name == "CNOT" else "Z"
            ops.append(
                Op(
                    base,
                    wires[1],
                    None,
                    controls + (wires[0],),
                    values + (1,),
                    None,
                    dagger,
                )
            )
        else:
            u = _pennylane_matrix(op.name, parameters)
            if dagger:
                u = u.conj().T
            ops.append(_matrix_op(u, wires[0], controls, values))
    return ops
