"""Direct PennyLane 0.45.1 lowering to tapes and default.qubit devices."""

from importlib.metadata import version
from typing import Any

import numpy as np
import pennylane as qml

from aeonq.adapters.base import BackendAdapter, BackendCapabilities, NormalizedResult
from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    ObjectiveReflection,
    Operator,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.registers import RegisterMap
from aeonq.synthesis.reflection import (
    synthesize_objective_reflection,
    synthesize_zero_reflection,
)


class PennyLaneAdapter(BackendAdapter):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            None,
            None,
            native_gate_set=frozenset(
                {"Hadamard", "PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ", "CNOT"}
            ),
            metadata={"pennylane": version("pennylane")},
        )

    def validate(self, ir: Operator) -> None:
        if ir.n_qubits > 12:
            raise ValueError(
                "local PennyLane adapter limits semantic lowering to 12 qubits"
            )
        self.lower(ir)

    def lower(self, ir: Operator) -> qml.tape.QuantumScript:
        with qml.queuing.AnnotatedQueue() as queue:
            # Preserve logical width even for k=0 or untouched wires.
            for wire in range(ir.n_qubits):
                qml.Identity(wires=wire)
            self._apply(ir, tuple(range(ir.n_qubits)))
        return qml.tape.QuantumScript.from_queue(queue)

    def _apply(self, ir: Operator, wires: tuple[int, ...]) -> None:
        if isinstance(ir, Gate):
            target = tuple(wires[i] for i in ir.targets)
            mapping = {
                "H": qml.Hadamard,
                "X": qml.PauliX,
                "Y": qml.PauliY,
                "Z": qml.PauliZ,
                "CX": qml.CNOT,
            }
            if ir.name in ("RX", "RY", "RZ"):
                if not isinstance(ir.parameter, (int, float)):
                    raise ValueError("bind symbolic parameters before lowering")
                getattr(qml, ir.name)(ir.parameter, wires=target[0])
            else:
                mapping[ir.name](wires=list(target) if len(target) == 2 else target[0])
        elif isinstance(ir, Sequence):
            for child in ir.children:
                self._apply(child, wires)
        elif isinstance(ir, StatePreparation):
            self._apply(ir.operation, wires)
        elif isinstance(ir, Adjoint):
            qml.adjoint(lambda: self._apply(ir.operation, wires))()
        elif isinstance(ir, GlobalPhase):
            # PennyLane's convention is exp(-i phi).
            qml.GlobalPhase(-ir.radians)
        elif isinstance(ir, ZeroReflection):
            self._apply(synthesize_zero_reflection(ir), wires)
        elif isinstance(ir, ObjectiveReflection):
            self._apply(synthesize_objective_reflection(ir), wires)
        elif isinstance(ir, GroverOperator):
            from aeonq.synthesis.grover import synthesize_grover

            self._apply(synthesize_grover(ir), wires)
        elif isinstance(ir, GroverPower):
            for _ in range(ir.power):
                self._apply(ir.grover, wires)
        elif isinstance(ir, ControlledOperator):
            controls = tuple(wires[c] for c in ir.controls)
            targets = tuple(wires[t] for t in ir.targets)
            qml.ctrl(
                lambda: self._apply(ir.operation, targets),
                control=controls,
                control_values=ir.control_values,
            )()
        elif isinstance(ir, EmbeddedOperator):
            self._apply(ir.operation, tuple(wires[t] for t in ir.targets))
        else:
            raise TypeError(f"PennyLane lowering does not support {type(ir).__name__}")

    def compile(
        self,
        native_program: qml.tape.QuantumScript,
        options: dict[str, Any] | None = None,
    ) -> qml.tape.QuantumScript:
        if options:
            raise ValueError("PennyLane compilation options are not implemented")
        return native_program

    def execute(
        self,
        compiled_program: qml.tape.QuantumScript,
        shots: int,
        *,
        seed: int | None = None,
    ) -> Any:
        if shots < 1:
            raise ValueError("shots must be positive")
        n = max(int(w) for w in compiled_program.wires) + 1
        device = qml.device("default.qubit", wires=list(reversed(range(n))), seed=seed)
        tape = qml.tape.QuantumScript(
            compiled_program.operations, [qml.sample(wires=list(range(n)))], shots=shots
        )
        samples = np.asarray(qml.execute([tape], device)[0])
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        raw: dict[str, int] = {}
        for row in samples:
            bitstring = "".join(str(int(bit)) for bit in reversed(row))
            raw[bitstring] = raw.get(bitstring, 0) + 1
        return raw

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("PennyLane counts must be a mapping")
        counts: dict[tuple[int, ...], int] = {}
        for bitstring, count in raw.items():
            if (
                not isinstance(bitstring, str)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("malformed count entry")
            bits = register_map.normalize_bitstring(bitstring)
            counts[bits] = counts.get(bits, 0) + count
        return NormalizedResult(counts, sum(counts.values()))

    def resource_metrics(
        self, compiled_program: qml.tape.QuantumScript
    ) -> dict[str, int | float | None]:
        ops = compiled_program.operations
        return {
            "qubits": len(compiled_program.wires),
            "depth": len(ops),
            "two_qubit_gates": sum(len(op.wires) == 2 for op in ops),
            "one_qubit_gates": sum(len(op.wires) == 1 for op in ops),
            "multi_qubit_gates": sum(len(op.wires) > 2 for op in ops),
            "t_count": None,
        }

    def unitary(self, ir: Operator) -> np.ndarray:
        return np.asarray(
            qml.matrix(self.lower(ir), wire_order=list(reversed(range(ir.n_qubits))))
        )

    def statevector(self, ir: Operator) -> np.ndarray:
        tape = self.lower(ir)
        device = qml.device("default.qubit", wires=list(reversed(range(ir.n_qubits))))
        state_tape = qml.tape.QuantumScript(tape.operations, [qml.state()])
        return np.asarray(qml.execute([state_tape], device)[0])
