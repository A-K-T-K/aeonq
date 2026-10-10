"""Direct Cirq 1.7 lowering with explicit logical qubit and bit ordering."""

from importlib.metadata import version
from typing import Any

import cirq
import numpy as np

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


class CirqAdapter(BackendAdapter):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            None,
            None,
            native_gate_set=frozenset({"H", "X", "Y", "Z", "rx", "ry", "rz", "CNOT"}),
            metadata={"cirq_core": version("cirq-core")},
        )

    def validate(self, ir: Operator) -> None:
        if ir.n_qubits > 12:
            raise ValueError("local Cirq adapter limits semantic lowering to 12 qubits")
        self.lower(ir)

    def lower(self, ir: Operator) -> cirq.Circuit:
        # Cirq infers width from touched qubits. Explicit identities preserve
        # logical locations that a program leaves idle.
        circuit = cirq.Circuit(cirq.I(q) for q in cirq.LineQubit.range(ir.n_qubits))
        self._append(circuit, ir)
        return circuit

    def _body(self, ir: Operator) -> cirq.Circuit:
        """Lower without the identity layer; repeated blocks must not add moments."""
        circuit = cirq.Circuit()
        self._append(circuit, ir)
        return circuit

    def _append(self, circuit: cirq.Circuit, ir: Operator) -> None:
        q = cirq.LineQubit.range(ir.n_qubits)
        if isinstance(ir, Gate):
            if ir.name in ("RX", "RY", "RZ"):
                if not isinstance(ir.parameter, (int, float)):
                    raise ValueError("bind symbolic parameters before lowering")
                operation = getattr(cirq, ir.name.lower())(ir.parameter).on(
                    q[ir.targets[0]]
                )
            elif ir.name == "CX":
                operation = cirq.CNOT(q[ir.targets[0]], q[ir.targets[1]])
            else:
                operation = getattr(cirq, ir.name).on(q[ir.targets[0]])
            circuit.append(operation)
        elif isinstance(ir, Sequence):
            for child in ir.children:
                self._append(circuit, child)
        elif isinstance(ir, StatePreparation):
            self._append(circuit, ir.operation)
        elif isinstance(ir, Adjoint):
            circuit.append(cirq.inverse(self._body(ir.operation)))
        elif isinstance(ir, GlobalPhase):
            circuit.append(cirq.global_phase_operation(np.exp(1j * ir.radians)))
        elif isinstance(ir, ZeroReflection):
            self._append(circuit, synthesize_zero_reflection(ir))
        elif isinstance(ir, ObjectiveReflection):
            self._append(circuit, synthesize_objective_reflection(ir))
        elif isinstance(ir, GroverOperator):
            from aeonq.synthesis.grover import synthesize_grover

            self._append(circuit, synthesize_grover(ir))
        elif isinstance(ir, GroverPower):
            block = self._body(ir.grover)
            for _ in range(ir.power):
                circuit.append(block)
        elif isinstance(ir, ControlledOperator):
            block = self.lower(ir.operation)
            op = cirq.CircuitOperation(cirq.FrozenCircuit(block))
            op = op.with_qubit_mapping(
                {cirq.LineQubit(i): q[target] for i, target in enumerate(ir.targets)}
            )
            circuit.append(
                op.controlled_by(
                    *(q[c] for c in ir.controls), control_values=ir.control_values
                )
            )
        elif isinstance(ir, EmbeddedOperator):
            block = self.lower(ir.operation)
            op = cirq.CircuitOperation(cirq.FrozenCircuit(block))
            circuit.append(
                op.with_qubit_mapping(
                    {
                        cirq.LineQubit(i): q[target]
                        for i, target in enumerate(ir.targets)
                    }
                )
            )
        else:
            raise TypeError(f"Cirq lowering does not support {type(ir).__name__}")

    def compile(
        self, native_program: cirq.Circuit, options: dict[str, Any] | None = None
    ) -> cirq.Circuit:
        if options:
            raise ValueError("Cirq compilation options are not implemented")
        return native_program.copy()

    def execute(
        self, compiled_program: cirq.Circuit, shots: int, *, seed: int | None = None
    ) -> Any:
        if shots < 1:
            raise ValueError("shots must be positive")
        circuit = compiled_program.copy()
        qubits = sorted(circuit.all_qubits())
        if not qubits:
            raise ValueError("cannot infer qubit count from empty Cirq circuit")
        circuit.append(cirq.measure(*qubits, key="logical"))
        # Cirq's integer seed path is limited to 32 bits. Feed all entropy
        # from the condition-specific benchmark seed to its RandomState.
        random = (
            None
            if seed is None
            else np.random.RandomState(np.random.SeedSequence(seed).generate_state(4))
        )
        result = cirq.Simulator(seed=random).run(circuit, repetitions=shots)
        raw: dict[str, int] = {}
        for row in result.measurements["logical"]:
            bitstring = "".join(str(int(bit)) for bit in reversed(row))
            raw[bitstring] = raw.get(bitstring, 0) + 1
        return raw

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("Cirq counts must be a mapping")
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
        self, compiled_program: cirq.Circuit
    ) -> dict[str, int | float | None]:
        ops = list(compiled_program.all_operations())
        return {
            "qubits": len(compiled_program.all_qubits()),
            "depth": len(compiled_program),
            "two_qubit_gates": sum(len(op.qubits) == 2 for op in ops),
            "one_qubit_gates": sum(len(op.qubits) == 1 for op in ops),
            "multi_qubit_gates": sum(len(op.qubits) > 2 for op in ops),
            "t_count": None,
        }

    def unitary(self, ir: Operator) -> np.ndarray:
        return np.asarray(
            self.lower(ir).unitary(
                qubit_order=list(reversed(cirq.LineQubit.range(ir.n_qubits)))
            )
        )

    def statevector(self, ir: Operator) -> np.ndarray:
        order = list(reversed(cirq.LineQubit.range(ir.n_qubits)))
        return np.asarray(
            cirq.Simulator(dtype=np.complex128)
            .simulate(self.lower(ir), qubit_order=order)
            .final_state_vector
        )
