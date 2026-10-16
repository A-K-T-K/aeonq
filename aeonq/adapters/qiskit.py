"""Direct lowering to Qiskit 2.5 circuits and local Aer execution."""

from importlib.metadata import version
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Operator as QiskitOperator, Statevector
from qiskit_aer import AerSimulator

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
from aeonq.compiler.resource_model import CompilerCost
from aeonq.adapters.qiskit_multicontrol import select_multi_control


class QiskitAdapter(BackendAdapter):
    def __init__(
        self,
        *,
        clean_ancillas: int = 0,
        dirty_ancillas: int = 0,
        multi_control_cost: CompilerCost = CompilerCost(),
        noise_model: Any = None,
    ) -> None:
        if clean_ancillas < 0 or dirty_ancillas < 0:
            raise ValueError("ancilla capacities must be nonnegative")
        # An Aer NoiseModel makes shots non-ideal; capabilities report it so
        # estimators withhold guarantees that assume ideal sampling.
        self.noise_model = noise_model
        self.backend = (
            AerSimulator()
            if noise_model is None
            else AerSimulator(noise_model=noise_model)
        )
        self.clean_ancillas = clean_ancillas
        self.dirty_ancillas = dirty_ancillas
        self.multi_control_cost = multi_control_cost

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
            native_gate_set=frozenset(
                {"h", "x", "y", "z", "rx", "ry", "rz", "cx", "mcphase"}
            ),
            metadata={
                "qiskit": version("qiskit"),
                "qiskit_aer": version("qiskit-aer"),
                **(
                    {"noise_model": str(self.noise_model)}
                    if self.noise_model is not None
                    else {}
                ),
            },
            ideal_sampling=self.noise_model is None,
        )

    def validate(self, ir: Operator) -> None:
        if ir.n_qubits > 12:
            raise ValueError(
                "local Qiskit adapter limits semantic lowering to 12 qubits"
            )
        self.lower(ir)

    def lower(self, ir: Operator) -> QuantumCircuit:
        circuit = QuantumCircuit(
            ir.n_qubits + self.clean_ancillas + self.dirty_ancillas
        )
        self._append(circuit, ir)
        return circuit

    def _lower_local(self, ir: Operator) -> QuantumCircuit:
        circuit = QuantumCircuit(ir.n_qubits)
        self._append(circuit, ir)
        return circuit

    def _append(self, circuit: QuantumCircuit, ir: Operator) -> None:
        if isinstance(ir, Gate):
            args = list(ir.targets)
            gate_name = ir.name.lower()
            if ir.name in ("RX", "RY", "RZ"):
                if not isinstance(ir.parameter, (int, float)):
                    raise ValueError("bind symbolic parameters before lowering")
                getattr(circuit, gate_name)(ir.parameter, *args)
            else:
                getattr(circuit, gate_name)(*args)
        elif isinstance(ir, Sequence):
            for child in ir.children:
                self._append(circuit, child)
        elif isinstance(ir, StatePreparation):
            self._append(circuit, ir.operation)
        elif isinstance(ir, Adjoint):
            inverse = self._lower_local(ir.operation).inverse()
            circuit.compose(inverse, qubits=list(range(ir.n_qubits)), inplace=True)
        elif isinstance(ir, GlobalPhase):
            circuit.global_phase += ir.radians
        elif isinstance(ir, ZeroReflection):
            self._append(circuit, synthesize_zero_reflection(ir))
        elif isinstance(ir, ObjectiveReflection):
            self._append(circuit, synthesize_objective_reflection(ir))
        elif isinstance(ir, GroverOperator):
            from aeonq.synthesis.grover import synthesize_grover

            self._append(circuit, synthesize_grover(ir))
        elif isinstance(ir, GroverPower):
            if ir.power:
                # Same width as the target, so ancilla-aware multi-control
                # selection inside Q is unchanged; lower Q once, repeat it.
                block = QuantumCircuit(circuit.num_qubits)
                self._append(block, ir.grover)
                for _ in range(ir.power):
                    circuit.compose(block, inplace=True)
        elif isinstance(ir, ControlledOperator):
            native_multicontrol = (
                len(ir.controls) >= 3
                and len(ir.targets) == 1
                and isinstance(ir.operation, Gate)
                and ir.operation.name in {"X", "Z"}
                and all(value == 1 for value in ir.control_values)
            )
            if native_multicontrol:
                assert isinstance(ir.operation, Gate)
                extra = circuit.num_qubits - ir.n_qubits
                clean = min(self.clean_ancillas, extra)
                dirty = min(self.dirty_ancillas, max(0, extra - clean))
                selected = select_multi_control(
                    len(ir.controls),
                    phase=ir.operation.name == "Z",
                    clean_available=clean,
                    dirty_available=dirty,
                    cost_model=self.multi_control_cost,
                ).selected
                ancilla_start = (
                    ir.n_qubits
                    if selected.strategy != "dirty_ladder"
                    else ir.n_qubits + clean
                )
                physical = [
                    *ir.controls,
                    *ir.targets,
                    *range(ancilla_start, ancilla_start + selected.ancillas),
                ]
                circuit.compose(selected.circuit, qubits=physical, inplace=True)
            else:
                target = self._lower_local(ir.operation)
                ctrl_state = sum(
                    value << i for i, value in enumerate(ir.control_values)
                )
                gate = target.to_gate().control(len(ir.controls), ctrl_state=ctrl_state)
                circuit.append(gate, [*ir.controls, *ir.targets])
        elif isinstance(ir, EmbeddedOperator):
            circuit.compose(
                self._lower_local(ir.operation), qubits=list(ir.targets), inplace=True
            )
        else:
            raise TypeError(f"Qiskit lowering does not support {type(ir).__name__}")

    def compile(
        self, native_program: QuantumCircuit, options: dict[str, Any] | None = None
    ) -> QuantumCircuit:
        options = options or {}
        kwargs = {
            "optimization_level": int(options.get("optimization_level", 1)),
            "seed_transpiler": int(options.get("seed_transpiler", 0)),
        }
        if "basis_gates" in options:
            kwargs["basis_gates"] = options["basis_gates"]
        if "basis_gates" in kwargs:
            # A gate-set-only comparison does not need backend timing metadata.
            return transpile(native_program, **kwargs)
        return transpile(native_program, backend=self.backend, **kwargs)

    def execute(
        self, compiled_program: QuantumCircuit, shots: int, *, seed: int | None = None
    ) -> Any:
        if shots < 1:
            raise ValueError("shots must be positive")
        circuit = compiled_program.copy()
        if not circuit.num_clbits:
            # Optimization levels 2 and 3 can elide swap networks into a final
            # qubit permutation, so read each logical qubit from the wire that
            # holds it at the end rather than from its starting position.
            layout = circuit.layout
            final = (
                layout.final_index_layout()
                if layout is not None
                else list(range(circuit.num_qubits))
            )
            register = ClassicalRegister(len(final), "meas")
            circuit.add_register(register)
            circuit.barrier()
            for logical, physical in enumerate(final):
                circuit.measure(physical, register[logical])
        return (
            self.backend.run(circuit, shots=shots, seed_simulator=seed)
            .result()
            .get_counts()
        )

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("Qiskit counts must be a mapping")
        counts: dict[tuple[int, ...], int] = {}
        for bitstring, count in raw.items():
            if (
                not isinstance(bitstring, str)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("malformed count entry")
            bits = register_map.normalize_bitstring(bitstring.replace(" ", ""))
            counts[bits] = counts.get(bits, 0) + count
        return NormalizedResult(counts, sum(counts.values()))

    def resource_metrics(
        self, compiled_program: QuantumCircuit
    ) -> dict[str, int | float | None]:
        counts = compiled_program.count_ops()
        ops = [
            (item.operation.name, len(item.qubits)) for item in compiled_program.data
        ]
        return {
            "qubits": compiled_program.num_qubits,
            "depth": compiled_program.depth(),
            "two_qubit_gates": sum(
                width == 2 and name not in {"measure", "barrier"} for name, width in ops
            ),
            "one_qubit_gates": sum(
                width == 1 and name not in {"measure", "barrier"} for name, width in ops
            ),
            "multi_qubit_gates": sum(
                width > 2 and name not in {"measure", "barrier"} for name, width in ops
            ),
            "t_count": counts.get("t", 0) + counts.get("tdg", 0)
            if all(name not in {"u", "unitary"} for name, _ in ops)
            else None,
        }

    def unitary(self, ir: Operator) -> np.ndarray:
        return np.asarray(QiskitOperator(self.lower(ir)).data)

    def statevector(self, ir: Operator) -> np.ndarray:
        return np.asarray(Statevector.from_instruction(self.lower(ir)).data)
