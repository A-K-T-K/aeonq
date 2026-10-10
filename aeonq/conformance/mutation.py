"""Seeded adapter faults for measuring what the conformance battery detects.

Each mutant models a lowering or readout bug of a kind that arises when
porting between SDK conventions. Mutants act on the semantic IR just before
the wrapped adapter lowers it (or on its raw results), so the same fault can be
injected into every SDK adapter without editing adapter source.
"""

from math import pi
from typing import Any

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
from aeonq.synthesis import (
    synthesize_grover,
    synthesize_objective_reflection,
    synthesize_zero_reflection,
)

MUTANTS: dict[str, str] = {
    "drop_global_phase": "global phases lowered as no-ops",
    "global_phase_sign": "global phase sign convention flipped (exp(-i phi))",
    "drop_phase_under_control": "global phase of a controlled block discarded",
    "rotation_sign": "rotation angles negated",
    "rotation_half_angle": "rotation angle convention off by a factor of two",
    "rz_as_phase_gate": "RZ lowered as the phase gate diag(1, e^{i theta})",
    "swap_cx_direction": "CX control and target exchanged",
    "invert_first_control": "first control value inverted",
    "adjoint_ignored": "Adjoint(U) lowered as U",
    "grover_power_off_by_one": "Q^k lowered as Q^(k+1)",
    "zero_reflection_about_ones": "S0 reflects about |1...1> instead of |0...0>",
    "objective_reflection_negated": "S_chi lowered as -S_chi",
    "reverse_qubit_order": "logical qubit q placed on native qubit n-1-q",
    "reverse_result_bits": "result bitstrings read in the opposite order",
    "drop_one_shot": "one shot lost from the returned counts",
}


def mutate(node: Operator, mutant: str, *, under_control: bool = False) -> Operator:
    """Rewrite ``node`` with one fault; semantic nodes are expanded first."""

    def again(child: Operator, controlled: bool = under_control) -> Operator:
        return mutate(child, mutant, under_control=controlled)

    n = node.n_qubits
    if isinstance(node, Gate):
        return _mutate_gate(node, mutant)
    if isinstance(node, GlobalPhase):
        if mutant == "drop_global_phase" or (
            mutant == "drop_phase_under_control" and under_control
        ):
            return GlobalPhase(0.0, n)
        if mutant == "global_phase_sign":
            return GlobalPhase(-node.radians, n)
        return node
    if isinstance(node, Sequence):
        return Sequence(tuple(again(child) for child in node.children), n)
    if isinstance(node, StatePreparation):
        return StatePreparation(again(node.operation), n)
    if isinstance(node, Adjoint):
        if mutant == "adjoint_ignored":
            return again(node.operation)
        return Adjoint(again(node.operation), n)
    if isinstance(node, ControlledOperator):
        values = node.control_values
        if mutant == "invert_first_control":
            values = (1 - values[0],) + values[1:]
        return ControlledOperator(
            again(node.operation, True), node.controls, values, node.targets, n
        )
    if isinstance(node, EmbeddedOperator):
        return EmbeddedOperator(again(node.operation), node.targets, n)
    if isinstance(node, ZeroReflection):
        expanded = again(synthesize_zero_reflection(node))
        if mutant == "zero_reflection_about_ones":
            flips = tuple(Gate("X", (q,), n) for q in node.state_qubits)
            return Sequence(flips + (expanded,) + flips, n)
        return expanded
    if isinstance(node, ObjectiveReflection):
        expanded = again(synthesize_objective_reflection(node))
        if mutant == "objective_reflection_negated":
            return Sequence((expanded, GlobalPhase(pi, n)), n)
        return expanded
    if isinstance(node, GroverOperator):
        return again(synthesize_grover(node))
    if isinstance(node, GroverPower):
        extra = int(mutant == "grover_power_off_by_one" and node.power > 0)
        return Sequence((again(node.grover),) * (node.power + extra), n)
    raise TypeError(f"cannot mutate {type(node).__name__}")


def _mutate_gate(gate: Gate, mutant: str) -> Operator:
    n = gate.n_qubits
    if gate.name in ("RX", "RY", "RZ") and isinstance(gate.parameter, (int, float)):
        theta = float(gate.parameter)
        if mutant == "rotation_sign":
            return Gate(gate.name, gate.targets, n, -theta)
        if mutant == "rotation_half_angle":
            return Gate(gate.name, gate.targets, n, theta / 2)
        if mutant == "rz_as_phase_gate" and gate.name == "RZ":
            # diag(1, e^{i theta}) = e^{i theta/2} RZ(theta): equal up to phase.
            return Sequence((gate, GlobalPhase(theta / 2, n)), n)
    if mutant == "swap_cx_direction" and gate.name == "CX":
        return Gate("CX", tuple(reversed(gate.targets)), n)
    return gate


def _reverse_qubits(node: Operator, width: int) -> Operator:
    """Conjugate by the qubit-reversal permutation on the top-level space."""

    def flip(q: int) -> int:
        return width - 1 - q

    if isinstance(node, Gate):
        return Gate(node.name, tuple(map(flip, node.targets)), width, node.parameter)
    if isinstance(node, Sequence):
        return Sequence(tuple(_reverse_qubits(c, width) for c in node.children), width)
    if isinstance(node, StatePreparation):
        return StatePreparation(_reverse_qubits(node.operation, width), width)
    if isinstance(node, Adjoint):
        return Adjoint(_reverse_qubits(node.operation, width), width)
    if isinstance(node, ControlledOperator):
        # The controlled operation acts in its own local space.
        return ControlledOperator(
            node.operation,
            tuple(map(flip, node.controls)),
            node.control_values,
            tuple(map(flip, node.targets)),
            width,
        )
    if isinstance(node, EmbeddedOperator):
        return EmbeddedOperator(node.operation, tuple(map(flip, node.targets)), width)
    if isinstance(node, GlobalPhase):
        return node
    raise TypeError(f"cannot reorder {type(node).__name__}")


class MutantAdapter(BackendAdapter):
    """Delegate to ``inner`` with one seeded fault in lowering or readout."""

    def __init__(self, inner: BackendAdapter, mutant: str) -> None:
        if mutant not in MUTANTS:
            raise ValueError(f"unknown mutant: {mutant}")
        self.inner = inner
        self.mutant = mutant
        self.unitary_precision = inner.unitary_precision

    def _ir(self, ir: Operator) -> Operator:
        if self.mutant == "reverse_qubit_order":
            return _reverse_qubits(mutate(ir, "none"), ir.n_qubits)
        if self.mutant in ("reverse_result_bits", "drop_one_shot"):
            return ir
        return mutate(ir, self.mutant)

    def capabilities(self) -> BackendCapabilities:
        return self.inner.capabilities()

    def validate(self, ir: Operator) -> None:
        self.inner.validate(self._ir(ir))

    def lower(self, ir: Operator) -> Any:
        return self.inner.lower(self._ir(ir))

    def compile(
        self, native_program: Any, options: dict[str, Any] | None = None
    ) -> Any:
        return self.inner.compile(native_program, options)

    def execute(
        self, compiled_program: Any, shots: int, *, seed: int | None = None
    ) -> Any:
        raw = self.inner.execute(compiled_program, shots, seed=seed)
        if self.mutant == "drop_one_shot" and raw:
            raw = dict(raw)
            key = max(raw, key=lambda outcome: raw[outcome])
            raw[key] -= 1
            if not raw[key]:
                del raw[key]
        return raw

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if self.mutant == "reverse_result_bits":
            raw = {key[::-1]: count for key, count in raw.items()}
        return self.inner.normalize_result(raw, register_map)

    def resource_metrics(self, compiled_program: Any) -> dict[str, int | float | None]:
        return self.inner.resource_metrics(compiled_program)

    def unitary(self, ir: Operator) -> np.ndarray:
        return self.inner.unitary(self._ir(ir))
