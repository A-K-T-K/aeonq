"""First-class amplitude-estimation operators, independent of SDK circuits.

Sequence children execute from left to right. A node's ``n_qubits`` is the
size of its enclosing logical state space, including controls where present.
"""

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from aeonq.ir.predicates import ObjectivePredicate


@dataclass(frozen=True)
class Parameter:
    name: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not isfinite(self.scale):
            raise ValueError("invalid symbolic parameter")


@dataclass(frozen=True)
class Gate:
    name: str
    targets: tuple[int, ...]
    n_qubits: int
    parameter: float | Parameter | None = None

    def __post_init__(self) -> None:
        arities = {"H": 1, "X": 1, "Y": 1, "Z": 1, "RX": 1, "RY": 1, "RZ": 1, "CX": 2}
        if self.name not in arities or len(self.targets) != arities[self.name]:
            raise ValueError("unsupported gate or wrong arity")
        _check_indices(self.targets, self.n_qubits)
        if self.name in ("RX", "RY", "RZ"):
            if not isinstance(self.parameter, (int, float, Parameter)) or (
                isinstance(self.parameter, (int, float))
                and not isfinite(self.parameter)
            ):
                raise ValueError(
                    "rotation needs a finite numeric or symbolic parameter"
                )
        elif self.parameter is not None:
            raise ValueError("this gate has no parameter")


@dataclass(frozen=True)
class Sequence:
    children: tuple["Operator", ...]
    n_qubits: int

    def __post_init__(self) -> None:
        _check_width(self.n_qubits)
        if any(child.n_qubits != self.n_qubits for child in self.children):
            raise ValueError("sequence children must have the same logical width")
        if any(isinstance(child, Measurement) for child in self.children):
            raise ValueError("measurement is not a unitary sequence child")


@dataclass(frozen=True)
class StatePreparation:
    operation: "Operator"
    n_qubits: int

    def __post_init__(self) -> None:
        if self.operation.n_qubits != self.n_qubits:
            raise ValueError("state preparation width mismatch")


@dataclass(frozen=True)
class Adjoint:
    operation: "Operator"
    n_qubits: int

    def __post_init__(self) -> None:
        if self.operation.n_qubits != self.n_qubits or isinstance(
            self.operation, Measurement
        ):
            raise ValueError("adjoint requires a unitary with matching width")


@dataclass(frozen=True)
class GlobalPhase:
    radians: float
    n_qubits: int

    def __post_init__(self) -> None:
        _check_width(self.n_qubits)
        if not isfinite(self.radians):
            raise ValueError("global phase must be finite")


@dataclass(frozen=True)
class ZeroReflection:
    n_qubits: int
    state_qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_indices(self.state_qubits, self.n_qubits)
        if not self.state_qubits:
            raise ValueError("zero reflection needs at least one state qubit")


@dataclass(frozen=True)
class ObjectiveReflection:
    n_qubits: int
    predicate: ObjectivePredicate

    def __post_init__(self) -> None:
        _check_width(self.n_qubits)
        self.predicate.validate(self.n_qubits)


@dataclass(frozen=True)
class GroverOperator:
    preparation: StatePreparation
    zero_reflection: ZeroReflection
    objective_reflection: ObjectiveReflection
    n_qubits: int

    def __post_init__(self) -> None:
        if any(
            node.n_qubits != self.n_qubits
            for node in (
                self.preparation,
                self.zero_reflection,
                self.objective_reflection,
            )
        ):
            raise ValueError("Grover components have mismatched widths")
        if set(self.zero_reflection.state_qubits) != set(range(self.n_qubits)):
            raise ValueError("Grover requires all qubits to be state qubits")


@dataclass(frozen=True)
class GroverPower:
    grover: GroverOperator
    power: int
    n_qubits: int

    def __post_init__(self) -> None:
        if (
            self.n_qubits != self.grover.n_qubits
            or not isinstance(self.power, int)
            or self.power < 0
        ):
            raise ValueError(
                "Grover power needs matching width and nonnegative integer"
            )


@dataclass(frozen=True)
class ControlledOperator:
    operation: "Operator"
    controls: tuple[int, ...]
    control_values: tuple[int, ...]
    targets: tuple[int, ...]
    n_qubits: int

    def __post_init__(self) -> None:
        _check_indices(self.controls, self.n_qubits)
        if (
            not self.controls
            or len(self.control_values) != len(self.controls)
            or any(v not in (0, 1) for v in self.control_values)
        ):
            raise ValueError("controls and control values must match")
        _check_indices(self.targets, self.n_qubits)
        if self.operation.n_qubits != len(self.targets):
            raise ValueError("controlled operation width must match targets")
        if isinstance(self.operation, Measurement):
            raise ValueError("measurement cannot be controlled")
        if set(self.controls) & set(self.targets):
            raise ValueError("control-target collision")


@dataclass(frozen=True)
class EmbeddedOperator:
    """Place a local operator on specified logical qubits of a larger space."""

    operation: "Operator"
    targets: tuple[int, ...]
    n_qubits: int

    def __post_init__(self) -> None:
        _check_indices(self.targets, self.n_qubits)
        if self.operation.n_qubits != len(self.targets):
            raise ValueError("embedded operation width must match target count")


@dataclass(frozen=True)
class Measurement:
    qubits: tuple[int, ...]
    n_qubits: int

    def __post_init__(self) -> None:
        _check_indices(self.qubits, self.n_qubits)
        if not self.qubits:
            raise ValueError("measurement requires qubits")


Operator: TypeAlias = (
    Gate
    | Sequence
    | StatePreparation
    | Adjoint
    | GlobalPhase
    | ZeroReflection
    | ObjectiveReflection
    | GroverOperator
    | GroverPower
    | ControlledOperator
    | EmbeddedOperator
)


def _check_width(n_qubits: int) -> None:
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("logical width must be a positive integer")


def _check_indices(indices: tuple[int, ...], n_qubits: int) -> None:
    _check_width(n_qubits)
    if len(set(indices)) != len(indices) or any(
        not isinstance(i, int) or i < 0 or i >= n_qubits for i in indices
    ):
        raise ValueError("qubit indices must be unique and in range")


def support(node: Operator) -> set[int]:
    """Conservative set of qubits on which an operator may act."""
    if isinstance(node, Gate):
        return set(node.targets)
    if isinstance(node, Sequence):
        return set().union(*(support(child) for child in node.children))
    if isinstance(node, (StatePreparation, Adjoint)):
        return support(node.operation)
    if isinstance(node, ZeroReflection):
        return set(node.state_qubits)
    if isinstance(node, ObjectiveReflection):
        return set(range(node.n_qubits))
    if isinstance(node, (GroverOperator, GroverPower)):
        return set(range(node.n_qubits))
    if isinstance(node, ControlledOperator):
        return set(node.controls) | set(node.targets)
    if isinstance(node, EmbeddedOperator):
        return set(node.targets)
    if isinstance(node, GlobalPhase):
        return set(range(node.n_qubits))
    raise TypeError(f"unsupported operator: {type(node).__name__}")
