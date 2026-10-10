"""Declarative amplitude-estimation problem."""

from dataclasses import dataclass, field
from typing import Any

from aeonq.ir.nodes import StatePreparation
from aeonq.ir.predicates import ObjectivePredicate


@dataclass(frozen=True)
class AEProblem:
    state_preparation: StatePreparation
    objective: ObjectivePredicate
    state_register: tuple[int, ...]
    work_register: tuple[int, ...] = ()
    objective_register: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.state_preparation.n_qubits
        groups = self.state_register + self.work_register + self.objective_register
        if (
            not self.state_register
            or len(groups) != len(set(groups))
            or set(groups) != set(range(n))
        ):
            raise ValueError("registers must partition the logical qubits")
        self.objective.validate(n)
