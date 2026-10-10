"""Safe declarative objective predicates on logical basis indices."""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObjectivePredicate:
    kind: Literal["marked", "equal", "ge", "bit", "and", "or", "not"]
    qubits: tuple[int, ...] = ()
    value: int | None = None
    marked: tuple[int, ...] = ()
    children: tuple["ObjectivePredicate", ...] = ()

    def validate(self, n_qubits: int) -> None:
        if (
            n_qubits < 1
            or len(set(self.qubits)) != len(self.qubits)
            or any(q < 0 or q >= n_qubits for q in self.qubits)
        ):
            raise ValueError("predicate qubits must be unique valid logical indices")
        if self.kind in ("equal", "ge"):
            if (
                not self.qubits
                or self.value is None
                or not 0 <= self.value < 2 ** len(self.qubits)
            ):
                raise ValueError(
                    "comparison predicate needs an in-range value and qubits"
                )
        elif self.kind == "bit":
            if len(self.qubits) != 1:
                raise ValueError("bit predicate requires one qubit")
        elif self.kind == "marked":
            if self.qubits or any(x < 0 or x >= 2**n_qubits for x in self.marked):
                raise ValueError("marked states must be valid full-register indices")
        elif self.kind in ("and", "or", "not"):
            if len(self.children) != (1 if self.kind == "not" else 2):
                raise ValueError("Boolean predicate has wrong number of children")
            for child in self.children:
                child.validate(n_qubits)
        else:
            raise ValueError(f"unsupported predicate kind: {self.kind}")

    def evaluate(self, basis: int, n_qubits: int) -> bool:
        self.validate(n_qubits)
        if not 0 <= basis < 2**n_qubits:
            raise ValueError("basis state out of range")
        return self._evaluate(basis)

    def evaluate_many(
        self, basis: NDArray[np.int64], n_qubits: int
    ) -> NDArray[np.bool_]:
        """Vectorized ``evaluate`` that validates the predicate once."""
        self.validate(n_qubits)
        basis = np.asarray(basis, dtype=np.int64)
        if basis.size and (basis.min() < 0 or basis.max() >= 2**n_qubits):
            raise ValueError("basis state out of range")
        return self._evaluate_many(basis)

    def truth_table(self, n_qubits: int) -> NDArray[np.bool_]:
        """χ(x) for every logical basis index x in 0..2^n-1."""
        return self.evaluate_many(np.arange(2**n_qubits, dtype=np.int64), n_qubits)

    @cached_property
    def _marked_set(self) -> frozenset[int]:
        return frozenset(self.marked)

    def _evaluate(self, basis: int) -> bool:
        if self.kind == "marked":
            return basis in self._marked_set
        if self.kind in ("equal", "ge"):
            assert self.value is not None
            # Qubits are in little-endian significance order within the integer.
            x = sum(((basis >> q) & 1) << i for i, q in enumerate(self.qubits))
            return x == self.value if self.kind == "equal" else x >= self.value
        if self.kind == "bit":
            return bool((basis >> self.qubits[0]) & 1)
        vals = tuple(c._evaluate(basis) for c in self.children)
        return (
            all(vals)
            if self.kind == "and"
            else any(vals)
            if self.kind == "or"
            else not vals[0]
        )

    def _evaluate_many(self, basis: NDArray[np.int64]) -> NDArray[np.bool_]:
        if self.kind == "marked":
            return np.isin(basis, np.asarray(self.marked, dtype=np.int64))
        if self.kind in ("equal", "ge"):
            assert self.value is not None
            x = np.zeros_like(basis)
            for i, q in enumerate(self.qubits):
                x |= ((basis >> q) & 1) << i
            return x == self.value if self.kind == "equal" else x >= self.value
        if self.kind == "bit":
            return ((basis >> self.qubits[0]) & 1).astype(bool)
        vals = [c._evaluate_many(basis) for c in self.children]
        if self.kind == "and":
            return vals[0] & vals[1]
        if self.kind == "or":
            return vals[0] | vals[1]
        return ~vals[0]
