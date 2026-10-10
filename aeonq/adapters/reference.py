"""Exact small-system sampling backend used as the reference baseline."""

from typing import Any
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from aeonq.adapters.base import BackendAdapter, BackendCapabilities, NormalizedResult
from aeonq.ir.nodes import Operator
from aeonq.ir.registers import RegisterMap
from aeonq.reference import matrix, statevector


@lru_cache(maxsize=512)
def _probability_vector(ir: Operator) -> NDArray[np.float64]:
    """Cache immutable ideal probabilities shared by repeated seeded trials."""
    amplitudes = statevector(ir)
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= probabilities.sum()
    probabilities.setflags(write=False)
    return probabilities


class ReferenceAdapter(BackendAdapter):
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            10,
            None,
            metadata={"reference": "aeonq-1"},
        )

    def validate(self, ir: Operator) -> None:
        if ir.n_qubits > 10:
            raise ValueError("reference backend supports at most 10 qubits")

    def lower(self, ir: Operator) -> Operator:
        self.validate(ir)
        return ir

    def compile(
        self, native_program: Operator, options: dict[str, Any] | None = None
    ) -> Operator:
        if options:
            raise ValueError("reference backend has no compile options")
        return native_program

    def execute(
        self, compiled_program: Operator, shots: int, *, seed: int | None = None
    ) -> Any:
        if shots < 1:
            raise ValueError("shots must be positive")
        counts = np.random.default_rng(seed).multinomial(
            shots, _probability_vector(compiled_program)
        )
        n = compiled_program.n_qubits
        return {f"{i:0{n}b}": int(count) for i, count in enumerate(counts) if count}

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("reference counts must be a mapping")
        counts: dict[tuple[int, ...], int] = {}
        for key, count in raw.items():
            if not isinstance(key, str) or not isinstance(count, int) or count < 0:
                raise ValueError("malformed reference count")
            bits = register_map.normalize_bitstring(key)
            counts[bits] = counts.get(bits, 0) + count
        return NormalizedResult(counts, sum(counts.values()))

    def unitary(self, ir: Operator) -> NDArray[np.complex128]:
        self.validate(ir)
        return matrix(ir)

    def resource_metrics(
        self, compiled_program: Operator
    ) -> dict[str, int | float | None]:
        return {
            "qubits": compiled_program.n_qubits,
            "depth": None,
            "one_qubit_gates": None,
            "two_qubit_gates": None,
            "t_count": None,
        }
