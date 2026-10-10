"""Backend contract used by compilers and algorithms, without SDK imports."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aeonq.ir.nodes import Operator
from aeonq.ir.registers import RegisterMap


@dataclass(frozen=True)
class BackendCapabilities:
    supports_statevector: bool
    supports_finite_shots: bool
    supports_controlled_operations: bool
    supports_arbitrary_unitary: bool
    supports_mid_circuit_measurement: bool
    supports_dynamic_circuits: bool
    supports_parameter_binding: bool
    supports_async_jobs: bool
    max_qubits: int | None
    max_shots: int | None
    connectivity: tuple[tuple[int, int], ...] | None = None
    native_gate_set: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)
    # False when shots are not ideal independent draws from the exact circuit
    # distribution (noise models, hardware); estimators then withhold guarantees.
    ideal_sampling: bool = True


@dataclass(frozen=True)
class NormalizedResult:
    counts: dict[tuple[int, ...], int]
    shots: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    # Largest per-entry error expected from ``unitary`` on an exact adapter;
    # adapters whose matrix export is rounded override this with a measured bound.
    unitary_precision: float = 1e-9

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def validate(self, ir: Operator) -> None: ...

    @abstractmethod
    def lower(self, ir: Operator) -> Any: ...

    @abstractmethod
    def compile(
        self, native_program: Any, options: dict[str, Any] | None = None
    ) -> Any: ...

    @abstractmethod
    def execute(
        self, compiled_program: Any, shots: int, *, seed: int | None = None
    ) -> Any: ...

    async def execute_async(
        self, compiled_program: Any, shots: int, *, seed: int | None = None
    ) -> Any:
        raise NotImplementedError(
            "asynchronous execution is not supported by this backend"
        )

    @abstractmethod
    def normalize_result(
        self, raw: Any, register_map: RegisterMap
    ) -> NormalizedResult: ...

    @abstractmethod
    def resource_metrics(
        self, compiled_program: Any
    ) -> dict[str, int | float | None]: ...

    def unitary(self, ir: Operator) -> np.ndarray:
        """Optional exact small-system matrix capability."""
        raise NotImplementedError("unitary matrix is not supported by this backend")
