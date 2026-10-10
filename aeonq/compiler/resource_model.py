"""Configurable cost model over measured compiled-circuit resources."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompilerCost:
    depth_weight: float = 1.0
    two_qubit_weight: float = 1.0
    ancilla_weight: float = 1.0
    t_weight: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.depth_weight,
            self.two_qubit_weight,
            self.ancilla_weight,
            self.t_weight,
        )
        if any(w < 0 for w in weights) or all(w == 0 for w in weights):
            raise ValueError("compiler cost weights must be nonnegative and nonzero")

    def evaluate(
        self, metrics: dict[str, int | float | None], semantic_qubits: int
    ) -> float:
        depth = metrics.get("depth")
        two_qubit = metrics.get("two_qubit_gates")
        t_count = metrics.get("t_count")
        qubits = metrics.get("qubits")
        if self.depth_weight and depth is None:
            raise ValueError("depth metric required")
        if self.two_qubit_weight and two_qubit is None:
            raise ValueError("two-qubit metric required")
        if self.t_weight and t_count is None:
            raise ValueError("T-count metric required")
        if self.ancilla_weight and qubits is None:
            raise ValueError("qubit count required")
        ancillas = max(0, int(qubits or 0) - semantic_qubits)
        if metrics.get("multi_qubit_gates", 0):
            raise ValueError("resource comparison requires a decomposed gate basis")
        return (
            self.depth_weight * float(depth or 0)
            + self.two_qubit_weight * float(two_qubit or 0)
            + self.ancilla_weight * ancillas
            + self.t_weight * float(t_count or 0)
        )
