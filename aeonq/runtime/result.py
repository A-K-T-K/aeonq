"""Stable estimator result with explicit interval and guarantee semantics."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AEResult:
    estimate: float
    confidence_or_credible_interval: tuple[float, float]
    interval_type: str
    requested_epsilon: float
    requested_alpha: float
    convergence_status: str
    guarantee_type: str
    guarantee_status: str
    assumptions: tuple[str, ...]
    total_shots: int
    total_oracle_queries: int
    total_circuit_evaluations: int
    powers_used: tuple[int, ...]
    shots_per_power: dict[int, int]
    max_power: int
    iterations: int
    compile_time: float
    execution_time: float
    wall_clock_time: float
    backend: str
    sdk: str
    sdk_version: str
    resource_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Readable estimate, interval, guarantee and the resources the run used."""
        from aeonq.resources import result_summary

        return result_summary(self)
