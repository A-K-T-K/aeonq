"""Select expected information per configurable execution cost."""

from collections.abc import Mapping
from dataclasses import dataclass

from aeonq.scheduling.base import ScheduleDecision, ScheduleState, decision
from aeonq.scheduling.information import expected_information


@dataclass(frozen=True)
class ExecutionCost:
    oracle_weight: float = 1.0
    shot_weight: float = 0.0
    depth_weight: float = 0.0
    two_qubit_weight: float = 0.0
    wall_time_weight: float = 0.0
    submission_overhead: float = 0.0
    seconds_per_depth_shot: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.oracle_weight,
            self.shot_weight,
            self.depth_weight,
            self.two_qubit_weight,
            self.wall_time_weight,
            self.submission_overhead,
        )
        if any(v < 0 for v in (*weights, self.seconds_per_depth_shot)) or all(
            v == 0 for v in weights
        ):
            raise ValueError("cost weights must be nonnegative and nonzero")
        if self.wall_time_weight and self.seconds_per_depth_shot <= 0:
            raise ValueError("wall-time cost requires positive seconds_per_depth_shot")

    @property
    def requires_metrics(self) -> bool:
        """Whether estimates need compiled depth or two-qubit counts."""
        return bool(self.depth_weight or self.two_qubit_weight or self.wall_time_weight)

    def estimate(
        self,
        power: int,
        shots: int,
        metrics: Mapping[str, int | float | None] | None = None,
    ) -> float:
        if power < 0 or shots < 1:
            raise ValueError("invalid power or shots")
        metrics = metrics or {}
        depth = metrics.get("depth")
        two_qubit = metrics.get("two_qubit_gates")
        if (self.depth_weight or self.wall_time_weight) and depth is None:
            raise ValueError("depth metric required by cost model")
        if self.two_qubit_weight and two_qubit is None:
            raise ValueError("two-qubit metric required by cost model")
        depth = float(depth or 0)
        two_qubit = float(two_qubit or 0)
        if depth < 0 or two_qubit < 0:
            raise ValueError("resource metrics must be nonnegative")
        cost = (
            self.oracle_weight * shots * (2 * power + 1)
            + self.shot_weight * shots
            + self.depth_weight * shots * depth
            + self.two_qubit_weight * shots * two_qubit
            + self.wall_time_weight * shots * depth * self.seconds_per_depth_shot
            + self.submission_overhead
        )
        if cost <= 0:
            raise ValueError("candidate has zero cost")
        return cost


@dataclass(frozen=True)
class CostAwareSchedule:
    cost: ExecutionCost = ExecutionCost()
    grid_points: int = 129

    def select(self, state: ScheduleState) -> ScheduleDecision:
        candidates = state.candidates()
        if not candidates:
            raise ValueError("no feasible cost-aware candidate")
        scored = []
        for power, shots, branch in candidates:
            info = expected_information(
                state.theta_interval, power, shots, grid_points=self.grid_points
            )
            # Metric lookups may compile the candidate power, so skip them when
            # the cost model ignores circuit resources.
            metrics = (
                state.resource_metrics_by_power.get(power)
                if self.cost.requires_metrics
                else None
            )
            cost = self.cost.estimate(power, shots, metrics)
            scored.append((info / cost, info, cost, power, shots, branch))
        score, info, cost, power, shots, branch = max(scored)
        return decision(
            power,
            shots,
            branch,
            state,
            score=score,
            information=info,
            estimated_cost=cost,
        )
