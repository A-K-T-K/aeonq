"""Validated scheduler state and decisions."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import pi
from typing import Protocol

from aeonq.inference.alias import candidate_branches, unambiguous_branch


@dataclass(frozen=True)
class ScheduleState:
    theta_interval: tuple[float, float]
    iteration: int
    max_power: int
    shot_options: tuple[int, ...]
    remaining_shots: int | None = None
    remaining_queries: int | None = None
    resource_metrics_by_power: Mapping[int, Mapping[str, int | float | None]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        lo, hi = self.theta_interval
        if not 0 <= lo <= hi <= pi / 2 or self.iteration < 0 or self.max_power < 0:
            raise ValueError("invalid schedule phase/power state")
        if not self.shot_options or any(
            not isinstance(n, int) or n < 1 for n in self.shot_options
        ):
            raise ValueError("shot options must be positive integers")
        if self.remaining_shots is not None and self.remaining_shots < 0:
            raise ValueError("negative remaining shot budget")
        if self.remaining_queries is not None and self.remaining_queries < 0:
            raise ValueError("negative remaining query budget")

    def candidates(self) -> tuple[tuple[int, int, int], ...]:
        result = []
        for power in range(self.max_power + 1):
            branch = unambiguous_branch(self.theta_interval, power)
            if branch is None:
                continue
            for shots in self.shot_options:
                if self.remaining_shots is not None and shots > self.remaining_shots:
                    continue
                if (
                    self.remaining_queries is not None
                    and shots * (2 * power + 1) > self.remaining_queries
                ):
                    continue
                result.append((power, shots, branch))
        return tuple(result)


@dataclass(frozen=True)
class ScheduleDecision:
    power: int
    shots: int
    branch: int | None
    alias_risk: int
    score: float | None = None
    information: float | None = None
    estimated_cost: float | None = None


class Schedule(Protocol):
    def select(self, state: ScheduleState) -> ScheduleDecision: ...


def decision(
    power: int,
    shots: int,
    branch: int | None,
    state: ScheduleState,
    *,
    score: float | None = None,
    information: float | None = None,
    estimated_cost: float | None = None,
) -> ScheduleDecision:
    return ScheduleDecision(
        power,
        shots,
        branch,
        len(candidate_branches(state.theta_interval, power)) - 1,
        score,
        information,
        estimated_cost,
    )
