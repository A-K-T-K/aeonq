"""Fixed schedule baseline with explicit alias policy."""

from dataclasses import dataclass

from aeonq.inference.alias import unambiguous_branch
from aeonq.scheduling.base import ScheduleDecision, ScheduleState, decision


@dataclass(frozen=True)
class FixedSchedule:
    entries: tuple[tuple[int, int], ...]
    enforce_alias_safety: bool = True

    def select(self, state: ScheduleState) -> ScheduleDecision:
        if state.iteration >= len(self.entries):
            raise ValueError("fixed schedule exhausted")
        power, shots = self.entries[state.iteration]
        if power < 0 or power > state.max_power or shots < 1:
            raise ValueError("fixed schedule entry exceeds configured range")
        if state.remaining_shots is not None and shots > state.remaining_shots:
            raise ValueError("fixed schedule exceeds remaining shots")
        if (
            state.remaining_queries is not None
            and shots * (2 * power + 1) > state.remaining_queries
        ):
            raise ValueError("fixed schedule exceeds remaining oracle queries")
        branch = unambiguous_branch(state.theta_interval, power)
        if self.enforce_alias_safety and branch is None:
            raise ValueError("fixed power crosses phase branches")
        return decision(power, shots, branch, state)
