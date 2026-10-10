"""Select maximum ideal Fisher information per state-preparation query."""

from dataclasses import dataclass
from math import sin

from aeonq.inference.fisher import fisher_information
from aeonq.scheduling.base import ScheduleDecision, ScheduleState, decision


@dataclass(frozen=True)
class FisherInformationSchedule:
    def select(self, state: ScheduleState) -> ScheduleDecision:
        if state.theta_interval[0] == state.theta_interval[1]:
            raise ValueError("phase is already determined")
        candidates = state.candidates()
        if not candidates:
            raise ValueError("no feasible Fisher schedule candidate")
        theta_mid = sum(state.theta_interval) / 2
        amplitude = sin(theta_mid) ** 2

        # Endpoints are nonregular. Use interval midpoint as planning proxy;
        # exact boundary information is never used as infinite priority.
        def score(item: tuple[int, int, int]) -> float:
            power, shots, _ = item
            information = fisher_information(amplitude, power).value
            return shots * information / (shots * (2 * power + 1))

        power, shots, branch = max(candidates, key=lambda item: (score(item), item[1]))
        information = shots * fisher_information(amplitude, power).value
        cost = shots * (2 * power + 1)
        return decision(
            power,
            shots,
            branch,
            state,
            score=information / cost,
            information=information,
            estimated_cost=float(cost),
        )
