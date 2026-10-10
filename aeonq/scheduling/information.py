"""Expected discrete-prior information gain for candidate power and shots."""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

from aeonq.scheduling.base import ScheduleDecision, ScheduleState, decision


def expected_information(
    theta_interval: tuple[float, float],
    power: int,
    shots: int,
    *,
    grid_points: int = 129,
) -> float:
    """Mutual information I(theta_grid; binomial count) for uniform phase prior.

    Used as a numerical decision score on a discrete phase grid.
    Deterministic probabilities use xlogy.
    """
    if shots < 1 or power < 0 or grid_points < 2:
        raise ValueError("invalid expected-information parameters")
    lo, hi = theta_interval
    if not 0 <= lo <= hi <= np.pi / 2:
        raise ValueError("invalid phase interval")
    if lo == hi:
        return 0.0
    theta = np.linspace(lo, hi, grid_points)
    p = np.sin((2 * power + 1) * theta) ** 2
    x = np.arange(shots + 1)[:, None]
    log_coeff = gammaln(shots + 1) - gammaln(x + 1) - gammaln(shots - x + 1)
    log_prob = log_coeff + xlogy(x, p[None, :]) + xlogy(shots - x, 1 - p[None, :])
    log_prior = -np.log(grid_points)
    log_predictive = logsumexp(log_prob + log_prior, axis=1)
    log_posterior = log_prob + log_prior - log_predictive[:, None]
    posterior = np.exp(log_posterior)
    # Mask 0 * -inf without changing possible outcomes.
    entropy = -np.sum(xlogy(posterior, posterior), axis=1)
    information = np.log(grid_points) - float(np.sum(np.exp(log_predictive) * entropy))
    return max(0.0, information)


@dataclass(frozen=True)
class ExpectedInformationSchedule:
    grid_points: int = 129

    def select(self, state: ScheduleState) -> ScheduleDecision:
        candidates = state.candidates()
        if not candidates:
            raise ValueError("no feasible expected-information candidate")
        scored = []
        for power, shots, branch in candidates:
            info = expected_information(
                state.theta_interval, power, shots, grid_points=self.grid_points
            )
            cost = float(shots)
            scored.append((info / cost, info, power, shots, branch))
        score, info, power, shots, branch = max(scored)
        return decision(
            power,
            shots,
            branch,
            state,
            score=score,
            information=info,
            estimated_cost=float(shots),
        )
