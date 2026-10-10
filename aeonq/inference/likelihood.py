"""Stable ideal-Bernoulli likelihood calculations."""

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from scipy.special import xlogy

from aeonq.utils.numerics import DEFAULT_POLICY, NumericalPolicy


def binomial_log_likelihood(
    successes: int,
    shots: int,
    probability: float,
    policy: NumericalPolicy = DEFAULT_POLICY,
) -> float:
    """Return log likelihood without combinatorial constant.

    Impossible outcomes have -inf log likelihood. Only floating-point excursions
    within the numerical policy are corrected; interior probabilities are not clipped.
    """
    if (
        not isinstance(successes, int)
        or not isinstance(shots, int)
        or shots < 0
        or not 0 <= successes <= shots
    ):
        raise ValueError("invalid binomial counts")
    if (
        not isfinite(probability)
        or probability < -policy.probability_clip
        or probability > 1 + policy.probability_clip
    ):
        raise ValueError("probability outside [0,1]")
    p = min(1.0, max(0.0, probability))
    return float(xlogy(successes, p) + xlogy(shots - successes, 1 - p))


def amplified_log_likelihood_grid(
    theta: NDArray[np.float64],
    powers: NDArray[np.int64],
    successes: NDArray[np.int64],
    shots_per_power: int,
) -> NDArray[np.float64]:
    """Evaluate all phase-grid points and observations in one array operation."""
    if (
        theta.ndim != 1
        or powers.ndim != 1
        or successes.ndim != 1
        or len(powers) != len(successes)
        or shots_per_power < 1
        or np.any(powers < 0)
        or np.any(successes < 0)
        or np.any(successes > shots_per_power)
        or np.any(~np.isfinite(theta))
        or np.any(theta < 0)
        or np.any(theta > np.pi / 2)
    ):
        raise ValueError("invalid amplified likelihood grid")
    probabilities = np.sin(theta[:, None] * (2 * powers[None, :] + 1)) ** 2
    return np.sum(
        xlogy(successes[None, :], probabilities)
        + xlogy(shots_per_power - successes[None, :], 1 - probabilities),
        axis=1,
    )
