"""Endpoint-safe exact binomial intervals and phase inversion."""

from math import asin, pi, sqrt

from scipy.stats import beta


def clopper_pearson(successes: int, shots: int, alpha: float) -> tuple[float, float]:
    """Two-sided Clopper–Pearson interval with exact binomial coverage."""
    if (
        not isinstance(successes, int)
        or not isinstance(shots, int)
        or shots < 1
        or not 0 <= successes <= shots
    ):
        raise ValueError("invalid binomial counts")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2, successes, shots - successes + 1))
    )
    upper = (
        1.0
        if successes == shots
        else float(beta.ppf(1 - alpha / 2, successes + 1, shots - successes))
    )
    return lower, upper


def phase_interval(
    probability_interval: tuple[float, float], power: int, branch: int
) -> tuple[float, float]:
    """Invert sin²((2k+1)θ) on a known monotone branch jπ/2..(j+1)π/2."""
    p_lower, p_upper = probability_interval
    if not 0 <= p_lower <= p_upper <= 1 or power < 0 or branch < 0:
        raise ValueError("invalid phase inversion input")
    lo = asin(sqrt(p_lower))
    hi = asin(sqrt(p_upper))
    scale = 2 * power + 1
    if branch % 2 == 0:
        return ((branch * pi / 2 + lo) / scale, (branch * pi / 2 + hi) / scale)
    return (((branch + 1) * pi / 2 - hi) / scale, ((branch + 1) * pi / 2 - lo) / scale)
