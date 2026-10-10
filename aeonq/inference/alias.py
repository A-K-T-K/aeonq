"""Monotonic-branch safety for amplified Bernoulli observations."""

from math import floor, pi


def unambiguous_branch(theta_interval: tuple[float, float], power: int) -> int | None:
    """Return branch only when the whole feasible interval is monotone.

    A boundary exactly at an extremum is allowed; a crossing is not.
    """
    lower, upper = theta_interval
    if not 0 <= lower <= upper <= pi / 2 or power < 0:
        raise ValueError("invalid feasible phase interval or power")
    scale = 2 * power + 1
    left = floor((scale * lower) / (pi / 2))
    right = floor((scale * upper) / (pi / 2))
    if right == left:
        return left
    if right == left + 1 and abs(scale * upper - right * pi / 2) <= 1e-14:
        return left
    return None


def candidate_branches(
    theta_interval: tuple[float, float], power: int
) -> tuple[int, ...]:
    lower, upper = theta_interval
    if not 0 <= lower <= upper <= pi / 2 or power < 0:
        raise ValueError("invalid feasible phase interval or power")
    scale = 2 * power + 1
    start = floor(scale * lower / (pi / 2))
    stop = floor(scale * upper / (pi / 2))
    return tuple(range(start, stop + 1))
