"""Stable Bernoulli success derivatives and Fisher information."""

from dataclasses import dataclass
from math import asin, isclose, sin, sqrt


def success_probability(amplitude: float, power: int) -> float:
    """Ideal p_k(a) = sin²((2k+1) asin sqrt(a))."""
    if not 0 <= amplitude <= 1 or not isinstance(power, int) or power < 0:
        raise ValueError("amplitude must be in [0,1] and power nonnegative")
    m = 2 * power + 1
    if amplitude > 0.5:
        delta = asin(sqrt(1 - amplitude))
        return min(1.0, max(0.0, 1 - sin(m * delta) ** 2))
    theta = asin(sqrt(amplitude))
    return min(1.0, max(0.0, sin(m * theta) ** 2))


def success_derivative(amplitude: float, power: int) -> float:
    """dp_k/da with exact endpoint limits m² for odd m=2k+1."""
    if not 0 <= amplitude <= 1 or not isinstance(power, int) or power < 0:
        raise ValueError("amplitude must be in [0,1] and power nonnegative")
    m = 2 * power + 1
    if amplitude in (0.0, 1.0):
        return float(m * m)
    angle = asin(sqrt(amplitude if amplitude <= 0.5 else 1 - amplitude))
    return m * sin(2 * m * angle) / (2 * sqrt(amplitude * (1 - amplitude)))


@dataclass(frozen=True)
class FisherValue:
    value: float
    regular: bool
    interpretation: str


def fisher_information(amplitude: float, power: int) -> FisherValue:
    """Limiting Fisher information m²/[a(1-a)] under the ideal model.

    At p=0 or 1 the ordinary Bernoulli score is not regular; the value is
    explicitly a limit. At a=0 or 1 the limit diverges and is not used by
    schedulers as an infinite priority.
    """
    if not 0 <= amplitude <= 1 or not isinstance(power, int) or power < 0:
        raise ValueError("amplitude must be in [0,1] and power nonnegative")
    m = 2 * power + 1
    if amplitude in (0.0, 1.0):
        return FisherValue(float("inf"), False, "divergent endpoint limit")
    p = success_probability(amplitude, power)
    regular = not isclose(p, 0.0, abs_tol=1e-14) and not isclose(p, 1.0, abs_tol=1e-14)
    return FisherValue(
        m * m / (amplitude * (1 - amplitude)),
        regular,
        "ordinary Fisher information"
        if regular
        else "limiting value at deterministic probability",
    )
