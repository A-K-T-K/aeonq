"""Problems restated from first principles, with closed-form amplitudes.

Each case gives the analytic success probability a, the good-state predicate
on little-endian basis indices (bit q of x is qubit q), and the gate list of
the state preparation A in the plain format of ``numpy_sim``. Angles come
from the problem's probabilities by a = sin^2(angle / 2); nothing here is
derived from aeonq's problem builders. Case names match the conformance
battery so that rows line up.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import asin, exp, log, sqrt

import numpy as np

from aeonq.oracles.numpy_sim import Op, adjoint


@dataclass(frozen=True)
class OracleCase:
    name: str
    n_qubits: int
    a: float
    good: Callable[[int], bool]
    preparation: tuple[Op, ...]
    # Named angles, reused verbatim by the hand-written SDK circuits.
    angles: dict[str, float]

    @property
    def theta(self) -> float:
        """sin^2(theta) = a; Q rotates by 2*theta on span{good, bad}."""
        return asin(sqrt(self.a))

    def good_mask(self) -> np.ndarray:
        return np.array([self.good(x) for x in range(2**self.n_qubits)])


def _ry_angle(probability: float) -> float:
    """RY(angle)|0> has probability ``probability`` of reading 1."""
    return 2 * asin(sqrt(probability))


def bernoulli_quarter() -> OracleCase:
    angle = _ry_angle(0.25)
    return OracleCase(
        "bernoulli_0.25",
        1,
        0.25,
        lambda x: x & 1 == 1,
        (Op("RY", 0, angle),),
        {"ry": angle},
    )


def uniform_threshold() -> OracleCase:
    # Two uniform bits, x >= 1: three of the four outcomes are good.
    return OracleCase(
        "uniform_threshold_n2_k1",
        2,
        3 / 4,
        lambda x: x >= 1,
        (Op("H", 0), Op("H", 1)),
        {},
    )


def two_bit_distribution() -> OracleCase:
    """p = (0.1, 0.2, 0.3, 0.4) over x = q0 + 2*q1, good when x is 1 or 2.

    Prepare q0 with P(q0 = 1) = p1 + p3, then q1 conditioned on each value
    of q0: P(q1 = 1 | q0 = 0) = p2 / (p0 + p2), P(q1 = 1 | q0 = 1) = p3 / (p1 + p3).
    """
    p = (0.1, 0.2, 0.3, 0.4)
    low = _ry_angle(p[1] + p[3])
    high_if_0 = _ry_angle(p[2] / (p[0] + p[2]))
    high_if_1 = _ry_angle(p[3] / (p[1] + p[3]))
    return OracleCase(
        "distribution_0.1_0.2_0.3_0.4_marked_1_2",
        2,
        p[1] + p[2],
        lambda x: x in (1, 2),
        (
            Op("RY", 0, low),
            Op("RY", 1, high_if_0, controls=(0,), control_values=(0,)),
            Op("RY", 1, high_if_1, controls=(0,), control_values=(1,)),
        ),
        {"low": low, "high_if_0": high_if_0, "high_if_1": high_if_1},
    )


def lognormal_probabilities(
    spot: float = 2.0,
    volatility: float = 0.4,
    rate: float = 0.05,
    maturity: float = 40 / 365,
) -> list[float]:
    """Risk-neutral log-normal price density at four equally spaced prices
    spanning mean +/- 3 standard deviations, normalized to sum to one."""
    mu = log(spot) + (rate - volatility**2 / 2) * maturity
    sigma = volatility * sqrt(maturity)
    mean = exp(mu + sigma**2 / 2)
    sd = sqrt((exp(sigma**2) - 1) * exp(2 * mu + sigma**2))
    lowest = max(1e-12, mean - 3 * sd)
    step = (mean + 3 * sd - lowest) / 3
    prices = [lowest + i * step for i in range(4)]
    density = [exp(-((log(s) - mu) ** 2) / (2 * sigma**2)) / s for s in prices]
    total = sum(density)
    return [d / total for d in density]


def lognormal_digital() -> OracleCase:
    """P(price index >= 2) on a four-point log-normal grid, with the flag
    [x >= 2] = q1 copied into qubit 2. Good when the flag is set.

    The high bit q1 is loaded first, P(q1 = 1) = p2 + p3, then q0 conditioned
    on q1: P(q0 = 1 | q1 = h) = p(2h + 1) / (p(2h) + p(2h + 1)).
    """
    p = lognormal_probabilities()
    high = _ry_angle(p[2] + p[3])
    low_if_0 = _ry_angle(p[1] / (p[0] + p[1]))
    low_if_1 = _ry_angle(p[3] / (p[2] + p[3]))
    return OracleCase(
        "lognormal_tail_n2_k2_flag",
        3,
        p[2] + p[3],
        lambda x: (x >> 2) & 1 == 1,
        (
            Op("RY", 1, high),
            Op("RY", 0, low_if_0, controls=(1,), control_values=(0,)),
            Op("RY", 0, low_if_1, controls=(1,), control_values=(1,)),
            Op("X", 2, controls=(1,)),
        ),
        {"high": high, "low_if_0": low_if_0, "low_if_1": low_if_1},
    )


CASES: tuple[Callable[[], OracleCase], ...] = (
    bernoulli_quarter,
    uniform_threshold,
    two_bit_distribution,
    lognormal_digital,
)


def grover_ops(case: OracleCase) -> tuple[Op, ...]:
    """Q = -A S0 A^dagger S_chi as an operation list, S_chi applied first."""
    return (
        Op("FLIP", None, predicate=case.good),
        *adjoint(case.preparation),
        Op("FLIP", None, predicate=lambda x: x == 0),
        *case.preparation,
        Op("GPHASE", None, np.pi),
    )
