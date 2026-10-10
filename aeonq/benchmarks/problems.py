"""Benchmark problems with independent analytic reference probabilities."""

from dataclasses import dataclass
from math import asin, exp, isclose, log, sqrt
from collections.abc import Sequence as FloatSequence

import numpy as np

from aeonq.ir import (
    ControlledOperator,
    Gate,
    ObjectivePredicate,
    Sequence,
    StatePreparation,
)
from aeonq.ir.nodes import Operator
from aeonq.problem import AEProblem
from aeonq.reference import reference_amplitude
from aeonq.synthesis.reflection import threshold_cubes


@dataclass(frozen=True)
class BenchmarkProblem:
    case_id: str
    problem: AEProblem
    a_true: float
    reference_method: str


def bernoulli(amplitude: float) -> BenchmarkProblem:
    if not 0 <= amplitude <= 1:
        raise ValueError("amplitude outside [0,1]")
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(amplitude))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    if abs(reference_amplitude(problem) - amplitude) > 1e-12:
        raise AssertionError("analytic and independent statevector references disagree")
    return BenchmarkProblem(
        f"bernoulli_{amplitude:.12g}",
        problem,
        amplitude,
        "analytic Bernoulli probability",
    )


def independent_threshold(n_qubits: int, threshold: int) -> BenchmarkProblem:
    if not 1 <= n_qubits <= 8 or not 0 <= threshold < 2**n_qubits:
        raise ValueError("invalid threshold benchmark size or threshold")
    prep = StatePreparation(
        Sequence(tuple(Gate("H", (q,), n_qubits) for q in range(n_qubits)), n_qubits),
        n_qubits,
    )
    objective = ObjectivePredicate("ge", qubits=tuple(range(n_qubits)), value=threshold)
    problem = AEProblem(prep, objective, tuple(range(n_qubits)))
    exact = (2**n_qubits - threshold) / 2**n_qubits
    if abs(reference_amplitude(problem) - exact) > 1e-12:
        raise AssertionError(
            "classical enumeration and statevector references disagree"
        )
    return BenchmarkProblem(
        f"uniform_threshold_n{n_qubits}_k{threshold}",
        problem,
        exact,
        "independent uniform classical enumeration",
    )


def uniform_marked(n_qubits: int, marked: tuple[int, ...]) -> BenchmarkProblem:
    """Uniform state with a full-register marked subspace."""
    if not 1 <= n_qubits <= 8 or not marked or len(set(marked)) != len(marked):
        raise ValueError("invalid marked benchmark")
    if any(not 0 <= index < 2**n_qubits for index in marked):
        raise ValueError("marked index out of range")
    prep = StatePreparation(
        Sequence(tuple(Gate("H", (q,), n_qubits) for q in range(n_qubits)), n_qubits),
        n_qubits,
    )
    problem = AEProblem(
        prep, ObjectivePredicate("marked", marked=marked), tuple(range(n_qubits))
    )
    exact = len(marked) / 2**n_qubits
    if not isclose(reference_amplitude(problem), exact, abs_tol=1e-12):
        raise AssertionError("marked classical count disagrees with statevector")
    return BenchmarkProblem(
        f"uniform_marked_n{n_qubits}_{'_'.join(map(str, marked))}",
        problem,
        exact,
        "independent marked-state count",
    )


def discrete_distribution(
    probabilities: FloatSequence[float], marked: tuple[int, ...]
) -> BenchmarkProblem:
    """Prepare an arbitrary two-bit distribution using conditional rotations."""
    if (
        len(probabilities) != 4
        or any(p < 0 or p > 1 for p in probabilities)
        or not isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("invalid four-outcome distribution")
    if (
        not marked
        or len(set(marked)) != len(marked)
        or any(x not in range(4) for x in marked)
    ):
        raise ValueError("invalid marked outcomes")
    # Little-endian outcomes: q0 is the low bit, q1 the high bit.
    low_zero = probabilities[0] + probabilities[2]
    low_one = probabilities[1] + probabilities[3]
    angle0 = 2 * asin(sqrt(low_one))
    angle_high_given_low_zero = (
        2 * asin(sqrt(probabilities[2] / low_zero)) if low_zero else 0.0
    )
    angle_high_given_low_one = (
        2 * asin(sqrt(probabilities[3] / low_one)) if low_one else 0.0
    )
    operations = (
        Gate("RY", (0,), 2, angle0),
        ControlledOperator(
            Gate("RY", (0,), 1, angle_high_given_low_zero), (0,), (0,), (1,), 2
        ),
        ControlledOperator(
            Gate("RY", (0,), 1, angle_high_given_low_one), (0,), (1,), (1,), 2
        ),
    )
    problem = AEProblem(
        StatePreparation(Sequence(operations, 2), 2),
        ObjectivePredicate("marked", marked=marked),
        (0, 1),
    )
    exact = sum(probabilities[x] for x in marked)
    if not isclose(reference_amplitude(problem), exact, abs_tol=1e-12):
        raise AssertionError("distribution classical mass disagrees with statevector")
    return BenchmarkProblem(
        f"distribution_{'_'.join(map(str, probabilities))}_marked_"
        f"{'_'.join(map(str, marked))}",
        problem,
        exact,
        "independent classical distribution mass",
    )


def independent_failures(failure_probabilities: tuple[float, ...]) -> BenchmarkProblem:
    """Reliability problem: success means at least one component failed."""
    if not 2 <= len(failure_probabilities) <= 8 or any(
        not 0 <= p <= 1 for p in failure_probabilities
    ):
        raise ValueError("invalid independent failure probabilities")
    n = len(failure_probabilities)
    operations = tuple(
        Gate("RY", (q,), n, 2 * asin(sqrt(p)))
        for q, p in enumerate(failure_probabilities)
    )
    objective = ObjectivePredicate(
        "not", children=(ObjectivePredicate("equal", qubits=tuple(range(n)), value=0),)
    )
    problem = AEProblem(
        StatePreparation(Sequence(operations, n), n), objective, tuple(range(n))
    )
    exact = 1.0
    for probability in failure_probabilities:
        exact *= 1 - probability
    exact = 1 - exact
    if not isclose(reference_amplitude(problem), exact, abs_tol=1e-12):
        raise AssertionError("reliability formula disagrees with statevector")
    return BenchmarkProblem(
        f"independent_failures_{'_'.join(map(str, failure_probabilities))}",
        problem,
        exact,
        "independent classical product formula",
    )


@dataclass(frozen=True)
class LogNormalGrid:
    """Risk-neutral geometric Brownian motion price, discretized on 2^n points."""

    points: tuple[float, ...]
    probabilities: tuple[float, ...]


def lognormal_grid(
    n_qubits: int,
    *,
    spot: float = 2.0,
    volatility: float = 0.4,
    rate: float = 0.05,
    maturity_years: float = 40 / 365,
) -> LogNormalGrid:
    """Normalized log-normal density at 2^n equally spaced prices (mean ± 3 sd).

    The estimated quantity is defined on this discrete distribution, so the
    reference truth below is exact; discretization error relative to the
    continuous model is outside the estimation problem.
    """
    if not 1 <= n_qubits <= 9 or min(spot, volatility, maturity_years) <= 0:
        raise ValueError("invalid log-normal grid")
    mu = (rate - volatility**2 / 2) * maturity_years + log(spot)
    sigma = volatility * sqrt(maturity_years)
    mean = exp(mu + sigma**2 / 2)
    sd = sqrt((exp(sigma**2) - 1) * exp(2 * mu + sigma**2))
    points = np.linspace(max(1e-12, mean - 3 * sd), mean + 3 * sd, 2**n_qubits)
    density = np.exp(-((np.log(points) - mu) ** 2) / (2 * sigma**2)) / points
    return LogNormalGrid(
        tuple(float(x) for x in points),
        tuple(float(p) for p in density / density.sum()),
    )


def _grover_rudolph(
    probabilities: tuple[float, ...], n_state: int, width: int
) -> tuple[Operator, ...]:
    """Exact loader: qubit q holds bit q of x; high bits are prepared first."""
    p = np.asarray(probabilities, dtype=float)
    operations: list[Operator] = []
    for level in range(n_state):
        q = n_state - 1 - level
        # mass[h, b]: probability that the bits above q equal h and bit q is b.
        mass = p.reshape(2**level, 2, 2**q).sum(axis=2)
        for prefix in range(2**level):
            total = mass[prefix].sum()
            if total <= 0:
                continue
            angle = 2 * asin(sqrt(min(1.0, max(0.0, mass[prefix, 1] / total))))
            if angle == 0:
                continue
            if level == 0:
                operations.append(Gate("RY", (q,), width, angle))
                continue
            operations.append(
                ControlledOperator(
                    Gate("RY", (0,), 1, angle),
                    tuple(q + 1 + j for j in range(level)),
                    tuple((prefix >> j) & 1 for j in range(level)),
                    (q,),
                    width,
                )
            )
    return tuple(operations)


def lognormal_tail(
    n_qubits: int, strike: float, *, objective_ancilla: bool = False
) -> BenchmarkProblem:
    """P(S_T >= strike) for a discretized log-normal price: a digital option
    price before discounting, or a tail-exceedance probability.

    With ``objective_ancilla`` the state preparation also computes the flag
    [x >= K] into an extra qubit (the common finance-library pattern), and the
    objective is that single qubit. Otherwise the objective is ``x >= K``.
    """
    grid = lognormal_grid(n_qubits)
    index = int(np.searchsorted(np.asarray(grid.points), strike, side="left"))
    if not 0 < index < 2**n_qubits:
        raise ValueError("strike must fall strictly inside the price grid")
    exact = float(sum(grid.probabilities[index:]))
    state = tuple(range(n_qubits))
    if objective_ancilla:
        width = n_qubits + 1
        comparator = tuple(
            ControlledOperator(
                Gate("X", (0,), 1),
                tuple(q for q, _ in cube),
                tuple(bit for _, bit in cube),
                (n_qubits,),
                width,
            )
            for cube in threshold_cubes(state, index)
        )
        operations = _grover_rudolph(grid.probabilities, n_qubits, width) + comparator
        problem = AEProblem(
            StatePreparation(Sequence(operations, width), width),
            ObjectivePredicate("bit", qubits=(n_qubits,)),
            state,
            objective_register=(n_qubits,),
        )
    else:
        operations = _grover_rudolph(grid.probabilities, n_qubits, n_qubits)
        problem = AEProblem(
            StatePreparation(Sequence(operations, n_qubits), n_qubits),
            ObjectivePredicate("ge", qubits=state, value=index),
            state,
        )
    if not isclose(reference_amplitude(problem), exact, abs_tol=1e-12):
        raise AssertionError("classical tail mass disagrees with statevector")
    return BenchmarkProblem(
        f"lognormal_tail_n{n_qubits}_k{index}" + ("_flag" if objective_ancilla else ""),
        problem,
        exact,
        "independent classical sum of discretized log-normal probabilities",
    )


def strike_for_index(n_qubits: int, index: int) -> float:
    """The grid price at ``index``, so a strike lands exactly on a grid point."""
    return lognormal_grid(n_qubits).points[index]
