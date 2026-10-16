"""Canonical QPE-based amplitude estimation on semantic controlled Grover powers.

The point estimate is the maximum-likelihood amplitude under the exact QPE
outcome model, and the interval is the likelihood-ratio set calibrated by the
chi-square(1) approximation, as in Suzuki et al. and qiskit-algorithms. The
interval is asymptotic: it carries no finite-sample guarantee, and with few
evaluation qubits its coverage can fall below nominal (about 92% at m=3, 256
shots, a=0.1 for a nominal 95%). The median of
the per-shot estimates sin^2(pi*y/M) is reported in the diagnostics only; it
takes just M/2+1 distinct values and rarely lies inside the likelihood-ratio
interval.
"""

from dataclasses import dataclass
from math import pi, sin
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.base import qualified_guarantee
from aeonq.ir.nodes import (
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverPower,
    Operator,
    Sequence,
)
from aeonq.ir.registers import RegisterMap
from aeonq.problem import AEProblem
from aeonq.reference import grover_for
from aeonq.runtime.result import AEResult


def _swap(a: int, b: int, n: int) -> tuple[Operator, ...]:
    return (Gate("CX", (a, b), n), Gate("CX", (b, a), n), Gate("CX", (a, b), n))


def inverse_qft(m: int, total_qubits: int) -> Sequence:
    """Exact inverse QFT on low m logical qubits, including output reversal."""
    if m < 1 or total_qubits < m:
        raise ValueError("invalid Fourier register")
    nodes: list[Operator] = []
    for j in range(m // 2):
        nodes.extend(_swap(j, m - j - 1, total_qubits))
    for j in range(m):
        for k in range(j):
            angle = -pi / (2 ** (j - k))
            local = Sequence((GlobalPhase(angle / 2, 1), Gate("RZ", (0,), 1, angle)), 1)
            nodes.append(ControlledOperator(local, (k,), (1,), (j,), total_qubits))
        nodes.append(Gate("H", (j,), total_qubits))
    return Sequence(tuple(nodes), total_qubits)


def _weighted_median(counts: dict[float, int]) -> float:
    """``statistics.median`` of the expanded sample, without expanding it."""
    ordered = sorted(counts.items())
    total = sum(count for _, count in ordered)
    if total < 1:
        raise ValueError("median of an empty sample")

    def order_statistic(position: int) -> float:
        seen = 0
        for value, count in ordered:
            seen += count
            if position < seen:
                return value
        raise AssertionError("position outside sample")

    if total % 2:
        return float(order_statistic(total // 2))
    return (order_statistic(total // 2 - 1) + order_statistic(total // 2)) / 2


def qpe_outcome_probabilities(theta: np.ndarray | float, m: int) -> np.ndarray:
    """Exact P(y | theta) of the m-qubit QPE readout, one row per theta.

    Q has eigenphases +-2*theta, each carrying half the probability, so the
    readout is the mean of two Fejer kernels centred at y/M = +-theta/pi.
    """
    modulus = 2**m
    thetas = np.atleast_1d(np.asarray(theta, dtype=float))
    phases = np.arange(modulus) / modulus

    def fejer(x: np.ndarray) -> np.ndarray:
        s = np.sin(pi * x)
        small = np.abs(s) < 1e-12
        safe = np.where(small, 1.0, s)
        kernel = np.sin(modulus * pi * x) ** 2 / (modulus**2 * safe**2)
        return np.where(small, 1.0, kernel)

    shift = thetas[:, None] / pi
    return 0.5 * (fejer(phases[None, :] - shift) + fejer(phases[None, :] + shift))


def likelihood_ratio_fit(
    histogram: dict[int, int], m: int, alpha: float
) -> tuple[float, tuple[float, float], float]:
    """Maximum-likelihood theta and the chi-square likelihood-ratio interval.

    Returns (theta_mle, (theta_low, theta_high), maximum log-likelihood) with
    theta in [0, pi/2], so a = sin^2(theta) is monotone in it.
    """
    modulus = 2**m
    counts = np.array([histogram.get(y, 0) for y in range(modulus)], dtype=float)

    def log_likelihood(theta: np.ndarray | float) -> np.ndarray:
        probabilities = np.clip(qpe_outcome_probabilities(theta, m), 1e-300, 1.0)
        return np.log(probabilities) @ counts

    def scalar(theta: float) -> float:
        return float(log_likelihood(theta)[0])

    # Each Fejer peak is about pi/M wide; 2048 grid points per peak width.
    grid = np.linspace(0.0, pi / 2, 1024 * modulus + 1)
    values = log_likelihood(grid)
    best = int(np.argmax(values))
    theta_mle, best_value = float(grid[best]), float(values[best])
    lo, hi = float(grid[max(0, best - 1)]), float(grid[min(len(grid) - 1, best + 1)])
    if hi > lo:
        refined = minimize_scalar(
            lambda t: -scalar(float(t)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-13},
        )
        if refined.success and -refined.fun > best_value:
            theta_mle, best_value = float(refined.x), float(-refined.fun)

    threshold = best_value - chi2.ppf(1 - alpha, 1) / 2
    accepted = np.flatnonzero(values >= threshold)
    if accepted.size == 0:  # the grid missed a very sharp peak
        return theta_mle, (theta_mle, theta_mle), best_value

    def boundary(inside: float, outside: float) -> float:
        """Bisect for the threshold crossing between an accepted and a rejected theta."""
        for _ in range(60):
            middle = (inside + outside) / 2
            if scalar(middle) >= threshold:
                inside = middle
            else:
                outside = middle
        return inside

    first, last = int(accepted[0]), int(accepted[-1])
    low = 0.0 if first == 0 else boundary(float(grid[first]), float(grid[first - 1]))
    high = (
        pi / 2
        if last == len(grid) - 1
        else boundary(float(grid[last]), float(grid[last + 1]))
    )
    return theta_mle, (min(low, theta_mle), max(high, theta_mle)), best_value


@dataclass(frozen=True)
class CanonicalQAE:
    evaluation_qubits: int = 4
    shots: int = 256
    epsilon: float = 0.1
    alpha: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.evaluation_qubits <= 6 or self.shots < 1:
            raise ValueError("evaluation qubits must be 1..6 and shots positive")
        if not 0 < self.epsilon < 1 or not 0 < self.alpha < 1:
            raise ValueError("epsilon and alpha must be in (0,1)")

    def circuit(self, problem: AEProblem) -> Sequence:
        q = grover_for(problem)
        m = self.evaluation_qubits
        n = m + q.n_qubits
        state_targets = tuple(range(m, n))
        nodes: list[Operator] = [
            EmbeddedOperator(problem.state_preparation, state_targets, n)
        ]
        nodes.extend(Gate("H", (j,), n) for j in range(m))
        for j in range(m):
            nodes.append(
                ControlledOperator(
                    GroverPower(q, 2**j, q.n_qubits), (j,), (1,), state_targets, n
                )
            )
        nodes.append(inverse_qft(m, n))
        return Sequence(tuple(nodes), n)

    def run(self, problem: AEProblem, backend: BackendAdapter) -> AEResult:
        started = perf_counter()
        circuit = self.circuit(problem)
        start = perf_counter()
        compiled = backend.compile(backend.lower(circuit))
        compile_time = perf_counter() - start
        start = perf_counter()
        raw = backend.execute(compiled, self.shots, seed=self.seed)
        execution_time = perf_counter() - start
        mapping = RegisterMap(
            tuple(range(circuit.n_qubits)), tuple(range(circuit.n_qubits))
        )
        result = backend.normalize_result(raw, mapping)
        if result.shots != self.shots:
            raise ValueError("backend returned wrong shot count")
        amplitudes: dict[float, int] = {}
        histogram: dict[int, int] = {}
        modulus = 2**self.evaluation_qubits
        for bits, count in result.counts.items():
            y = sum(bits[j] << j for j in range(self.evaluation_qubits))
            histogram[y] = histogram.get(y, 0) + count
            value = sin(pi * y / modulus) ** 2
            amplitudes[value] = amplitudes.get(value, 0) + count
        theta_mle, (theta_low, theta_high), log_likelihood = likelihood_ratio_fit(
            histogram, self.evaluation_qubits, self.alpha
        )
        estimate = sin(theta_mle) ** 2
        interval = (sin(theta_low) ** 2, sin(theta_high) ** 2)
        metrics = dict(backend.resource_metrics(compiled))
        metrics["total_compiled_depth"] = metrics.get("depth")
        metrics["total_two_qubit_gates"] = metrics.get("two_qubit_gates")
        guarantee, warnings = qualified_guarantee(backend, "EMPIRICAL_VALIDATION_ONLY")
        return AEResult(
            estimate,
            interval,
            "asymptotic likelihood-ratio interval",
            self.epsilon,
            self.alpha,
            "FIXED_BUDGET",
            "asymptotic (chi-square likelihood ratio); no finite-sample guarantee",
            guarantee,
            (
                "ideal phase estimation circuit",
                "exact controlled Grover powers",
                "exact QPE outcome model; chi-square(1) approximation for the "
                "likelihood-ratio statistic",
            ),
            self.shots,
            self.shots * (2 * (modulus - 1) + 1),
            1,
            tuple(2**j for j in range(self.evaluation_qubits)),
            {2**j: self.shots for j in range(self.evaluation_qubits)},
            2 ** (self.evaluation_qubits - 1),
            1,
            compile_time,
            execution_time,
            perf_counter() - started,
            type(backend).__name__,
            type(backend).__module__.split(".")[-1],
            next(iter(backend.capabilities().metadata.values()), "unknown"),
            metrics,
            warnings,
            {
                "phase_histogram": histogram,
                "median_estimate": _weighted_median(amplitudes),
                "theta_mle": theta_mle,
                "theta_interval": (theta_low, theta_high),
                "log_likelihood": log_likelihood,
            },
        )
