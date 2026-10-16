"""Fixed-schedule maximum-likelihood amplitude estimation.

The estimate uses all amplified counts. Its reported exact confidence interval
comes only from the k=0 Clopper–Pearson observation, so interval coverage does
not rely on asymptotic likelihood-ratio theory or alias resolution.

The maximum-likelihood search covers the whole phase range [0, pi/2], not only
the k=0 interval, so the estimate is not pinned inside an interval that misses
the true phase. The estimate can therefore lie outside the reported interval;
the ``estimate_outside_interval`` diagnostic records when it does, and such a
run is INCONCLUSIVE rather than CONVERGED, because the epsilon bound on the
estimate holds only when it lies inside the interval.
"""

from dataclasses import dataclass
from math import asin, pi, sin, sqrt
from time import perf_counter

import numpy as np
from scipy.optimize import minimize_scalar

from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.base import Experiment, qualified_guarantee
from aeonq.inference.intervals import clopper_pearson
from aeonq.inference.likelihood import (
    amplified_log_likelihood_grid,
    binomial_log_likelihood,
)
from aeonq.problem import AEProblem
from aeonq.runtime.result import AEResult


@dataclass(frozen=True)
class MLQAE:
    epsilon: float
    alpha: float = 0.05
    powers: tuple[int, ...] = (0, 1, 2, 4, 8)
    shots_per_power: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 1 or not 0 < self.alpha < 1:
            raise ValueError("epsilon and alpha must be in (0,1)")
        if (
            not self.powers
            or 0 not in self.powers
            or any(not isinstance(k, int) or k < 0 for k in self.powers)
        ):
            raise ValueError(
                "MLQAE powers must include zero and be nonnegative integers"
            )
        if self.shots_per_power < 1:
            raise ValueError("shots_per_power must be positive")

    def run(self, problem: AEProblem, backend: BackendAdapter) -> AEResult:
        start = perf_counter()
        experiment = Experiment(problem, backend)
        observations = [
            (k, experiment.sample(k, self.shots_per_power, self.seed + i))
            for i, k in enumerate(self.powers)
        ]
        successes_0 = sum(y for k, y in observations if k == 0)
        shots_0 = self.shots_per_power * sum(k == 0 for k, _ in observations)
        interval = clopper_pearson(successes_0, shots_0, self.alpha)
        lower = asin(sqrt(interval[0]))
        upper = asin(sqrt(interval[1]))

        def log_likelihood(theta: float) -> float:
            total = 0.0
            for k, successes in observations:
                p = sin((2 * k + 1) * theta) ** 2
                total += binomial_log_likelihood(successes, self.shots_per_power, p)
            return total

        powers = np.asarray([k for k, _ in observations], dtype=np.int64)
        counts = np.asarray(
            [successes for _, successes in observations], dtype=np.int64
        )

        def grid_values(grid: np.ndarray) -> np.ndarray:
            return amplified_log_likelihood_grid(
                grid, powers, counts, self.shots_per_power
            )

        def local_maxima(values: np.ndarray) -> list[int]:
            return (
                np.flatnonzero(
                    (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
                )
                + 1
            ).tolist()

        # Alias diagnostics count competing modes inside the k=0 interval only;
        # modes it excludes at the stated confidence are not plausible aliases.
        local_grid = np.linspace(lower, upper, max(1025, 128 * max(self.powers)))
        local_values = grid_values(local_grid)
        # Endpoints are modes only if the likelihood does not rise away from them.
        mode_count = (
            len(local_maxima(local_values))
            + int(local_values[0] >= local_values[1])
            + int(local_values[-1] >= local_values[-2])
        )

        # Search every basin of the whole phase range and refine each local
        # candidate. The finest likelihood term has period pi/(2k+1), so the
        # grid scales with 2k+1.
        grid = np.linspace(0.0, pi / 2, max(4097, 512 * (2 * max(self.powers) + 1)))
        values = grid_values(grid)
        candidate_indices = local_maxima(values) + [0, len(grid) - 1]
        best_theta = float(grid[int(np.nanargmax(values))])
        best_value = log_likelihood(best_theta)
        for i in candidate_indices:
            lo = float(grid[max(0, i - 1)])
            hi = float(grid[min(len(grid) - 1, i + 1)])
            if hi <= lo:
                continue
            optimum = minimize_scalar(
                lambda t: -log_likelihood(float(t)),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-13},
            )
            if optimum.success and -optimum.fun > best_value:
                best_theta, best_value = float(optimum.x), float(-optimum.fun)
        estimate = sin(best_theta) ** 2
        outside = not interval[0] <= estimate <= interval[1]
        # The point estimate is the MLE, not the midpoint, so the epsilon bound
        # needs the full width <= epsilon and the estimate inside the interval.
        # An MLE outside it means the k=0 interval missed or the MLE aliased.
        if outside:
            status = "INCONCLUSIVE"
        elif interval[1] - interval[0] > self.epsilon:
            status = "INSUFFICIENT_BUDGET"
        else:
            status = "CONVERGED"
        guarantee, warnings = qualified_guarantee(
            backend,
            "THEORETICAL_GUARANTEE_SATISFIED" if status == "CONVERGED" else status,
        )
        return AEResult(
            estimate,
            interval,
            "frequentist confidence interval (k=0 only)",
            self.epsilon,
            self.alpha,
            status,
            "finite-sample frequentist interval",
            guarantee,
            (
                "ideal independent Bernoulli shots",
                "fixed, nonadaptive power schedule",
                "exact state preparation and objective measurement",
                "MLE is numerically optimized over the whole phase range",
            ),
            experiment.shots,
            experiment.oracle_queries,
            experiment.evaluations,
            self.powers,
            {k: self.shots_per_power * self.powers.count(k) for k in set(self.powers)},
            max(self.powers),
            len(self.powers),
            experiment.compile_time,
            experiment.execution_time,
            perf_counter() - start,
            type(backend).__name__,
            type(backend).__module__.split(".")[-1],
            next(iter(backend.capabilities().metadata.values()), "unknown"),
            experiment.resource_metrics or {},
            warnings,
            {
                "observations": tuple(observations),
                "log_likelihood": best_value,
                "candidate_modes": mode_count,
                "alias_risk": mode_count > 1,
                "estimate_outside_interval": outside,
            },
        )
