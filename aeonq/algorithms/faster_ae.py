"""Two-stage Faster Amplitude Estimation following Nakaji (2020).

The extra Bernoulli qubit attenuates the *amplitude* by 1/5. This makes
theta <= asin(1/5) < 0.252, inside Nakaji's stated initial domain.
Shot counts and the two-stage cosine reconstruction follow Algorithm 1.
Nakaji's finite-sample argument applies under ideal independent shots;
this implementation reports a guarantee only when its returned interval
is narrow enough for the requested epsilon and no consistency check fails.

The returned interval intersects every stage's interval. When it is at most
2*epsilon wide the point estimate is its midpoint, which carries the epsilon
guarantee. Otherwise (budget exhausted or inconclusive) the point estimate is
Nakaji's: the centre (2*pi*n + rho) / (2^(j+1) + 2) of the last second-stage
interval, clipped into the intersection. The midpoint of a wide intersection
is not centred on the measured phase and is strongly biased when theta is
near either end of the domain (for example a close to 1).
"""

from dataclasses import dataclass
from math import acos, asin, atan2, ceil, cos, floor, isfinite, log, pi, sin
from time import perf_counter

from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.base import Experiment, qualified_guarantee
from aeonq.ir.nodes import EmbeddedOperator, Gate, Sequence, StatePreparation
from aeonq.ir.predicates import ObjectivePredicate
from aeonq.problem import AEProblem
from aeonq.runtime.result import AEResult


ATTENUATION = 5.0
THETA_MAX = asin(1 / ATTENUATION)


def _lift_predicate(
    predicate: ObjectivePredicate, old_width: int
) -> ObjectivePredicate:
    """Extend a predicate to one extra high qubit without changing its truth set."""
    if predicate.kind == "marked":
        return ObjectivePredicate(
            "marked",
            marked=tuple(y for x in predicate.marked for y in (x, x + 2**old_width)),
        )
    if predicate.kind in ("and", "or", "not"):
        return ObjectivePredicate(
            predicate.kind,
            children=tuple(
                _lift_predicate(child, old_width) for child in predicate.children
            ),
        )
    return predicate


def attenuated_problem(problem: AEProblem) -> AEProblem:
    """Construct A' = A ⊗ Ry(2 asin(1/5)), χ' = χ AND extra_bit."""
    n = problem.state_preparation.n_qubits
    prep = StatePreparation(
        Sequence(
            (
                EmbeddedOperator(problem.state_preparation, tuple(range(n)), n + 1),
                Gate("RY", (n,), n + 1, 2 * THETA_MAX),
            ),
            n + 1,
        ),
        n + 1,
    )
    objective = ObjectivePredicate(
        "and",
        children=(
            _lift_predicate(problem.objective, n),
            ObjectivePredicate("bit", qubits=(n,)),
        ),
    )
    return AEProblem(
        prep,
        objective,
        problem.state_register + (n,),
        problem.work_register,
        problem.objective_register,
        metadata={"attenuation": ATTENUATION},
    )


@dataclass(frozen=True)
class FasterAE:
    epsilon: float
    alpha: float = 0.05
    max_iterations: int = 4
    max_total_shots: int = 1_000_000
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 1 or not 0 < self.alpha < 1:
            raise ValueError("epsilon and alpha must be in (0,1)")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(self.max_total_shots, int) or self.max_total_shots < 1:
            raise ValueError("max_total_shots must be positive")

    @property
    def required_shots(self) -> tuple[int, int]:
        delta = self.alpha / (2 * self.max_iterations)
        multiplier = log(2 / delta)
        return ceil(1944 * multiplier), ceil(972 * multiplier)

    def run(self, problem: AEProblem, backend: BackendAdapter) -> AEResult:
        started = perf_counter()
        adapted = attenuated_problem(problem)
        experiment = Experiment(adapted, backend)
        n_first, n_second = self.required_shots
        delta = self.alpha / (2 * self.max_iterations)
        theta = (0.0, THETA_MAX)
        first_stage = True
        j0: int | None = None
        nu: float | None = None
        centre: float | None = None
        powers: list[int] = []
        per_power: dict[int, int] = {}
        history: list[dict[str, float | int | str]] = []
        status = "INSUFFICIENT_BUDGET"

        def sample_cosine(power: int, shots: int, salt: int) -> float:
            successes = experiment.sample(power, shots, seed=self.seed + salt)
            powers.append(power)
            per_power[power] = per_power.get(power, 0) + shots
            return 1.0 - 2.0 * successes / shots

        for j in range(1, self.max_iterations + 1):
            main_power = 2 ** (j - 1)
            shots = n_first if first_stage else n_second
            needed = shots if first_stage else 2 * shots
            if experiment.shots + needed > self.max_total_shots:
                break
            previous = theta
            cosine = sample_cosine(main_power, shots, 2 * j)
            denominator = 2 ** (j + 1) + 2
            if first_stage:
                radius = (12 * log(2 / delta) / shots) ** 0.5
                c_low = max(-1.0, cosine - radius)
                c_high = min(1.0, cosine + radius)
                candidate = (acos(c_high) / denominator, acos(c_low) / denominator)
                history.append(
                    {
                        "iteration": j,
                        "stage": "first",
                        "power": main_power,
                        "cosine": cosine,
                        "cosine_radius": radius,
                    }
                )
            else:
                assert j0 is not None and nu is not None
                shifted_power = main_power + 2 ** (j0 - 1)
                shifted_cosine = sample_cosine(shifted_power, shots, 2 * j + 1)
                if abs(sin(nu)) < 1e-12:
                    status = "INCONCLUSIVE"
                    break
                sine_estimate = (cosine * cos(nu) - shifted_cosine) / sin(nu)
                rho = atan2(sine_estimate, cosine)
                branch = floor((denominator * previous[1] - rho + pi / 3) / (2 * pi))
                candidate = (
                    (2 * pi * branch + rho - pi / 3) / denominator,
                    (2 * pi * branch + rho + pi / 3) / denominator,
                )
                history.append(
                    {
                        "iteration": j,
                        "stage": "second",
                        "power": main_power,
                        "shifted_power": shifted_power,
                        "branch": branch,
                        "cosine": cosine,
                        "shifted_cosine": shifted_cosine,
                        "sine_estimate": sine_estimate,
                        "rho": rho,
                    }
                )
            theta = (
                max(previous[0], candidate[0], 0.0),
                min(previous[1], candidate[1], THETA_MAX),
            )
            if not all(isfinite(x) for x in theta) or theta[0] > theta[1] + 1e-14:
                theta = previous
                status = "INCONCLUSIVE"
                break
            if theta[0] > theta[1]:
                # Touching within rounding: collapse rather than invert.
                middle = min(max(sum(theta) / 2, previous[0]), previous[1])
                theta = (middle, middle)
            if not first_stage:
                centre = sum(candidate) / 2
            original_interval = (
                max(0.0, (ATTENUATION * sin(theta[0])) ** 2),
                min(1.0, (ATTENUATION * sin(theta[1])) ** 2),
            )
            if original_interval[1] - original_interval[0] <= 2 * self.epsilon:
                status = "CONVERGED"
                break
            if (
                first_stage
                and 2 ** (j + 1) * theta[1] >= 3 * pi / 8
                and j < self.max_iterations
            ):
                j0 = j
                nu = 2**j0 * sum(theta)
                first_stage = False

        original_interval = (
            max(0.0, (ATTENUATION * sin(theta[0])) ** 2),
            min(1.0, (ATTENUATION * sin(theta[1])) ** 2),
        )
        theta_estimate = (
            sum(theta) / 2 if centre is None else min(max(centre, theta[0]), theta[1])
        )
        if status == "CONVERGED":
            # Width <= 2*epsilon: only the midpoint keeps the epsilon guarantee.
            estimate = sum(original_interval) / 2
        else:
            estimate = min(1.0, (ATTENUATION * sin(theta_estimate)) ** 2)
        guarantee, warnings = qualified_guarantee(
            backend,
            "THEORETICAL_GUARANTEE_SATISFIED"
            if status == "CONVERGED"
            else "INCONCLUSIVE"
            if status == "INCONCLUSIVE"
            else "INSUFFICIENT_BUDGET",
        )
        return AEResult(
            estimate,
            original_interval,
            "frequentist confidence interval",
            self.epsilon,
            self.alpha,
            status,
            "finite-sample frequentist",
            guarantee,
            (
                "ideal independent Bernoulli shots",
                "exact attenuation by 1/5",
                "exact state preparation, reflections and measurement",
                "Nakaji two-stage cosine reconstruction with published shot bounds",
            ),
            experiment.shots,
            experiment.oracle_queries,
            experiment.evaluations,
            tuple(powers),
            per_power,
            max(powers, default=0),
            len(history),
            experiment.compile_time,
            experiment.execution_time,
            perf_counter() - started,
            type(backend).__name__,
            type(backend).__module__.split(".")[-1],
            next(iter(backend.capabilities().metadata.values()), "unknown"),
            experiment.resource_metrics or {},
            warnings,
            {
                "theta_interval": theta,
                "theta_estimate": theta_estimate,
                "stage_history": tuple(history),
                "switch_iteration": j0,
                "nu": nu,
                "required_shots": (n_first, n_second),
                "delta_per_cosine": delta,
                "attenuation": ATTENUATION,
            },
        )
