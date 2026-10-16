from math import pi, sin

import pytest

import aeonq.algorithms.mlqae as mlqae_module
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.faster_ae import (
    ATTENUATION,
    THETA_MAX,
    FasterAE,
    iterations_for_epsilon,
)
from aeonq.algorithms.iqae import IQAE, interval_levels
from aeonq.algorithms.mlqae import MLQAE
from aeonq.benchmarks.problems import bernoulli
from aeonq.scheduling import CostAwareSchedule


@pytest.mark.parametrize("budget", [1, 2, 7, 40])
def test_iqae_interval_levels_spend_exactly_alpha(budget):
    fresh, pooled = interval_levels(0.05, budget)
    # At most B fresh intervals and B(B-1)/2 pooled (segment, depth) pairs.
    assert budget * fresh + budget * (budget - 1) // 2 * pooled == pytest.approx(0.05)


def test_iqae_keeps_converging_when_the_power_cap_binds():
    # Once max_power binds, every round repeats the same power. Pooling those
    # rounds must keep shrinking the interval instead of stalling.
    widths = []
    for rounds in (10, 40):
        result = IQAE(epsilon=1e-5, max_power=2, max_rounds=rounds, seed=5).run(
            bernoulli(0.3).problem, ReferenceAdapter()
        )
        lo, hi = result.confidence_or_credible_interval
        assert lo <= 0.3 <= hi
        widths.append(hi - lo)
        pooled = [
            row["pooled_shots"] for row in result.diagnostics["observation_history"]
        ]
        assert max(pooled) > result.diagnostics["observation_history"][0]["shots"]
    assert widths[1] < 0.6 * widths[0]


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_default_cost_aware_schedule_converges(a):
    result = IQAE(epsilon=0.01, seed=3, scheduler=CostAwareSchedule()).run(
        bernoulli(a).problem, ReferenceAdapter()
    )
    assert result.convergence_status == "CONVERGED"
    assert abs(result.estimate - a) <= 0.01


@pytest.mark.parametrize("epsilon", [0.1, 0.03, 0.01, 0.001])
def test_faster_ae_iterations_meet_epsilon_in_the_worst_case(epsilon):
    slope = ATTENUATION**2 * sin(2 * THETA_MAX)

    def worst_width(iterations):
        return slope * (2 * pi / 3) / (2 ** (iterations + 1) + 2)

    iterations = iterations_for_epsilon(epsilon)
    assert worst_width(iterations) <= 2 * epsilon
    assert iterations == 1 or worst_width(iterations - 1) > 2 * epsilon
    assert FasterAE(epsilon=epsilon).iterations == iterations
    assert FasterAE(epsilon=epsilon, max_iterations=3).iterations == 3


@pytest.mark.parametrize("a", [0.25, 0.5, 0.9, 1.0])
def test_faster_ae_default_iterations_converge(a):
    result = FasterAE(epsilon=0.01, seed=2).run(
        bernoulli(a).problem, ReferenceAdapter()
    )
    assert result.convergence_status == "CONVERGED"
    assert abs(result.estimate - a) <= 0.01


def test_mlqae_withholds_guarantee_when_estimate_leaves_interval(monkeypatch):
    # A narrow k=0 interval that misses the MLE: the epsilon bound on the
    # estimate no longer follows, so the run must not claim it.
    monkeypatch.setattr(mlqae_module, "clopper_pearson", lambda *_: (0.6, 0.605))
    result = MLQAE(0.01, shots_per_power=4096, seed=1).run(
        bernoulli(0.3).problem, ReferenceAdapter()
    )
    assert result.diagnostics["estimate_outside_interval"]
    assert result.convergence_status == "INCONCLUSIVE"
    assert result.guarantee_status == "INCONCLUSIVE"
