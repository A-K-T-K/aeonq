from math import pi
from statistics import median

import numpy as np
import pytest

import aeonq.algorithms.iqae as iqae_module
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.canonical_qae import _weighted_median
from aeonq.algorithms.iqae import IQAE
from aeonq.algorithms.mlqae import MLQAE
from aeonq.benchmarks.problems import bernoulli
from aeonq.scheduling import CostAwareSchedule, ExecutionCost


def test_iqae_interval_touching_within_tolerance_is_not_inverted(monkeypatch):
    # The candidate starts 5e-15 above the feasible interval. The 1e-14 overlap
    # tolerance accepts it, and it must be stored as a valid interval rather
    # than with lower > upper.
    monkeypatch.setattr(
        iqae_module, "phase_interval", lambda *_: (pi / 2 + 5e-15, pi / 2 + 0.1)
    )
    result = IQAE(epsilon=0.1, max_rounds=2, max_power=0).run(
        bernoulli(0.999).problem, ReferenceAdapter()
    )
    lower, upper = result.diagnostics["theta_interval"]
    assert lower == upper == pi / 2
    assert result.convergence_status == "CONVERGED"
    assert result.confidence_or_credible_interval[0] <= result.estimate
    assert result.estimate <= result.confidence_or_credible_interval[1]


@pytest.mark.parametrize("a", [0.0, 0.3, 1.0])
def test_mlqae_single_mode_is_not_reported_as_alias(a):
    # With only k=0 the likelihood is unimodal. Grid endpoints are always
    # refined but must not count as competing modes by themselves.
    result = MLQAE(0.1, powers=(0,)).run(bernoulli(a).problem, ReferenceAdapter())
    assert result.diagnostics["candidate_modes"] == 1
    assert result.diagnostics["alias_risk"] is False


def test_mlqae_estimate_is_global_mle_even_when_k0_interval_misses():
    # The MLE search is not confined to the k=0 Clopper-Pearson interval, so
    # when that interval misses the true phase the estimate can leave it.
    shots, powers = 512, (0, 1, 2, 4, 8)
    theta = np.linspace(0, pi / 2, 400_001)
    a = 0.25
    misses = 0
    for trial in range(60):
        result = MLQAE(
            0.01, powers=powers, shots_per_power=shots, seed=810_001 + trial * 1_000_003
        ).run(bernoulli(a).problem, ReferenceAdapter())
        lo, hi = result.confidence_or_credible_interval
        misses += not lo <= a <= hi
        log_likelihood = np.zeros_like(theta)
        for k, successes in result.diagnostics["observations"]:
            p = np.clip(np.sin((2 * k + 1) * theta) ** 2, 1e-300, 1 - 1e-16)
            log_likelihood += successes * np.log(p) + (shots - successes) * np.log1p(-p)
        best = float(np.sin(theta[np.argmax(log_likelihood)]) ** 2)
        assert result.estimate == pytest.approx(best, abs=1e-5)
        assert result.diagnostics["estimate_outside_interval"] == (
            not lo <= result.estimate <= hi
        )
    assert misses > 0  # the seeds include intervals that miss the true phase


def test_weighted_median_matches_expanded_median():
    rng = np.random.default_rng(3)
    for _ in range(200):
        values = rng.choice([0.0, 0.1464, 0.5, 0.8536, 1.0], size=4, replace=False)
        counts = {float(v): int(c) for v, c in zip(values, rng.integers(0, 6, 4))}
        counts = {v: c for v, c in counts.items() if c} or {0.5: 1}
        expanded = [v for v, c in counts.items() for _ in range(c)]
        assert _weighted_median(counts) == median(expanded)


class _DepthReportingReference(ReferenceAdapter):
    """Reference sampling with a synthetic depth metric and compile counting."""

    def __init__(self) -> None:
        self.compiled = 0

    def compile(self, native_program, options=None):
        self.compiled += 1
        return super().compile(native_program, options)

    def resource_metrics(self, compiled_program):
        return {**super().resource_metrics(compiled_program), "depth": 3}


def test_iqae_supplies_compiled_metrics_to_depth_aware_cost_schedule():
    backend = _DepthReportingReference()
    schedule = CostAwareSchedule(
        ExecutionCost(oracle_weight=1, depth_weight=1), grid_points=33
    )
    result = IQAE(epsilon=0.05, max_rounds=3, max_power=2, scheduler=schedule).run(
        bernoulli(0.3).problem, backend
    )
    assert result.total_shots > 0
    # Candidate powers are compiled once each for their metrics.
    assert backend.compiled <= 3


def test_oracle_only_cost_schedule_compiles_only_executed_powers():
    backend = _DepthReportingReference()
    schedule = CostAwareSchedule(ExecutionCost(oracle_weight=1), grid_points=33)
    result = IQAE(epsilon=0.05, max_rounds=3, max_power=4, scheduler=schedule).run(
        bernoulli(0.3).problem, backend
    )
    assert backend.compiled == len(set(result.powers_used))
