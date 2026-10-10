from math import asin, isfinite, sqrt

import pytest

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.algorithms.mlqae import MLQAE
from aeonq.inference.alias import candidate_branches, unambiguous_branch
from aeonq.inference.intervals import clopper_pearson
from aeonq.inference.likelihood import binomial_log_likelihood
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation


AMPLITUDE_GRID = (
    0.0,
    1e-12,
    1e-9,
    1e-6,
    1e-3,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.99,
    1 - 1e-3,
    1 - 1e-6,
    1 - 1e-9,
    1 - 1e-12,
    1.0,
)


def problem(a: float) -> AEProblem:
    return AEProblem(
        StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )


@pytest.mark.parametrize("a", AMPLITUDE_GRID)
def test_boundary_grid_has_finite_estimates_and_intervals(a: float) -> None:
    backend = QiskitAdapter()
    results = (
        IQAE(epsilon=0.1, shots_per_round=128, max_rounds=3, max_power=4, seed=32).run(
            problem(a), backend
        ),
        MLQAE(epsilon=0.1, powers=(0, 1, 2), shots_per_power=64, seed=32).run(
            problem(a), backend
        ),
    )
    for result in results:
        lo, hi = result.confidence_or_credible_interval
        assert all(isfinite(x) for x in (lo, hi, result.estimate))
        assert 0 <= lo <= result.estimate <= hi <= 1
        assert lo - 1e-12 <= a <= hi + 1e-12


def test_log_likelihood_endpoints_and_impossible_outcomes() -> None:
    assert binomial_log_likelihood(0, 100, 0) == 0
    assert binomial_log_likelihood(100, 100, 1) == 0
    assert binomial_log_likelihood(1, 100, 0) == float("-inf")
    assert binomial_log_likelihood(99, 100, 1) == float("-inf")
    assert isfinite(binomial_log_likelihood(1, 100, 1e-300))
    assert clopper_pearson(0, 1, 0.05)[0] == 0
    assert clopper_pearson(1, 1, 0.05)[1] == 1
    with pytest.raises(ValueError):
        binomial_log_likelihood(2, 1, 0.5)


def test_alias_branch_count_and_rejection() -> None:
    feasible = (0.1, 0.8)
    assert len(candidate_branches(feasible, 8)) > 1
    assert unambiguous_branch(feasible, 8) is None
    assert unambiguous_branch(feasible, 0) == 0


class AdversarialQiskitAdapter(QiskitAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def execute(self, compiled_program, shots, *, seed=None):
        self.calls += 1
        if self.calls == 2:
            # Deliberate high-power fluctuation incompatible with the current
            # feasible phase region; later calls use the real simulator.
            return {"1": shots}
        return super().execute(compiled_program, shots, seed=seed)


def test_inconsistent_high_power_observation_triggers_resampling() -> None:
    backend = AdversarialQiskitAdapter()
    a = 0.1
    result = IQAE(
        epsilon=0.01,
        shots_per_round=512,
        max_rounds=3,
        max_power=5,
        max_resamples_per_round=1,
        seed=19,
    ).run(problem(a), backend)
    assert result.powers_used[1] > 0
    assert result.diagnostics["resampling_triggered"]
    assert backend.calls > result.iterations
    assert (
        result.confidence_or_credible_interval[0]
        <= a
        <= result.confidence_or_credible_interval[1]
    )
