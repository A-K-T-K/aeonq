import math

import pytest

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.algorithms.mlqae import MLQAE
from aeonq.inference.alias import unambiguous_branch
from aeonq.inference.intervals import clopper_pearson
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation


def problem(a: float) -> AEProblem:
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * math.asin(math.sqrt(a))), 1)
    return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))


@pytest.mark.parametrize("a", [0.0, 1e-9, 0.1, 0.5, 0.9, 1 - 1e-9, 1.0])
def test_iqae_seeded_interval(a: float) -> None:
    result = IQAE(
        epsilon=0.1, shots_per_round=256, max_rounds=5, max_power=8, seed=41
    ).run(problem(a), QiskitAdapter())
    lo, hi = result.confidence_or_credible_interval
    assert 0 <= lo <= result.estimate <= hi <= 1
    assert lo - 1e-12 <= a <= hi + 1e-12
    assert result.total_shots == result.iterations * 256
    assert result.total_oracle_queries == sum(
        (2 * k + 1) * 256 for k in result.powers_used
    )
    if result.guarantee_status == "THEORETICAL_GUARANTEE_SATISFIED":
        assert hi - lo <= 0.2 + 1e-12


@pytest.mark.parametrize("a", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_mlqae_seeded_interval(a: float) -> None:
    result = MLQAE(epsilon=0.1, powers=(0, 1, 2), shots_per_power=128, seed=11).run(
        problem(a), QiskitAdapter()
    )
    lo, hi = result.confidence_or_credible_interval
    assert 0 <= lo <= result.estimate <= hi <= 1
    assert lo - 1e-12 <= a <= hi + 1e-12
    assert result.total_shots == 384
    assert result.interval_type == "frequentist confidence interval (k=0 only)"


def test_endpoint_intervals_and_alias_rejection() -> None:
    assert clopper_pearson(0, 10, 0.05)[0] == 0
    assert clopper_pearson(10, 10, 0.05)[1] == 1
    assert unambiguous_branch((0, math.pi / 2), 0) == 0
    assert unambiguous_branch((0, math.pi / 2), 2) is None
    with pytest.raises(ValueError):
        IQAE(epsilon=0)
    with pytest.raises(ValueError):
        MLQAE(epsilon=0.1, powers=(1, 2))


def test_invalid_alpha_and_infeasible_precision_terminate_cleanly() -> None:
    with pytest.raises(ValueError, match="epsilon and alpha"):
        IQAE(epsilon=0.1, alpha=1.0)
    with pytest.raises(ValueError, match="epsilon and alpha"):
        MLQAE(epsilon=0.1, alpha=0.0)
    result = IQAE(
        epsilon=1e-8, shots_per_round=8, max_rounds=1, max_power=0, seed=9
    ).run(problem(0.37), QiskitAdapter())
    assert result.convergence_status == "INSUFFICIENT_BUDGET"
    assert result.guarantee_status == "INSUFFICIENT_BUDGET"
    assert result.iterations == 1
