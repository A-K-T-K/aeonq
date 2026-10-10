"""Edge cases in precision, sampling, register layout, aliasing and numerics."""

from math import asin, isfinite, sqrt
import pytest

from aeonq import AEProblem
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.algorithms.mlqae import MLQAE
from aeonq.inference.alias import candidate_branches, unambiguous_branch
from aeonq.inference.fisher import fisher_information, success_derivative
from aeonq.inference.intervals import clopper_pearson
from aeonq.inference.likelihood import binomial_log_likelihood
from aeonq.ir import (
    ControlledOperator,
    Gate,
    ObjectivePredicate,
    StatePreparation,
)


def _problem(a: float) -> AEProblem:
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))


# Precision and confidence


@pytest.mark.parametrize("eps", [0.0, -0.1, -1.0])
def test_precision_nonpositive_rejected(eps: float) -> None:
    with pytest.raises(ValueError):
        IQAE(epsilon=eps)
    with pytest.raises(ValueError):
        MLQAE(epsilon=eps)


@pytest.mark.parametrize("eps", [1.0, 1.5])
def test_precision_greater_or_equal_one_rejected(eps: float) -> None:
    with pytest.raises(ValueError):
        IQAE(epsilon=eps)
    with pytest.raises(ValueError):
        MLQAE(epsilon=eps)


@pytest.mark.parametrize("alpha", [0.0, -0.05, 1.0, 1.05])
def test_alpha_out_of_bounds_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        IQAE(epsilon=0.1, alpha=alpha)
    with pytest.raises(ValueError):
        MLQAE(epsilon=0.1, alpha=alpha)


def test_very_small_alpha_terminates_validly() -> None:
    """Extreme confidence requirement (alpha = 1e-6) produces valid finite intervals."""
    problem = _problem(0.3)
    res = IQAE(
        epsilon=0.2, alpha=1e-6, shots_per_round=128, max_rounds=3, max_power=4
    ).run(problem, ReferenceAdapter())
    lo, hi = res.confidence_or_credible_interval
    assert 0.0 <= lo <= res.estimate <= hi <= 1.0
    assert isfinite(res.estimate)


# Sampling and count imbalance


@pytest.mark.parametrize("shots", [1, 2, 5, 10000])
def test_all_zeros_and_all_ones_sampling(shots: int) -> None:
    """Zero successes and full successes produce valid non-NaN intervals."""
    lo_zero, hi_zero = clopper_pearson(0, shots, 0.05)
    assert lo_zero == 0.0
    assert 0.0 < hi_zero <= 1.0

    lo_full, hi_full = clopper_pearson(shots, shots, 0.05)
    assert 0.0 <= lo_full < 1.0
    assert hi_full == 1.0


def test_highly_imbalanced_counts() -> None:
    """Rare events (1 in 100,000) evaluate stably."""
    shots = 100_000
    lo, hi = clopper_pearson(1, shots, 0.05)
    assert 0.0 < lo < hi < 0.001
    ll = binomial_log_likelihood(1, shots, 1e-5)
    assert isfinite(ll)


# Circuit structure validation


def test_invalid_objective_qubit_out_of_range() -> None:
    with pytest.raises(ValueError):
        ObjectivePredicate("bit", qubits=(2,)).validate(2)
    with pytest.raises(ValueError):
        ObjectivePredicate("bit", qubits=(-1,)).validate(2)


def test_duplicate_qubits_in_circuit_nodes() -> None:
    with pytest.raises(ValueError):
        Gate("CX", (0, 0), 2)


def test_control_target_collision() -> None:
    with pytest.raises(ValueError):
        ControlledOperator(Gate("Z", (0,), 1), (0,), (1,), (0,), 2)


def test_register_partitioning_validation() -> None:
    prep = StatePreparation(Gate("H", (0,), 2), 2)
    obj = ObjectivePredicate("bit", qubits=(0,))

    # Missing qubit 1 from partition
    with pytest.raises(ValueError):
        AEProblem(prep, obj, (0,))

    # Overlapping registers
    with pytest.raises(ValueError):
        AEProblem(prep, obj, (0,), work_register=(0, 1))


# Powers and aliasing


def test_k_zero_and_large_k_powers() -> None:
    """k=0 is always monotone; large k correctly detects ambiguous branches."""
    assert unambiguous_branch((0.1, 0.2), 0) == 0

    # Large power creates multiple branches in (0.1, 0.5)
    branches = candidate_branches((0.1, 0.5), 16)
    assert len(branches) > 1
    assert unambiguous_branch((0.1, 0.5), 16) is None


# Numerics and Fisher information


def test_fisher_information_divergent_and_singular_endpoints() -> None:
    f0 = fisher_information(0.0, 2)
    assert not f0.regular
    assert f0.value == float("inf")

    f1 = fisher_information(1.0, 2)
    assert not f1.regular
    assert f1.value == float("inf")

    # Regular interior point
    f_mid = fisher_information(0.5, 2)
    assert f_mid.regular
    assert isfinite(f_mid.value)


def test_success_derivative_exact_endpoints() -> None:
    """dp_k/da at endpoints equals (2k+1)^2."""
    for k in (0, 1, 2, 5):
        m = 2 * k + 1
        assert success_derivative(0.0, k) == float(m * m)
        assert success_derivative(1.0, k) == float(m * m)
