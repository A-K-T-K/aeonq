from math import asin, pi, sin, sqrt

import numpy as np
import pytest

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.algorithms.canonical_qae import (
    CanonicalQAE,
    inverse_qft,
    likelihood_ratio_fit,
    qpe_outcome_probabilities,
)
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation
from aeonq.reference import matrix, statevector


@pytest.mark.parametrize("successes", [0, 15_000_000, 50_000_000])
def test_sharp_likelihood_interval_matches_independent_binomial_roots(successes):
    # For m=1 QPE is exactly Bernoulli(a), so its LR boundaries can be
    # independently solved without the QPE model or its phase grid.
    from scipy.optimize import brentq
    from scipy.special import xlogy
    from scipy.stats import chi2

    shots = 50_000_000
    probability = successes / shots

    def likelihood(a):
        return float(xlogy(successes, a) + xlogy(shots - successes, 1 - a))

    cutoff = likelihood(probability) - chi2.ppf(0.95, 1) / 2
    expected_low = (
        0.0
        if successes == 0
        else brentq(lambda a: likelihood(a) - cutoff, 0.0, probability, xtol=1e-15)
    )
    expected_high = (
        1.0
        if successes == shots
        else brentq(lambda a: likelihood(a) - cutoff, probability, 1.0, xtol=1e-15)
    )
    theta, (low, high), _ = likelihood_ratio_fit(
        {0: shots - successes, 1: successes}, 1, 0.05
    )
    assert low < high
    assert sin(theta) ** 2 == pytest.approx(probability, abs=2e-8)
    assert sin(low) ** 2 == pytest.approx(expected_low, abs=2e-9)
    assert sin(high) ** 2 == pytest.approx(expected_high, abs=2e-9)


def test_likelihood_interval_contains_both_sharp_disconnected_modes():
    # At m=2, P(0)=(1-a)(1-2a)^2 and P(2)=a(1-2a)^2. Equal counts
    # have two exact MLEs, (1 +/- sqrt(2/3))/2, separated by a zero at a=1/2.
    from scipy.optimize import brentq
    from scipy.stats import chi2

    shots = 50_000_000
    mode = (1 - sqrt(2 / 3)) / 2

    def likelihood(a):
        return shots / 2 * np.log(a * (1 - a)) + shots * np.log((1 - 2 * a) ** 2)

    cutoff = likelihood(mode) - chi2.ppf(0.95, 1) / 2
    expected_low = brentq(lambda a: likelihood(a) - cutoff, 0.08, mode, xtol=1e-15)
    theta, (low, high), _ = likelihood_ratio_fit(
        {0: shots // 2, 2: shots // 2}, 2, 0.05
    )
    assert min(abs(sin(theta) ** 2 - mode), abs(sin(theta) ** 2 - (1 - mode))) < 2e-8
    assert sin(low) ** 2 == pytest.approx(expected_low, abs=2e-9)
    assert sin(high) ** 2 == pytest.approx(1 - expected_low, abs=2e-9)


def test_sharp_mode_between_endpoint_and_first_grid_point_is_refined():
    from scipy.optimize import brentq
    from scipy.stats import chi2

    shots = 1_000_000_000_000
    probability = 1 / shots

    def likelihood(a):
        return np.log(a) + (shots - 1) * np.log1p(-a)

    cutoff = likelihood(probability) - chi2.ppf(0.95, 1) / 2
    expected_low = brentq(
        lambda a: likelihood(a) - cutoff, 1e-20, probability, xtol=1e-25
    )
    expected_high = brentq(
        lambda a: likelihood(a) - cutoff, probability, 1e-10, xtol=1e-25
    )
    theta, (low, high), _ = likelihood_ratio_fit({0: shots - 1, 1: 1}, 1, 0.05)
    # At this scale the finite-precision QPE probabilities perturb the MLE
    # slightly; the independently solved boundaries remain the tighter check.
    assert sin(theta) ** 2 == pytest.approx(probability, abs=5e-14)
    assert sin(low) ** 2 == pytest.approx(expected_low, abs=5e-15)
    assert sin(high) ** 2 == pytest.approx(expected_high, abs=5e-15)


def test_unresolvable_likelihood_threshold_is_inconclusive():
    from aeonq.adapters.reference import ReferenceAdapter
    from aeonq.benchmarks.problems import bernoulli

    # The valid but tiny LR cutoff rounds to zero at the likelihood's scale.
    result = CanonicalQAE(
        evaluation_qubits=1, shots=512, alpha=float(np.nextafter(1.0, 0.0))
    ).run(bernoulli(0.3).problem, ReferenceAdapter())
    assert result.convergence_status == result.guarantee_status == "INCONCLUSIVE"
    assert result.confidence_or_credible_interval == (0.0, 1.0)
    assert result.diagnostics["theta_mle"] is None
    assert result.diagnostics["likelihood_fit_error"]
    assert result.warnings


@pytest.mark.parametrize("m", [1, 2, 3])
def test_inverse_qft_matrix(m: int) -> None:
    qft_inv = matrix(inverse_qft(m, m))
    modulus = 2**m
    expected = np.array(
        [
            [
                np.exp(-2j * pi * x * y / modulus) / np.sqrt(modulus)
                for y in range(modulus)
            ]
            for x in range(modulus)
        ]
    )
    assert np.allclose(qft_inv, expected, atol=1e-10)


@pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_canonical_seeded(a: float) -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    result = CanonicalQAE(evaluation_qubits=3, shots=128, seed=7).run(
        problem, QiskitAdapter()
    )
    lo, hi = result.confidence_or_credible_interval
    assert 0 <= lo <= result.estimate <= hi <= 1
    assert abs(result.estimate - a) <= 0.05
    assert result.interval_type == "asymptotic likelihood-ratio interval"
    assert result.guarantee_status == "EMPIRICAL_VALIDATION_ONLY"


@pytest.mark.parametrize("a", [0.0, 0.01, 0.3, 0.5, 0.77, 1.0])
@pytest.mark.parametrize("m", [1, 2, 4])
def test_qpe_outcome_model_matches_circuit(a: float, m: int) -> None:
    # The likelihood model must match the exact readout distribution of the
    # constructed circuit, in the order run() reads y.
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    state = statevector(CanonicalQAE(evaluation_qubits=m).circuit(problem))
    probabilities = np.abs(state) ** 2
    modulus = 2**m
    readout = np.array(
        [
            probabilities[np.arange(len(state)) % modulus == y].sum()
            for y in range(modulus)
        ]
    )
    model = qpe_outcome_probabilities(asin(sqrt(a)), m)[0]
    assert np.allclose(model, readout, atol=1e-12)
    assert model.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("a", [1e-6, 0.1, 0.25, 0.5, 0.9, 1 - 1e-6])
def test_likelihood_ratio_interval_coverage_is_near_nominal(a: float) -> None:
    # The interval is asymptotic, so this checks calibration, not a guarantee:
    # 200 exact multinomial samples per amplitude, nominal 0.95.
    m, shots = 4, 512
    model = qpe_outcome_probabilities(asin(sqrt(a)), m)[0]
    rng = np.random.default_rng(2024)
    covered = 0
    for _ in range(200):
        draws = rng.multinomial(shots, model)
        histogram = {y: int(c) for y, c in enumerate(draws) if c}
        theta, (low, high), _ = likelihood_ratio_fit(histogram, m, 0.05)
        assert low <= theta <= high
        covered += sin(low) ** 2 <= a <= sin(high) ** 2
    assert covered / 200 >= 0.9


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_canonical_cross_sdk_small_circuit(backend: type) -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 0.7), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    circuit = CanonicalQAE(evaluation_qubits=2, shots=8).circuit(problem)
    assert np.allclose(backend().unitary(circuit), matrix(circuit), atol=1e-10)


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_embedded_idle_width_cross_sdk(backend: type) -> None:
    from aeonq.ir import EmbeddedOperator, GlobalPhase

    node = EmbeddedOperator(GlobalPhase(0.3, 1), (2,), 3)
    assert np.allclose(backend().unitary(node), matrix(node), atol=1e-10)
