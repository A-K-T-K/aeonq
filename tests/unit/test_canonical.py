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
