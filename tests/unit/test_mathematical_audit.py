"""Independent finite-dimensional and finite-sample mathematical checks."""

from math import asin, sin, sqrt

import numpy as np
import pytest
from scipy.stats import binom

from aeonq.benchmarks.problems import independent_failures
from aeonq.inference.intervals import clopper_pearson
from aeonq.reference import ideal_success_probability, reference_amplitude


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_random_multiqubit_amplification_matches_classical_formula(n_qubits):
    random = np.random.default_rng(1403 + n_qubits)
    for _ in range(4):
        failure_probabilities = tuple(
            float(p) for p in random.uniform(0.02, 0.8, n_qubits)
        )
        case = independent_failures(failure_probabilities)
        classical = 1 - np.prod([1 - p for p in failure_probabilities])
        assert case.a_true == pytest.approx(classical, abs=1e-12)
        assert reference_amplitude(case.problem) == pytest.approx(classical, abs=1e-12)
        theta = asin(sqrt(float(classical)))
        for power in (0, 1, 2, 4):
            expected = sin((2 * power + 1) * theta) ** 2
            assert ideal_success_probability(case.problem, power) == pytest.approx(
                expected, abs=2e-12
            )


@pytest.mark.parametrize("shots", [1, 5, 12])
@pytest.mark.parametrize("probability", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_clopper_pearson_coverage_by_exhaustive_binomial_enumeration(
    shots, probability
):
    alpha = 0.05
    coverage = sum(
        binom.pmf(successes, shots, probability)
        for successes in range(shots + 1)
        if clopper_pearson(successes, shots, alpha)[0]
        <= probability
        <= clopper_pearson(successes, shots, alpha)[1]
    )
    assert coverage >= 1 - alpha - 1e-12
