from math import pi, sin

import numpy as np
import pytest

from aeonq.inference.likelihood import (
    amplified_log_likelihood_grid,
    binomial_log_likelihood,
)


def test_vectorized_grid_matches_scalar_likelihood_at_boundaries_and_aliases():
    grid = np.linspace(0, pi / 2, 257)
    powers = np.array([0, 1, 3, 8], dtype=np.int64)
    successes = np.array([0, 7, 3, 4], dtype=np.int64)
    shots = 7
    measured = amplified_log_likelihood_grid(grid, powers, successes, shots)
    scalar = np.array(
        [
            sum(
                binomial_log_likelihood(
                    int(y), shots, sin((2 * int(k) + 1) * float(theta)) ** 2
                )
                for k, y in zip(powers, successes)
            )
            for theta in grid
        ]
    )
    np.testing.assert_allclose(measured, scalar, rtol=0, atol=2e-11)


def test_vectorized_grid_rejects_invalid_observations():
    with pytest.raises(ValueError):
        amplified_log_likelihood_grid(np.array([0.2]), np.array([0]), np.array([9]), 8)
