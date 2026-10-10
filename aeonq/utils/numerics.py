"""Central numerical policy and phase-aware comparisons."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NumericalPolicy:
    atol: float = 1e-10
    rtol: float = 1e-10
    probability_clip: float = 1e-14
    log_floor: float = 1e-300


DEFAULT_POLICY = NumericalPolicy()


def aligned_operator_distance(
    a: NDArray[np.complex128], b: NDArray[np.complex128]
) -> float:
    """Frobenius distance after aligning a *permissible* global phase."""
    if a.shape != b.shape:
        raise ValueError("operator shapes differ")
    overlap = np.vdot(b, a)
    if abs(overlap) == 0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - (overlap / abs(overlap)) * b))


def unitary_equivalent_up_to_phase(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    policy: NumericalPolicy = DEFAULT_POLICY,
) -> bool:
    return aligned_operator_distance(a, b) <= policy.atol + policy.rtol * float(
        np.linalg.norm(a)
    )


def controlled_unitary_equivalent(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    policy: NumericalPolicy = DEFAULT_POLICY,
) -> bool:
    """Exact matrix comparison; a controlled global phase is observable."""
    return bool(np.allclose(a, b, atol=policy.atol, rtol=policy.rtol))


def global_phase_aligned_fidelity(
    a: NDArray[np.complex128], b: NDArray[np.complex128]
) -> float:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("statevectors must have matching one-dimensional shape")
    denom = float(np.vdot(a, a).real * np.vdot(b, b).real)
    if denom == 0:
        raise ValueError("zero-norm statevector")
    return min(1.0, float(abs(np.vdot(a, b)) ** 2 / denom))
