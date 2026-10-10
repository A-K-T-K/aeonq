"""A second simulator: plain NumPy, basis-index arithmetic, no tensors.

Gates are single-target 2x2 matrices with optional controls, diagonal sign
flips given by a predicate on basis indices, and (controlled) global phases.
Basis index x is little-endian: bit q of x is qubit q. Each gate updates the
pairs of amplitudes (x, x | 1 << t) with bit t of x clear and every control
satisfied, for all columns at once.

Matrices are the textbook definitions, written out here rather than taken
from an SDK or from aeonq.reference.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import cos, sin
from typing import Any

import numpy as np


def gate_matrix(name: str, parameters: tuple[Any, ...]) -> np.ndarray:
    if name == "H":
        return np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    if name == "X":
        return np.array([[0, 1], [1, 0]], complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], complex)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], complex)
    if name in ("RX", "RY", "RZ"):
        (theta,) = parameters
        c, s = cos(theta / 2), sin(theta / 2)
        if name == "RX":
            return np.array([[c, -1j * s], [-1j * s, c]])
        if name == "RY":
            return np.array([[c, -s], [s, c]], complex)
        return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])
    if name == "P":
        (phi,) = parameters
        return np.array([[1, 0], [0, np.exp(1j * phi)]])
    if name == "U":
        # Qiskit's documented U(theta, phi, lambda); no extra global phase.
        theta, phi, lam = parameters
        c, s = cos(theta / 2), sin(theta / 2)
        return np.array(
            [
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
            ]
        )
    if name == "M":
        # Explicit 2x2 matrix given as four complex entries in row order.
        return np.asarray(parameters, complex).reshape(2, 2)
    raise ValueError(f"unknown gate {name}")


@dataclass(frozen=True)
class Op:
    """One operation. ``target`` is None for FLIP and GPHASE.

    FLIP multiplies basis states x with predicate(x) by -1. GPHASE multiplies
    by exp(i * parameter), restricted to the control subspace if controlled.
    """

    name: str
    target: int | None
    # A float, or a tuple of floats (U) or of complex matrix entries (M).
    parameter: float | tuple[Any, ...] | None = None
    controls: tuple[int, ...] = ()
    control_values: tuple[int, ...] = ()
    predicate: Callable[[int], bool] | None = field(default=None, compare=False)
    dagger: bool = False

    def __post_init__(self) -> None:
        if self.controls and not self.control_values:
            object.__setattr__(self, "control_values", (1,) * len(self.controls))
        if len(self.controls) != len(self.control_values):
            raise ValueError("controls and control values differ in length")

    @property
    def parameters(self) -> tuple[Any, ...]:
        if self.parameter is None:
            return ()
        if isinstance(self.parameter, tuple):
            return self.parameter
        return (float(self.parameter),)


def adjoint(ops: tuple[Op, ...] | list[Op]) -> tuple[Op, ...]:
    return tuple(
        Op(
            op.name,
            op.target,
            op.parameter,
            op.controls,
            op.control_values,
            op.predicate,
            not op.dagger,
        )
        for op in reversed(ops)
    )


def _restricted(
    predicate: Callable[[int], bool], control: int, value: int
) -> Callable[[int], bool]:
    def flipped(x: int) -> bool:
        return ((x >> control) & 1) == value and predicate(x & ~(1 << control))

    return flipped


def controlled(ops: tuple[Op, ...], control: int, value: int = 1) -> tuple[Op, ...]:
    """Every operation also conditioned on ``control``; FLIP is restricted."""
    out = []
    for op in ops:
        if op.name == "FLIP":
            assert op.predicate is not None
            out.append(
                Op("FLIP", None, predicate=_restricted(op.predicate, control, value))
            )
        else:
            out.append(
                Op(
                    op.name,
                    op.target,
                    op.parameter,
                    op.controls + (control,),
                    op.control_values + (value,),
                    op.predicate,
                    op.dagger,
                )
            )
    return tuple(out)


def _selected(n: int, op: Op) -> np.ndarray:
    index = np.arange(2**n)
    keep = np.ones(2**n, dtype=bool)
    for control, value in zip(op.controls, op.control_values):
        keep &= ((index >> control) & 1) == value
    return keep


def apply(ops: tuple[Op, ...] | list[Op], psi: np.ndarray, n: int) -> np.ndarray:
    """Apply ``ops`` in order to the columns of ``psi`` (shape 2^n or 2^n x k)."""
    psi = np.array(psi, dtype=complex, copy=True)
    index = np.arange(2**n)
    for op in ops:
        keep = _selected(n, op)
        if op.name == "FLIP":
            assert op.predicate is not None
            flip = np.array([bool(op.predicate(int(x))) for x in index]) & keep
            psi[flip] *= -1
            continue
        if op.name == "GPHASE":
            (phi,) = op.parameters
            psi[keep] *= np.exp((-1j if op.dagger else 1j) * phi)
            continue
        assert op.target is not None
        u = gate_matrix(op.name, op.parameters)
        if op.dagger:
            u = u.conj().T
        bit = 1 << op.target
        low = index[keep & ((index & bit) == 0)]
        high = low | bit
        a, b = psi[low].copy(), psi[high].copy()
        psi[low] = u[0, 0] * a + u[0, 1] * b
        psi[high] = u[1, 0] * a + u[1, 1] * b
    return psi


def unitary(ops: tuple[Op, ...] | list[Op], n: int) -> np.ndarray:
    return apply(ops, np.eye(2**n, dtype=complex), n)


def statevector(ops: tuple[Op, ...] | list[Op], n: int) -> np.ndarray:
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1
    return apply(ops, psi, n)
