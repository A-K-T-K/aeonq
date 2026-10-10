"""Exact reference simulation for small systems; exponential memory by design.

Operators act on a state tensor with one length-two axis per logical qubit, so
each gate costs O(2^n) work rather than a dense O(8^n) matrix product. Dense
matrices are produced by the same code acting on every basis column at once.
"""

from functools import lru_cache
from math import cos, sin

import numpy as np
from numpy.typing import NDArray

from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    ObjectiveReflection,
    Operator,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.predicates import ObjectivePredicate
from aeonq.problem import AEProblem


def _local_matrix(node: Gate) -> NDArray[np.complex128]:
    """Gate matrix on its own targets; local bit i belongs to ``targets[i]``."""
    name = node.name
    if name == "H":
        return np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    if name == "X":
        return np.array([[0, 1], [1, 0]], complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], complex)
    if name == "Z":
        return np.diag([1, -1]).astype(complex)
    if name == "CX":
        small = np.eye(4, dtype=complex)
        small[1, 1] = small[3, 3] = 0
        small[1, 3] = small[3, 1] = 1
        return small
    if not isinstance(node.parameter, (int, float)):
        raise ValueError(f"unbound symbolic parameter: {node.parameter}")
    theta = float(node.parameter)
    c, s = cos(theta / 2), sin(theta / 2)
    if name == "RX":
        return np.array([[c, -1j * s], [-1j * s, c]], complex)
    if name == "RY":
        return np.array([[c, -s], [s, c]], complex)
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


@lru_cache(maxsize=256)
def _objective_signs(predicate: ObjectivePredicate, n_qubits: int) -> NDArray:
    """(-1)^χ(x) as a tensor whose axis j is logical qubit n-1-j."""
    signs = np.where(predicate.truth_table(n_qubits), -1.0, 1.0)
    signs = signs.reshape((2,) * n_qubits)
    signs.setflags(write=False)
    return signs


def _apply(
    node: Operator, psi: NDArray, axes: tuple[int, ...], adjoint: bool
) -> NDArray:
    """Apply ``node`` (or its adjoint) without mutating ``psi``.

    ``axes[q]`` is the tensor axis of the node's logical qubit ``q``; any other
    axes are batch axes (spectator qubits or matrix columns) and are untouched.
    """
    if isinstance(node, Gate):
        k = len(node.targets)
        u = _local_matrix(node)
        if adjoint:
            u = u.conj().T
        # Tensor axis j of a local index is bit k-1-j, i.e. targets[k-1-j].
        target_axes = [axes[node.targets[k - 1 - j]] for j in range(k)]
        result = np.tensordot(
            u.reshape((2,) * (2 * k)), psi, axes=(list(range(k, 2 * k)), target_axes)
        )
        return np.moveaxis(result, list(range(k)), target_axes)
    if isinstance(node, Sequence):
        for child in reversed(node.children) if adjoint else node.children:
            psi = _apply(child, psi, axes, adjoint)
        return psi
    if isinstance(node, StatePreparation):
        return _apply(node.operation, psi, axes, adjoint)
    if isinstance(node, Adjoint):
        return _apply(node.operation, psi, axes, not adjoint)
    if isinstance(node, GlobalPhase):
        return psi * np.exp((-1j if adjoint else 1j) * node.radians)
    if isinstance(node, ZeroReflection):
        index: list[int | slice] = [slice(None)] * psi.ndim
        for q in node.state_qubits:
            index[axes[q]] = 0
        result = psi.copy()
        result[tuple(index)] *= -1
        return result
    if isinstance(node, ObjectiveReflection):
        n = node.n_qubits
        order = sorted(range(n), key=lambda q: axes[q])
        shape = [1] * psi.ndim
        for q in range(n):
            shape[axes[q]] = 2
        signs = np.transpose(
            _objective_signs(node.predicate, n), [n - 1 - q for q in order]
        )
        return psi * signs.reshape(shape)
    if isinstance(node, GroverOperator):
        # Q = -A S0 A† Sχ executes Sχ, A†, S0, A; Q† runs the adjoints reversed.
        steps: list[tuple[Operator, bool]] = [
            (node.objective_reflection, False),
            (node.preparation, True),
            (node.zero_reflection, False),
            (node.preparation, False),
        ]
        if adjoint:
            steps = [(step, not inverse) for step, inverse in reversed(steps)]
        for step, inverse in steps:
            psi = _apply(step, psi, axes, inverse)
        return -psi
    if isinstance(node, GroverPower):
        for _ in range(node.power):
            psi = _apply(node.grover, psi, axes, adjoint)
        return psi
    if isinstance(node, ControlledOperator):
        index = [slice(None)] * psi.ndim
        for control, value in zip(node.controls, node.control_values):
            index[axes[control]] = value
        # Fixing the control axes removes them from the selected sub-tensor.
        control_axes = [axes[c] for c in node.controls]
        sub_axes = tuple(
            axes[t] - sum(a < axes[t] for a in control_axes) for t in node.targets
        )
        result = psi.copy()
        result[tuple(index)] = _apply(
            node.operation, psi[tuple(index)], sub_axes, adjoint
        )
        return result
    if isinstance(node, EmbeddedOperator):
        return _apply(
            node.operation, psi, tuple(axes[t] for t in node.targets), adjoint
        )
    raise TypeError(f"unsupported reference operator: {type(node).__name__}")


def _qubit_axes(n: int) -> tuple[int, ...]:
    # A C-order reshape of a basis index puts its most significant bit first.
    return tuple(n - 1 - q for q in range(n))


def matrix(node: Operator, *, max_qubits: int = 10) -> NDArray[np.complex128]:
    """Evaluate the precise logical unitary, including global phase."""
    n = node.n_qubits
    if n > max_qubits:
        raise ValueError("reference matrix qubit limit exceeded")
    dim = 2**n
    columns = np.eye(dim, dtype=complex).reshape((2,) * n + (dim,))
    return _apply(node, columns, _qubit_axes(n), False).reshape(dim, dim)


def statevector(node: Operator, *, max_qubits: int = 10) -> NDArray[np.complex128]:
    """Exact ``node|0...0>`` without building the operator matrix."""
    n = node.n_qubits
    if n > max_qubits:
        raise ValueError("reference statevector qubit limit exceeded")
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1
    return _apply(node, psi, _qubit_axes(n), False).reshape(2**n)


def _good_probability(state: NDArray[np.complex128], problem: AEProblem) -> float:
    n = problem.state_preparation.n_qubits
    good = problem.objective.truth_table(n)
    return float(np.sum(np.abs(state[good]) ** 2))


def reference_amplitude(problem: AEProblem) -> float:
    """Independently sum good-state probabilities of A|0>."""
    return _good_probability(statevector(problem.state_preparation), problem)


def grover_for(problem: AEProblem) -> GroverOperator:
    """Q = -A S0 A† Sχ over every logical qubit.

    A prepares from |0...0> on all registers (state, work and objective), so
    S0 reflects about the all-zero state of the whole register. Work qubits
    need not be returned clean: a is defined on the full output state.
    """
    n = problem.state_preparation.n_qubits
    return GroverOperator(
        problem.state_preparation,
        ZeroReflection(n, tuple(range(n))),
        ObjectiveReflection(n, problem.objective),
        n,
    )


def ideal_success_probability(problem: AEProblem, power: int) -> float:
    if power < 0:
        raise ValueError("power must be nonnegative")
    q = grover_for(problem)
    circuit = Sequence(
        (problem.state_preparation, GroverPower(q, power, q.n_qubits)), q.n_qubits
    )
    return _good_probability(statevector(circuit), problem)
