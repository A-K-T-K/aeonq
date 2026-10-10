"""The tensor-contraction simulator against an independent dense construction."""

from functools import reduce

import numpy as np
import pytest

from aeonq.benchmarks.problems import independent_failures, uniform_marked
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    Gate,
    GlobalPhase,
    GroverPower,
    ObjectivePredicate,
    Sequence,
)
from aeonq.ir.nodes import EmbeddedOperator
from aeonq.reference import _local_matrix, grover_for, matrix, statevector


def _dense_embed(u: np.ndarray, targets: tuple[int, ...], n: int) -> np.ndarray:
    """Place local u (bit i = targets[i]) by explicit basis enumeration."""
    dim = 2**n
    result = np.zeros((dim, dim), dtype=complex)
    mask = sum(1 << q for q in targets)
    for column in range(dim):
        local_in = sum(((column >> q) & 1) << i for i, q in enumerate(targets))
        for local_out in range(2 ** len(targets)):
            row = (column & ~mask) | sum(
                ((local_out >> i) & 1) << q for i, q in enumerate(targets)
            )
            result[row, column] = u[local_out, local_in]
    return result


def _dense_controlled(
    u: np.ndarray, controls, values, targets: tuple[int, ...], n: int
) -> np.ndarray:
    embedded = _dense_embed(u, targets, n)
    result = np.eye(2**n, dtype=complex)
    for column in range(2**n):
        if all(((column >> c) & 1) == v for c, v in zip(controls, values)):
            result[:, column] = embedded[:, column]
    return result


LOCAL = Sequence(
    (
        Gate("RY", (0,), 3, 0.7),
        Gate("CX", (2, 0), 3),
        Gate("RZ", (1,), 3, -1.3),
        Gate("H", (2,), 3),
        GlobalPhase(0.4, 3),
    ),
    3,
)


def test_sequence_equals_product_of_children():
    for child in LOCAL.children:
        if isinstance(child, Gate):
            expected_gate = _dense_embed(_local_matrix(child), child.targets, 3)
            assert np.allclose(matrix(child), expected_gate)
    expected = reduce(
        lambda acc, child: matrix(child) @ acc, LOCAL.children, np.eye(8, dtype=complex)
    )
    assert np.allclose(matrix(LOCAL), expected)
    assert np.allclose(matrix(Adjoint(LOCAL, 3)), expected.conj().T)


@pytest.mark.parametrize("targets", [(0, 1, 2), (4, 1, 3), (2, 4, 0)])
def test_embedded_operator_matches_basis_enumeration(targets):
    node = EmbeddedOperator(LOCAL, targets, 5)
    assert np.allclose(matrix(node), _dense_embed(matrix(LOCAL), targets, 5))


@pytest.mark.parametrize(
    "controls,values,targets",
    [((4,), (1,), (0, 1, 2)), ((0, 3), (0, 1), (4, 2, 1)), ((2,), (0,), (3, 0, 4))],
)
def test_controlled_operator_matches_block_construction(controls, values, targets):
    node = ControlledOperator(LOCAL, controls, values, targets, 5)
    expected = _dense_controlled(matrix(LOCAL), controls, values, targets, 5)
    assert np.allclose(matrix(node), expected)
    assert np.allclose(matrix(Adjoint(node, 5)), expected.conj().T)


@pytest.mark.parametrize(
    "case",
    [uniform_marked(3, (1, 6)), independent_failures((0.1, 0.35, 0.2))],
    ids=lambda case: case.case_id,
)
def test_grover_powers_statevector_and_adjoint(case):
    q = grover_for(case.problem)
    n = q.n_qubits
    dense_q = matrix(q)
    for power in range(4):
        node = Sequence((case.problem.state_preparation, GroverPower(q, power, n)), n)
        assert np.allclose(statevector(node), matrix(node)[:, 0])
        assert np.allclose(
            matrix(GroverPower(q, power, n)), np.linalg.matrix_power(dense_q, power)
        )
    assert np.allclose(matrix(Adjoint(q, n)), dense_q.conj().T)


def test_vectorized_predicate_matches_scalar_evaluation():
    predicate = ObjectivePredicate(
        "or",
        children=(
            ObjectivePredicate("marked", marked=(3, 17, 30)),
            ObjectivePredicate(
                "and",
                children=(
                    ObjectivePredicate("ge", qubits=(4, 0, 2), value=5),
                    ObjectivePredicate(
                        "not", children=(ObjectivePredicate("bit", qubits=(1,)),)
                    ),
                ),
            ),
        ),
    )
    table = predicate.truth_table(5)
    assert table.tolist() == [predicate.evaluate(x, 5) for x in range(32)]
    with pytest.raises(ValueError, match="out of range"):
        predicate.evaluate_many(np.array([0, 32]), 5)
