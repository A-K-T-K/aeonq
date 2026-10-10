import numpy as np
import pytest

from aeonq.ir import (
    ControlledOperator,
    Gate,
    ObjectivePredicate,
    ObjectiveReflection,
    Parameter,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.nodes import GroverOperator
from aeonq.ir.nodes import Adjoint, EmbeddedOperator, GlobalPhase, GroverPower
from aeonq.reference import matrix
from aeonq.synthesis import adjoint, bind_parameters, synthesize_grover
from aeonq.synthesis import synthesize_objective_reflection, synthesize_zero_reflection
from aeonq.utils.numerics import unitary_equivalent_up_to_phase


def test_symbolic_adjoint_and_order() -> None:
    u = Sequence((Gate("H", (0,), 1), Gate("RY", (0,), 1, Parameter("theta"))), 1)
    symbolic = adjoint(u)
    assert symbolic.children[0].parameter == Parameter("theta", -1)
    assert adjoint(symbolic) == u
    bound = bind_parameters(u, {"theta": 0.71})
    inverse = bind_parameters(symbolic, {"theta": 0.71})
    assert np.allclose(matrix(inverse), matrix(bound).conj().T)
    with pytest.raises(ValueError, match="missing parameter"):
        bind_parameters(u, {})


def test_reflections_and_exact_grover_synthesis() -> None:
    n = 2
    a = StatePreparation(Sequence((Gate("RY", (0,), n, 0.7), Gate("H", (1,), n)), n), n)
    s0 = ZeroReflection(n, (0, 1))
    sx = ObjectiveReflection(n, ObjectivePredicate("marked", marked=(1, 3)))
    q = GroverOperator(a, s0, sx, n)
    generated = matrix(synthesize_grover(q))
    reference = -matrix(a) @ matrix(s0) @ matrix(a).conj().T @ matrix(sx)
    assert np.allclose(generated, reference)
    assert unitary_equivalent_up_to_phase(generated, matrix(q))
    assert np.allclose(matrix(adjoint(synthesize_grover(q))), generated.conj().T)
    assert np.allclose(np.diag(matrix(s0)), [-1, 1, 1, 1])
    assert np.allclose(np.diag(matrix(sx)), [1, -1, 1, -1])
    assert np.allclose(matrix(synthesize_zero_reflection(s0)), matrix(s0))
    assert np.allclose(matrix(synthesize_objective_reflection(sx)), matrix(sx))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reflection_synthesis_multiple_widths(n: int) -> None:
    s0 = ZeroReflection(n, tuple(range(n)))
    assert np.allclose(matrix(synthesize_zero_reflection(s0)), matrix(s0))
    predicate = ObjectivePredicate("ge", qubits=tuple(range(n)), value=2 ** (n - 1))
    sx = ObjectiveReflection(n, predicate)
    assert np.allclose(matrix(synthesize_objective_reflection(sx)), matrix(sx))


def test_controlled_grover_requires_exact_phase() -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 0.63), 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    exact = ControlledOperator(synthesize_grover(q), (1,), (1,), (0,), 2)
    without_minus = ControlledOperator(
        Sequence(synthesize_grover(q).children[:-1], 1), (1,), (1,), (0,), 2
    )
    assert np.allclose(
        matrix(exact), matrix(ControlledOperator(q, (1,), (1,), (0,), 2))
    )
    assert not np.allclose(matrix(exact), matrix(without_minus))
    assert unitary_equivalent_up_to_phase(matrix(q), -matrix(q))


def test_adjoint_and_binding_cover_nested_semantic_operators() -> None:
    angle = Parameter("angle")
    local = Gate("RY", (0,), 1, angle)
    controlled = ControlledOperator(local, (1,), (1,), (0,), 2)
    embedded = EmbeddedOperator(local, (1,), 2)
    prep = StatePreparation(local, 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    nodes = (
        local,
        prep,
        Adjoint(local, 1),
        controlled,
        embedded,
        GlobalPhase(0.13, 1),
        q.zero_reflection,
        q.objective_reflection,
        q,
        GroverPower(q, 2, 1),
    )
    for node in nodes:
        bound = bind_parameters(node, {"angle": 0.41})
        inverse = adjoint(bound)
        expected = matrix(bound).conj().T
        assert np.allclose(matrix(inverse), expected, atol=1e-12)
    assert isinstance(bind_parameters(q, {"angle": 0.41}), GroverOperator)
    assert isinstance(
        bind_parameters(GroverPower(q, 2, 1), {"angle": 0.41}), GroverPower
    )
