from math import asin, pi, sin, sqrt

import numpy as np
import pytest

from aeonq import AEProblem
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    Gate,
    GlobalPhase,
    GroverPower,
    ObjectivePredicate,
    Parameter,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.ir.serialization import dumps, loads
from aeonq.reference import (
    grover_for,
    ideal_success_probability,
    matrix,
    reference_amplitude,
)
from aeonq.utils.numerics import (
    controlled_unitary_equivalent,
    global_phase_aligned_fidelity,
    unitary_equivalent_up_to_phase,
)


def bernoulli_problem(a: float) -> AEProblem:
    angle = 2 * asin(sqrt(a))
    prep = StatePreparation(Gate("RY", (0,), 1, angle), 1)
    return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))


@pytest.mark.parametrize("a", [0.0, 1e-12, 1e-6, 0.01, 0.25, 0.5, 0.9, 1 - 1e-9, 1.0])
def test_amplitude_and_grover_law(a: float) -> None:
    problem = bernoulli_problem(a)
    assert reference_amplitude(problem) == pytest.approx(a, abs=1e-12)
    for k in range(5):
        expected = sin((2 * k + 1) * asin(sqrt(a))) ** 2
        assert ideal_success_probability(problem, k) == pytest.approx(
            expected, abs=2e-12
        )


def test_grover_exact_sign_and_adjoint() -> None:
    problem = bernoulli_problem(0.23)
    q = grover_for(problem)
    a = matrix(problem.state_preparation)
    s0 = matrix(q.zero_reflection)
    sx = matrix(q.objective_reflection)
    assert np.allclose(matrix(q), -a @ s0 @ a.conj().T @ sx)
    assert np.allclose(matrix(Adjoint(q, 1)) @ matrix(q), np.eye(2))
    assert np.allclose(
        matrix(GroverPower(q, 3, 1)), np.linalg.matrix_power(matrix(q), 3)
    )


def test_sequence_adjoint_and_unitarity() -> None:
    sequence = Sequence(
        (Gate("H", (0,), 2), Gate("CX", (0, 1), 2), Gate("RZ", (1,), 2, 0.37)), 2
    )
    u = matrix(sequence)
    assert np.allclose(matrix(Adjoint(sequence, 2)), u.conj().T)
    assert np.allclose(u.conj().T @ u, np.eye(4))
    assert np.allclose(matrix(Adjoint(Adjoint(sequence, 2), 2)), u)


def test_controlled_phase_is_not_global() -> None:
    phase = GlobalPhase(pi, 1)
    controlled = ControlledOperator(phase, (1,), (1,), (0,), 2)
    target_identity = matrix(GlobalPhase(0.0, 1))
    assert unitary_equivalent_up_to_phase(matrix(phase), target_identity)
    assert not controlled_unitary_equivalent(matrix(controlled), np.eye(4))
    assert np.allclose(np.diag(matrix(controlled)), [1, 1, -1, -1])


def test_predicates_and_asymmetric_register_mapping() -> None:
    pred = ObjectivePredicate(
        "and",
        children=(
            ObjectivePredicate("ge", qubits=(2, 0), value=2),
            ObjectivePredicate(
                "not", children=(ObjectivePredicate("bit", qubits=(1,)),)
            ),
        ),
    )
    assert pred.evaluate(1, 3)
    assert not pred.evaluate(4, 3)
    mapping = RegisterMap((2, 0, 1), (1, 2, 0))
    for index in range(8):
        assert mapping.logical_index(mapping.physical_index(index)) == index
    assert mapping.normalize_bitstring("101") == (0, 1, 1)


def test_global_phase_fidelity_and_invalid_inputs() -> None:
    psi = np.array([1, 2j]) / sqrt(5)
    assert global_phase_aligned_fidelity(psi, -psi) == pytest.approx(1)
    with pytest.raises(ValueError):
        Gate("RY", (0,), 1, float("nan"))
    with pytest.raises(ValueError):
        Gate("CX", (0, 0), 2)
    with pytest.raises(ValueError):
        ObjectivePredicate("equal", qubits=(0,), value=2).validate(1)
    with pytest.raises(ValueError):
        ObjectivePredicate("python_function").validate(1)
    with pytest.raises(ValueError):
        matrix(Gate("RX", (0,), 1, Parameter("theta")))
    with pytest.raises(ValueError):
        ControlledOperator(Gate("X", (0,), 1), (0,), (1,), (0,), 2)


def test_ir_serialization_round_trip_and_schema_rejection() -> None:
    problem = bernoulli_problem(0.3)
    assert loads(dumps(problem)) == problem
    node = GroverPower(grover_for(problem), 5, 1)
    assert loads(dumps(node)) == node
    assert np.allclose(matrix(loads(dumps(node))), matrix(node))
    with pytest.raises(ValueError, match="schema"):
        loads('{"schema_version":2,"node":{}}')
    with pytest.raises(ValueError, match="unsupported IR kind"):
        loads('{"schema_version":1,"node":{"__type__":"PythonCode"}}')
