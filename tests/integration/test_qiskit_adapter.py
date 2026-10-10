import numpy as np
import pytest

from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    Gate,
    GroverPower,
    ObjectivePredicate,
    ObjectiveReflection,
    RegisterMap,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.nodes import GroverOperator
from aeonq.reference import matrix


@pytest.fixture(scope="module")
def adapter() -> QiskitAdapter:
    return QiskitAdapter()


def test_qiskit_operator_conformance(adapter: QiskitAdapter) -> None:
    n = 2
    a = StatePreparation(
        Sequence((Gate("RY", (0,), n, 0.81), Gate("H", (1,), n)), n), n
    )
    q = GroverOperator(
        a,
        ZeroReflection(n, (0, 1)),
        ObjectiveReflection(n, ObjectivePredicate("marked", marked=(1, 3))),
        n,
    )
    nodes = (
        a,
        Adjoint(a, n),
        q.zero_reflection,
        q.objective_reflection,
        q,
        GroverPower(q, 0, n),
        GroverPower(q, 3, n),
    )
    for node in nodes:
        assert np.allclose(adapter.unitary(node), matrix(node), atol=1e-10)


def test_qiskit_controlled_phase_and_counts(adapter: QiskitAdapter) -> None:
    from aeonq.ir import GlobalPhase

    controlled = ControlledOperator(GlobalPhase(np.pi, 1), (1,), (1,), (0,), 2)
    assert np.allclose(adapter.unitary(controlled), matrix(controlled), atol=1e-10)
    prep = Sequence((Gate("X", (0,), 3), Gate("X", (2,), 3)), 3)
    compiled = adapter.compile(adapter.lower(prep))
    raw = adapter.execute(compiled, 32, seed=10)
    normalized = adapter.normalize_result(raw, RegisterMap((0, 1, 2), (0, 1, 2)))
    assert normalized.counts == {(1, 0, 1): 32}
    assert normalized.shots == 32


def test_qiskit_asymmetric_mapping(adapter: QiskitAdapter) -> None:
    prep = Sequence((Gate("X", (0,), 3), Gate("X", (2,), 3)), 3)
    raw = adapter.execute(adapter.compile(adapter.lower(prep)), 7, seed=5)
    mapping = RegisterMap((2, 0, 1), (2, 0, 1))
    assert adapter.normalize_result(raw, mapping).counts == {(1, 1, 0): 7}


@pytest.mark.parametrize("a", [0.0, 1e-9, 0.1, 0.5, 0.9, 1 - 1e-9, 1.0])
@pytest.mark.parametrize("power", [0, 1, 2, 5])
def test_qiskit_probability_law(adapter: QiskitAdapter, a: float, power: int) -> None:
    from math import asin, sin, sqrt

    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    circuit = adapter.lower(Sequence((prep, GroverPower(q, power, 1)), 1))
    actual = (
        abs(adapter.statevector(Sequence((prep, GroverPower(q, power, 1)), 1))[1]) ** 2
    )
    assert actual == pytest.approx(sin((2 * power + 1) * asin(sqrt(a))) ** 2, abs=2e-12)
    assert circuit.num_qubits == 1


def test_qiskit_controlled_grover_exact(adapter: QiskitAdapter) -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 0.44), 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    controlled = ControlledOperator(q, (1,), (1,), (0,), 2)
    assert np.allclose(adapter.unitary(controlled), matrix(controlled), atol=1e-10)
