import numpy as np
import pytest

from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    Gate,
    GlobalPhase,
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
def adapter() -> PennyLaneAdapter:
    return PennyLaneAdapter()


def test_pennylane_differential_semantics(adapter: PennyLaneAdapter) -> None:
    n = 2
    prep = StatePreparation(
        Sequence((Gate("RY", (0,), n, 0.43), Gate("H", (1,), n)), n), n
    )
    q = GroverOperator(
        prep,
        ZeroReflection(n, (0, 1)),
        ObjectiveReflection(n, ObjectivePredicate("marked", marked=(1, 3))),
        n,
    )
    nodes = (
        prep,
        Adjoint(prep, n),
        q.zero_reflection,
        q.objective_reflection,
        q,
        GroverPower(q, 3, n),
    )
    qiskit = QiskitAdapter()
    for node in nodes:
        assert np.allclose(adapter.unitary(node), matrix(node), atol=1e-10)
        assert np.allclose(adapter.unitary(node), qiskit.unitary(node), atol=1e-10)


def test_pennylane_controlled_phase_and_grover(adapter: PennyLaneAdapter) -> None:
    phase = ControlledOperator(GlobalPhase(np.pi, 1), (1,), (1,), (0,), 2)
    assert np.allclose(adapter.unitary(phase), matrix(phase), atol=1e-10)
    prep = StatePreparation(Gate("RY", (0,), 1, 0.36), 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    controlled = ControlledOperator(q, (1,), (1,), (0,), 2)
    assert np.allclose(adapter.unitary(controlled), matrix(controlled), atol=1e-10)


def test_pennylane_finite_shots_and_endianness(adapter: PennyLaneAdapter) -> None:
    prep = Sequence((Gate("X", (0,), 3), Gate("X", (2,), 3)), 3)
    raw = adapter.execute(adapter.compile(adapter.lower(prep)), 17, seed=33)
    mapping = RegisterMap((2, 0, 1), (2, 0, 1))
    assert adapter.normalize_result(raw, mapping).counts == {(1, 1, 0): 17}
