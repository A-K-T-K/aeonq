import numpy as np
import pytest

from aeonq.adapters.qsharp import QSharpAdapter
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
def adapter() -> QSharpAdapter:
    return QSharpAdapter()


def test_qsharp_operator_conformance(adapter: QSharpAdapter) -> None:
    n = 2
    prep = StatePreparation(
        Sequence((Gate("RY", (0,), n, 0.37), Gate("H", (1,), n)), n), n
    )
    q = GroverOperator(
        prep,
        ZeroReflection(n, (0, 1)),
        ObjectiveReflection(n, ObjectivePredicate("marked", marked=(1, 3))),
        n,
    )
    for node in (
        prep,
        Adjoint(prep, n),
        q.zero_reflection,
        q.objective_reflection,
        q,
        GroverPower(q, 2, n),
    ):
        assert np.allclose(adapter.unitary(node), matrix(node), atol=1e-10)


def test_qsharp_controlled_phase_and_grover(adapter: QSharpAdapter) -> None:
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


def test_qsharp_finite_shots_and_endianness(adapter: QSharpAdapter) -> None:
    prep = Sequence((Gate("X", (0,), 3), Gate("X", (2,), 3)), 3)
    raw = adapter.execute(adapter.compile(adapter.lower(prep)), 11, seed=33)
    mapping = RegisterMap((2, 0, 1), (2, 0, 1))
    assert adapter.normalize_result(raw, mapping).counts == {(1, 1, 0): 11}


def test_qsharp_consecutive_seeds_give_independent_unbiased_samples(
    adapter: QSharpAdapter,
) -> None:
    # QDK seeds shot i with seed + i, so consecutive estimator seeds must be
    # spread out or consecutive rounds would share their draws.
    from aeonq.adapters.qsharp import qdk_seed

    assert qdk_seed(None) is None
    assert qdk_seed(7) == qdk_seed(7)
    spread = [qdk_seed(s) for s in range(1000, 1100)]
    assert all(0 <= value < 2**63 for value in spread)
    ordered = sorted(spread)
    assert min(b - a for a, b in zip(ordered, ordered[1:])) > 10**6
    prep = Sequence((Gate("RY", (0,), 1, 2 * np.arcsin(np.sqrt(0.25))),), 1)
    program = adapter.compile(adapter.lower(prep))
    shots, runs = 256, 120
    successes = [
        adapter.execute(program, shots, seed=1_300_001 + i).get("1", 0)
        for i in range(runs)
    ]
    mean = np.mean(successes) / shots
    assert abs(mean - 0.25) < 5 * np.sqrt(0.25 * 0.75 / (shots * runs))
    ratio = np.var(successes, ddof=1) / (shots * 0.25 * 0.75)
    assert 0.6 < ratio < 1.6


def test_qsharp_evaluates_each_source_once_per_session(
    adapter: QSharpAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Re-evaluating identical sources would grow the interpreter session with
    # every trial, so each distinct source is evaluated once.
    from aeonq.adapters import qsharp as module

    calls = []
    real_eval = module.qsharp.eval
    monkeypatch.setattr(
        module.qsharp, "eval", lambda source: calls.append(source) or real_eval(source)
    )
    module.clear_evaluation_cache()
    program = adapter.lower(Sequence((Gate("X", (0,), 2), Gate("H", (1,), 2)), 2))
    for seed in range(3):
        raw = adapter.execute(adapter.compile(program), 8, seed=seed)
        assert sum(raw.values()) == 8
    assert len(calls) == 1
    module.clear_evaluation_cache()
    adapter.compile(program)
    assert len(calls) == 2
