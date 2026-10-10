from math import asin, sqrt
import json

import pytest

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation
from aeonq.runtime.checkpoint import IncompatibleCheckpoint


def problem(a: float) -> AEProblem:
    return AEProblem(
        StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )


class FailOnceAdapter(QiskitAdapter):
    def __init__(self, fail_at: int | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.fail_at = fail_at

    def execute(self, compiled_program, shots, *, seed=None):
        self.calls += 1
        if self.calls == self.fail_at:
            self.fail_at = None
            raise ConnectionError("injected dropped connection")
        return super().execute(compiled_program, shots, seed=seed)


def test_resume_after_injected_failure(tmp_path) -> None:
    path = tmp_path / "iqae.json"
    solver = IQAE(
        epsilon=0.001,
        shots_per_round=128,
        max_rounds=3,
        max_power=3,
        seed=18,
        checkpoint=path,
    )
    original = problem(0.13)
    backend = FailOnceAdapter(fail_at=2)
    with pytest.raises(ConnectionError, match="injected"):
        solver.run(original, backend)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["state"]["scheduler_state"]["next_iteration"] == 1
    assert saved["state"]["completed_job_ids"] == []
    resumed = solver.run(original, backend)
    uninterrupted = IQAE(
        epsilon=0.001, shots_per_round=128, max_rounds=3, max_power=3, seed=18
    ).run(original, FailOnceAdapter())
    assert resumed.estimate == uninterrupted.estimate
    assert (
        resumed.confidence_or_credible_interval
        == uninterrupted.confidence_or_credible_interval
    )
    assert resumed.total_shots == uninterrupted.total_shots
    assert resumed.powers_used == uninterrupted.powers_used


def test_checkpoint_identity_and_schema_rejection(tmp_path) -> None:
    path = tmp_path / "iqae.json"
    solver = IQAE(
        epsilon=0.1,
        shots_per_round=64,
        max_rounds=2,
        max_power=2,
        seed=3,
        checkpoint=path,
    )
    solver.run(problem(0.2), QiskitAdapter())
    with pytest.raises(IncompatibleCheckpoint, match="differs"):
        solver.run(problem(0.3), QiskitAdapter())
    with pytest.raises(IncompatibleCheckpoint, match="differs"):
        IQAE(
            epsilon=0.2,
            shots_per_round=64,
            max_rounds=2,
            max_power=2,
            seed=3,
            checkpoint=path,
        ).run(problem(0.2), QiskitAdapter())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(IncompatibleCheckpoint, match="schema"):
        solver.run(problem(0.2), QiskitAdapter())


def test_malformed_checkpoint_state_is_rejected(tmp_path) -> None:
    path = tmp_path / "iqae.json"
    solver = IQAE(
        epsilon=0.1,
        shots_per_round=64,
        max_rounds=2,
        max_power=2,
        seed=3,
        checkpoint=path,
    )
    solver.run(problem(0.2), QiskitAdapter())
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["state"]["theta_interval"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(IncompatibleCheckpoint, match="malformed"):
        solver.run(problem(0.2), QiskitAdapter())
