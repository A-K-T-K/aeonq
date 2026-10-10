import pytest

from aeonq import AEProblem, estimate
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation


def test_top_level_estimate_and_unsupported_algorithm() -> None:
    problem = AEProblem(
        StatePreparation(Gate("RY", (0,), 1, 0.8), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    result = estimate(
        problem,
        algorithm="iqae",
        backend=QiskitAdapter(),
        epsilon=0.2,
        shots_per_round=64,
        max_rounds=3,
        seed=4,
    )
    assert 0 <= result.estimate <= 1
    with pytest.raises(ValueError, match="unsupported algorithm"):
        estimate(problem, algorithm="unknown", backend=QiskitAdapter())
