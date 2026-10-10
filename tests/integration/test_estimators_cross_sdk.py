from math import asin, sqrt

import pytest

from aeonq import AEProblem
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.algorithms.mlqae import MLQAE
from aeonq.algorithms.canonical_qae import CanonicalQAE
from aeonq.algorithms.faster_ae import FasterAE, attenuated_problem
from aeonq.reference import grover_for, matrix
import numpy as np
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_iqae_cross_sdk_local(backend: type) -> None:
    a = 0.25
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    result = IQAE(
        epsilon=0.2, shots_per_round=64, max_rounds=2, max_power=2, seed=11
    ).run(problem, backend())
    lo, hi = result.confidence_or_credible_interval
    assert lo <= a <= hi
    assert result.total_shots > 0


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_mlqae_cross_sdk_local(backend: type) -> None:
    a = 0.25
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    result = MLQAE(epsilon=0.2, powers=(0, 1), shots_per_power=64, seed=11).run(
        problem, backend()
    )
    lo, hi = result.confidence_or_credible_interval
    assert lo <= a <= hi
    assert result.total_shots == 128


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_canonical_cross_sdk_local(backend: type) -> None:
    a = 0.5
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    result = CanonicalQAE(evaluation_qubits=2, shots=16, seed=11).run(
        problem, backend()
    )
    assert result.estimate == pytest.approx(a, abs=1e-10)


@pytest.mark.parametrize(
    "backend", [QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter]
)
def test_faster_ae_attenuation_semantics_cross_sdk(backend: type) -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 0.91), 1)
    original = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    adapted = attenuated_problem(original)
    for node in (adapted.state_preparation, grover_for(adapted)):
        assert np.allclose(backend().unitary(node), matrix(node), atol=1e-10)


def test_faster_ae_qiskit_local_execution() -> None:
    a = 0.25
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    original = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
    result = FasterAE(epsilon=0.2, max_iterations=1, seed=13).run(
        original, QiskitAdapter()
    )
    assert (
        result.confidence_or_credible_interval[0]
        <= a
        <= result.confidence_or_credible_interval[1]
    )
    assert result.total_shots == result.diagnostics["required_shots"][0]
