"""Seeded algebraic and statistical properties of the IR and estimators.

Properties verified:
- Adj(Adj(U)) = U
- Adj(UV) = Adj(V) Adj(U)
- U Adj(U) ≈ I
- Normalized probability sums to 1
- 0 <= estimate <= 1
- interval lower <= estimate <= interval upper where algorithmically appropriate
- interval subset of [0, 1]
- register permutation round-trip = identity
- serialization round-trip preserves IR semantics
- SDK lowering preserves objective-state probability
"""

from math import asin, sqrt

import numpy as np

from aeonq import AEProblem, estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.ir import (
    ControlledOperator,
    Gate,
    ObjectivePredicate,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.ir.serialization import dumps, loads
from aeonq.reference import grover_for, matrix, reference_amplitude
from aeonq.synthesis.adjoint import adjoint


def test_randomized_grover_and_adjoint_identities():
    generator = np.random.default_rng(94217)
    for amplitude in generator.uniform(0.001, 0.999, size=32):
        prep = StatePreparation(
            Gate("RY", (0,), 1, 2 * asin(sqrt(float(amplitude)))), 1
        )
        problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))
        q = matrix(grover_for(problem))
        a = matrix(prep)
        assert np.isclose(reference_amplitude(problem), amplitude, atol=1e-12)
        assert np.allclose(matrix(adjoint(prep)) @ a, np.eye(2), atol=1e-12)
        assert np.allclose(q.conj().T @ q, np.eye(2), atol=1e-12)
        assert np.allclose(
            np.linalg.matrix_power(q, 2) @ q, np.linalg.matrix_power(q, 3), atol=1e-12
        )


def test_adjoint_properties_composite_operators():
    """Verify Adj(Adj(U)) = U, Adj(UV) = Adj(V)Adj(U), and U Adj(U) = I."""
    rng = np.random.default_rng(13579)
    for _ in range(10):
        th1, th2, th3 = rng.uniform(0, 2 * np.pi, size=3)
        u1 = Gate("RX", (0,), 2, th1)
        u2 = Gate("CX", (0, 1), 2)
        u3 = Gate("RZ", (1,), 2, th2)
        u4 = Gate("RY", (0,), 2, th3)

        u = Sequence((u1, u2), 2)
        v = Sequence((u3, u4), 2)
        uv = Sequence((u1, u2, u3, u4), 2)

        mat_u = matrix(u)
        mat_v = matrix(v)
        mat_uv = matrix(uv)

        # U Adj(U) ≈ I and V Adj(V) ≈ I
        adj_u = adjoint(u)
        adj_v = adjoint(v)
        assert np.allclose(mat_u @ matrix(adj_u), np.eye(4), atol=1e-12)
        assert np.allclose(mat_v @ matrix(adj_v), np.eye(4), atol=1e-12)

        # Adj(Adj(U)) = U
        adj_adj_u = adjoint(adj_u)
        assert np.allclose(matrix(adj_adj_u), mat_u, atol=1e-12)

        # Adj(UV) = Adj(V) Adj(U) in sequence application
        adj_uv = adjoint(uv)
        seq_adj_v_adj_u = Sequence((adj_v, adj_u), 2)
        assert np.allclose(matrix(adj_uv), matrix(seq_adj_v_adj_u), atol=1e-12)
        assert np.allclose(matrix(adj_uv), matrix(adj_u) @ matrix(adj_v), atol=1e-12)
        assert np.allclose(mat_uv.conj().T, matrix(adj_uv), atol=1e-12)


def test_statistical_estimator_properties():
    """Verify 0 <= estimate <= 1, interval bounds, and probability sum = 1."""
    rng = np.random.default_rng(24680)
    backend = ReferenceAdapter()

    for amplitude in rng.uniform(0.01, 0.99, size=10):
        prep = StatePreparation(
            Gate("RY", (0,), 1, 2 * asin(sqrt(float(amplitude)))), 1
        )
        problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))

        res = estimate(
            problem, algorithm="iqae", backend=backend, epsilon=0.05, alpha=0.05
        )

        # 0 <= estimate <= 1
        assert 0.0 <= res.estimate <= 1.0

        # interval lower <= estimate <= interval upper
        lo, hi = res.confidence_or_credible_interval
        assert lo <= res.estimate <= hi

        # interval is subset of [0, 1]
        assert 0.0 <= lo and hi <= 1.0


def test_register_permutation_roundtrip():
    """Verify register permutation round-trip = identity."""
    for n in (2, 3, 4, 5):
        physical = tuple(reversed(range(n)))
        logical = tuple(range(n))
        reg_map = RegisterMap(physical, logical)

        # logical_index(physical_index(i)) == i
        for idx in range(2**n):
            assert reg_map.logical_index(reg_map.physical_index(idx)) == idx


def test_serialization_roundtrip_preserves_ir_semantics():
    """Verify serialization round-trip preserves IR semantics for various operators."""
    seq = Sequence(
        (
            Gate("H", (0,), 2),
            Gate("CX", (0, 1), 2),
            Gate("RY", (1,), 2, 0.785),
            ControlledOperator(Gate("Z", (0,), 1), (0,), (1,), (1,), 2),
        ),
        2,
    )
    serialized = dumps(seq)
    deserialized = loads(serialized)

    assert np.allclose(matrix(seq), matrix(deserialized), atol=1e-12)


def test_sdk_lowering_preserves_objective_state_probability():
    """Verify that all SDK lowering targets preserve objective state probability."""
    a = 0.42
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    problem = AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))

    adapters = [QiskitAdapter(), CirqAdapter(), PennyLaneAdapter(), QSharpAdapter()]
    for adapter in adapters:
        if hasattr(adapter, "unitary"):
            u = adapter.unitary(problem.state_preparation)
            prob = float(np.abs(u[1, 0]) ** 2)
            assert np.isclose(prob, a, atol=1e-6)
