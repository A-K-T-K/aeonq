"""Metamorphic tests of semantic invariants.

Metamorphic relations test semantic invariants under program transformations:
1. Insert U U† into state preparation; true amplitude remains unchanged.
2. Permute physical qubits while updating logical mapping; AE result is invariant.
3. Add an unused clean ancilla; AE semantics remain unchanged.
4. Reexpress equivalent rotations; results remain equivalent.
5. Translate to another supported SDK; logical success probability is unchanged.
6. Serialize -> deserialize; semantic result remains unchanged.
"""

from math import asin, sin, sqrt
import numpy as np

from aeonq import AEProblem, estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.ir import (
    Adjoint,
    EmbeddedOperator,
    Gate,
    ObjectivePredicate,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.ir.serialization import dumps, loads
from aeonq.reference import grover_for, matrix, reference_amplitude


def _bernoulli(a: float) -> AEProblem:
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))


def test_metamorphic_insert_identity_pair() -> None:
    """MR 1: Inserting U U† into state preparation leaves amplitude unchanged."""
    a = 0.35
    base = _bernoulli(a)
    u = Gate("H", (0,), 1)
    u_adj = Adjoint(u, 1)
    transformed_prep = StatePreparation(
        Sequence((base.state_preparation.operation, u, u_adj), 1), 1
    )
    transformed_problem = AEProblem(
        transformed_prep, base.objective, base.state_register
    )

    # Unitary matrices and true amplitudes must agree exactly
    assert np.allclose(
        matrix(base.state_preparation), matrix(transformed_prep), atol=1e-12
    )
    assert np.isclose(
        reference_amplitude(base), reference_amplitude(transformed_problem), atol=1e-12
    )

    # Execution on reference backend must yield identical estimates
    res1 = estimate(base, algorithm="iqae", backend=ReferenceAdapter(), seed=42)
    res2 = estimate(
        transformed_problem, algorithm="iqae", backend=ReferenceAdapter(), seed=42
    )
    assert np.isclose(res1.estimate, res2.estimate, atol=1e-12)


def test_metamorphic_qubit_permutation() -> None:
    """MR 2: Permuting physical qubits while updating logical mapping leaves AE invariant."""
    # 2-qubit state preparation where qubit 0 is the objective
    a = 0.4
    theta = 2 * asin(sqrt(a))
    ops = (Gate("RY", (0,), 2, theta), Gate("CX", (0, 1), 2))
    p1 = AEProblem(
        StatePreparation(Sequence(ops, 2), 2),
        ObjectivePredicate("bit", qubits=(0,)),
        (0, 1),
    )

    # Swap physical qubits (0 <-> 1)
    permuted_ops = (Gate("RY", (1,), 2, theta), Gate("CX", (1, 0), 2))
    p2 = AEProblem(
        StatePreparation(Sequence(permuted_ops, 2), 2),
        ObjectivePredicate("bit", qubits=(1,)),
        (0, 1),
    )

    assert np.isclose(reference_amplitude(p1), reference_amplitude(p2), atol=1e-12)

    reg_map = RegisterMap((1, 0), (0, 1))
    assert reg_map.logical_index(reg_map.physical_index(0)) == 0
    assert reg_map.logical_index(reg_map.physical_index(1)) == 1

    res1 = estimate(p1, algorithm="iqae", backend=ReferenceAdapter(), seed=123)
    res2 = estimate(p2, algorithm="iqae", backend=ReferenceAdapter(), seed=123)
    assert np.isclose(res1.estimate, res2.estimate, atol=1e-12)


def test_metamorphic_unused_clean_ancilla() -> None:
    """MR 3: Adding an unused clean ancilla preserves AE semantics."""
    a = 0.65
    base = _bernoulli(a)

    # Embed 1-qubit problem into 2-qubit space with qubit 1 unused |0>
    embedded_prep = StatePreparation(
        EmbeddedOperator(base.state_preparation.operation, (0,), 2), 2
    )
    embedded_problem = AEProblem(
        embedded_prep,
        ObjectivePredicate("bit", qubits=(0,)),
        (0, 1),
    )

    assert np.isclose(
        reference_amplitude(base), reference_amplitude(embedded_problem), atol=1e-12
    )

    res_base = estimate(base, algorithm="iqae", backend=ReferenceAdapter(), seed=99)
    res_embed = estimate(
        embedded_problem, algorithm="iqae", backend=ReferenceAdapter(), seed=99
    )
    assert np.isclose(res_base.estimate, res_embed.estimate, atol=1e-12)


def test_metamorphic_reexpress_equivalent_rotations() -> None:
    """MR 4: Reexpressing Ry(θ1 + θ2) as Ry(θ1)Ry(θ2) preserves results."""
    theta1 = 0.3
    theta2 = 0.5
    theta_total = theta1 + theta2
    a_expected = sin(theta_total / 2) ** 2

    p_fused = AEProblem(
        StatePreparation(Gate("RY", (0,), 1, theta_total), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    p_split = AEProblem(
        StatePreparation(
            Sequence((Gate("RY", (0,), 1, theta1), Gate("RY", (0,), 1, theta2)), 1), 1
        ),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )

    assert np.allclose(
        matrix(p_fused.state_preparation), matrix(p_split.state_preparation), atol=1e-12
    )
    assert np.isclose(reference_amplitude(p_fused), a_expected, atol=1e-12)
    assert np.isclose(reference_amplitude(p_split), a_expected, atol=1e-12)


def test_metamorphic_cross_sdk_invariance() -> None:
    """MR 5: Lowering to distinct SDKs preserves logical success probability."""
    a = 0.28
    problem = _bernoulli(a)
    adapters = [QiskitAdapter(), CirqAdapter(), PennyLaneAdapter(), QSharpAdapter()]

    for adapter in adapters:
        circuit = problem.state_preparation
        if hasattr(adapter, "statevector"):
            sv = adapter.statevector(circuit)
            prob = float(np.abs(sv[1]) ** 2)
            assert np.isclose(prob, a, atol=1e-6)
        elif hasattr(adapter, "unitary"):
            u = adapter.unitary(circuit)
            prob = float(np.abs(u[1, 0]) ** 2)
            assert np.isclose(prob, a, atol=1e-6)


def test_metamorphic_serialization_invariance() -> None:
    """MR 6: Serialization -> deserialization leaves semantic result unchanged."""
    a = 0.72
    original = _bernoulli(a)
    serialized = dumps(original)
    restored = loads(serialized)

    assert isinstance(restored, AEProblem)
    assert np.allclose(
        matrix(original.state_preparation),
        matrix(restored.state_preparation),
        atol=1e-12,
    )
    assert np.allclose(
        matrix(grover_for(original)), matrix(grover_for(restored)), atol=1e-12
    )
    assert np.isclose(
        reference_amplitude(original), reference_amplitude(restored), atol=1e-12
    )

    res_orig = estimate(original, algorithm="mlqae", backend=ReferenceAdapter(), seed=7)
    res_rest = estimate(restored, algorithm="mlqae", backend=ReferenceAdapter(), seed=7)
    assert np.isclose(res_orig.estimate, res_rest.estimate, atol=1e-12)
