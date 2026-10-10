import numpy as np
import pytest

from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.compiler import AECompiler, CompilerCost, rewrite
from aeonq.ir import (
    Adjoint,
    Gate,
    GlobalPhase,
    GroverPower,
    ObjectivePredicate,
    ObjectiveReflection,
    Parameter,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.nodes import GroverOperator
from aeonq.reference import matrix


def test_exact_adjacent_rewrites_and_provenance() -> None:
    n = 2
    u = Sequence((Gate("H", (0,), n), Gate("CX", (0, 1), n)), n)
    source = Sequence(
        (
            Gate("H", (0,), n),
            Gate("H", (0,), n),
            Gate("RY", (1,), n, 0.31),
            Gate("RY", (1,), n, -0.31),
            u,
            Adjoint(u, n),
            GlobalPhase(0.3, n),
            GlobalPhase(0.4, n),
        ),
        n,
    )
    output, records = rewrite(source)
    assert np.allclose(matrix(source), matrix(output))
    assert {r.rule for r in records} >= {
        "self_inverse_pair",
        "same_axis_rotation_fusion",
        "adjoint_right",
        "phase_fusion",
    }
    assert np.allclose(matrix(output), np.exp(0.7j) * np.eye(4))


def test_symbolic_rotation_fusion_and_nonmatching_parameters() -> None:
    source = Sequence(
        (
            Gate("RX", (0,), 1, Parameter("x", 1)),
            Gate("RX", (0,), 1, Parameter("x", -1)),
            Gate("RY", (0,), 1, Parameter("y", 1)),
            Gate("RY", (0,), 1, Parameter("z", 1)),
        ),
        1,
    )
    output, records = rewrite(source)
    assert len(output.children) == 2
    assert len(records) == 1


def test_grover_power_fusion_does_not_cancel_iterations() -> None:
    prep = StatePreparation(Gate("RY", (0,), 1, 0.72), 1)
    q = GroverOperator(
        prep,
        ZeroReflection(1, (0,)),
        ObjectiveReflection(1, ObjectivePredicate("bit", qubits=(0,))),
        1,
    )
    source = Sequence((GroverPower(q, 2, 1), GroverPower(q, 3, 1)), 1)
    output, records = rewrite(source)
    assert output.children == (GroverPower(q, 5, 1),)
    assert records[-1].rule == "same_grover_power_fusion"
    assert np.allclose(matrix(source), matrix(output))
    repeated, repeated_records = rewrite(Sequence((q, q), 1))
    assert repeated.children == (q, q)
    assert not repeated_records


def test_compiler_selects_minimum_measured_cost() -> None:
    n = 2
    source = Sequence(
        (
            Gate("H", (0,), n),
            Gate("H", (0,), n),
            Gate("RY", (1,), n, 0.42),
            Gate("CX", (1, 0), n),
        ),
        n,
    )
    result = AECompiler().compile(source, QiskitAdapter())
    assert result.selected_cost == min(cost for _, cost in result.candidate_costs)
    assert len(result.candidate_costs) == 4
    assert result.metrics["multi_qubit_gates"] == 0
    assert any(record.rule == "self_inverse_pair" for record in result.rewrite_records)
    assert np.allclose(
        np.asarray(
            __import__("qiskit").quantum_info.Operator(result.compiled_program).data
        ),
        matrix(source),
        atol=1e-10,
    )


def test_resource_cost_rejects_unknown_gate_basis() -> None:
    cost = CompilerCost()
    with pytest.raises(ValueError, match="decomposed"):
        cost.evaluate(
            {"qubits": 3, "depth": 2, "two_qubit_gates": 0, "multi_qubit_gates": 1}, 3
        )
    with pytest.raises(ValueError, match="T-count"):
        CompilerCost(t_weight=1).evaluate(
            {"qubits": 2, "depth": 2, "two_qubit_gates": 1, "t_count": None}, 2
        )
