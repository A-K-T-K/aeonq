import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from aeonq.adapters.qiskit_multicontrol import select_multi_control
from aeonq.compiler import CompilerCost
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.compiler import AECompiler
from aeonq.ir import ZeroReflection
from aeonq.reference import matrix


@pytest.mark.parametrize("controls", [3, 4])
@pytest.mark.parametrize("phase", [False, True])
def test_multicontrol_strategies_preserve_contract(controls: int, phase: bool) -> None:
    selection = select_multi_control(
        controls,
        phase=phase,
        clean_available=controls - 2,
        dirty_available=controls - 2,
    )
    assert {c.strategy for c in selection.candidates} == {
        "ancilla_free",
        "clean_ladder",
        "dirty_ladder",
    }
    assert selection.selected.cost == min(c.cost for c in selection.candidates)
    for candidate in selection.candidates:
        n = candidate.circuit.num_qubits
        ideal = QuantumCircuit(n)
        if phase:
            ideal.h(controls)
        ideal.mcx(list(range(controls)), controls)
        if phase:
            ideal.h(controls)
        actual = np.asarray(Operator(candidate.circuit).data)
        reference = np.asarray(Operator(ideal).data)
        if candidate.strategy == "clean_ladder":
            # Clean ancillas are promised |0>; compare only that input subspace.
            clean_columns = list(range(2 ** (controls + 1)))
            assert np.allclose(
                actual[:, clean_columns], reference[:, clean_columns], atol=1e-10
            )
        else:
            # Dirty ancillas must be restored for arbitrary input basis states.
            assert np.allclose(actual, reference, atol=1e-10)
        assert candidate.metrics["qubits"] == n
        assert candidate.metrics["two_qubit_gates"] >= 0
        assert candidate.metrics["depth"] >= 0


def test_workspace_and_cost_selection() -> None:
    no_workspace = select_multi_control(4)
    assert len(no_workspace.candidates) == 1
    only_clean = select_multi_control(4, clean_available=2)
    assert {c.strategy for c in only_clean.candidates} == {
        "ancilla_free",
        "clean_ladder",
    }
    with pytest.raises(ValueError):
        select_multi_control(2)
    with pytest.raises(ValueError):
        select_multi_control(4, clean_available=-1)
    with pytest.raises(ValueError, match="T-count"):
        select_multi_control(3, cost_model=CompilerCost(t_weight=1))


def test_adapter_uses_available_workspace_for_reflection() -> None:
    reflection = ZeroReflection(5, tuple(range(5)))
    expected = matrix(reflection)
    clean_adapter = QiskitAdapter(clean_ancillas=2)
    clean_native = clean_adapter.lower(reflection)
    actual_clean = np.asarray(Operator(clean_native).data)
    expected_clean = np.zeros((128, 32), dtype=complex)
    expected_clean[:32, :] = expected
    assert np.allclose(actual_clean[:, :32], expected_clean, atol=1e-10)
    dirty_adapter = QiskitAdapter(dirty_ancillas=2)
    actual_dirty = np.asarray(Operator(dirty_adapter.lower(reflection)).data)
    assert np.allclose(actual_dirty, np.kron(np.eye(4), expected), atol=1e-10)
    compiled = AECompiler(qiskit_optimization_levels=(0, 1)).compile(
        reflection, clean_adapter
    )
    assert compiled.metrics["qubits"] == 7
    assert compiled.selected_cost == min(cost for _, cost in compiled.candidate_costs)
