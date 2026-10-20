"""Translator-fidelity machinery, and the converter behaviour it observed.

The ``observed`` tests pin what the pinned third-party converters did when the
study was run (benchmarks/results/processed/translator_fidelity.md). If one
fails after a dependency change, the converter changed: rerun the study and
update the stored results rather than this test.
"""

from math import pi

import numpy as np
import pytest

from aeonq.conformance import translation
from aeonq.conformance.translation import (
    PATHS,
    angle_survival,
    evaluate_translation,
    qasm_parameters,
    qubit_order,
    study_cases,
)

pytest.importorskip("cirq.contrib.qasm_import")
pytest.importorskip("qiskit_qasm3_import")
pytest.importorskip("pytket.extensions.cirq")

PATH = {path.name: path for path in PATHS}
NODES = dict(study_cases())["bernoulli_0.25"]


def _row(path: str, node: str) -> dict:
    return evaluate_translation(PATH[path], "bernoulli_0.25", node, NODES[node], seed=7)


def test_qasm_parameters_evaluate_pi_expressions():
    text = "OPENQASM 2.0;\nry(-pi/3) q[0];\ncu(pi*0.5,0,0,1e-1) q[0],q[1];\nh q[0];"
    assert np.allclose(qasm_parameters(text), [-pi / 3, pi / 2, 0, 0, 0.1])


def test_angle_survival_is_periodic_in_four_pi_and_ignores_structural_angles():
    assert angle_survival([0.3, pi / 2], [0.3 + 4 * pi]) == (1, pytest.approx(0), 0)
    checked, _, reexpressed = angle_survival([0.3], [0.3 + 2 * pi])
    assert (checked, reexpressed) == (1, 1)  # Ry(t + 2 pi) = -Ry(t)


def test_qubit_order_detects_reversal():
    u = np.kron(np.diag([1, 1j]), np.eye(2))  # S on qubit 1 (little-endian)
    swapped = np.kron(np.eye(2), np.diag([1, 1j]))
    assert qubit_order(u, u, 1e-12) == "identity"
    assert qubit_order(u, swapped, 1e-12) == "reversed"


def test_summary_distinguishes_permutations_from_unmatched_operators():
    from benchmarks.translator_fidelity import summarize

    rows = [
        {"path": "test", "status": "evaluated", "qubit_order": order}
        for order in ("identity", "none", "reversed", "permuted102")
    ]
    summary = summarize(rows)[0]
    assert summary["qubit_order_changed"] == 2
    assert summary["qubit_order_unmatched"] == 1


def test_row_flags_are_consistent():
    row = _row("qiskit>qasm2>qiskit", "Q")
    assert row["status"] == "evaluated"
    assert row["source_vs_ir_max_error"] < 1e-9
    if row["equal_exact"]:
        assert row["equal_up_to_global_phase"]
    assert row["count_invisible_failure"] == (
        row["count_pass"] and not row["operator_pass"]
    )


def test_observed_qasm_exporters_drop_the_minus_one_of_q():
    for path in ("qiskit>qasm2>qiskit", "qiskit>qasm3>qiskit", "cirq>qasm2>qiskit"):
        row = _row(path, "Q")
        assert row["equal_up_to_global_phase"] and not row["equal_exact"], path
        assert abs(abs(row["global_phase_offset"]) - pi) < 1e-9, path
        assert row["equal_when_controlled"] is False, path
        assert row["count_pass"] and row["count_invisible_failure"], path


def test_observed_pytket_cirq_changes_the_phase_of_ry():
    row = _row("qiskit>pytket>cirq", "A")
    assert row["equal_up_to_global_phase"] and not row["equal_exact"]
    assert row["global_phase_offset"] == pytest.approx(pi / 6)  # RY(pi/3)


@pytest.mark.parametrize("statement", ["ry(-0.7) q[0];", "rz(7.0) q[0];"])
def test_observed_qdk_import_reduces_rotation_angles_modulo_two_pi(statement):
    pytest.importorskip("qdk.openqasm")
    text = f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n{statement}\n'
    qiskit = translation.TARGETS["qiskit"].unitary(
        translation._into_qiskit_qasm3(text, 1)
    )
    qdk = translation.TARGETS["qsharp"].unitary(translation._into_qsharp(text, 1))
    assert np.allclose(qdk, -qiskit, atol=1e-6)


def test_observed_pennylane_reads_gphase_with_the_opposite_sign():
    probe = [
        row
        for row in translation.importer_gate_probe()
        if row["importer"] == "pennylane from_qasm3"
        and row["statement"] == "gphase(0.7);"
    ]
    assert probe[0]["equal_up_to_global_phase"] and not probe[0]["equal_exact"]
    assert probe[0]["global_phase_offset"] == pytest.approx(-1.4)


def test_unavailable_paths_are_reported_not_skipped():
    row = evaluate_translation(
        translation.TranslationPath(
            "x>y>z",
            "qiskit",
            "y",
            "qiskit",
            lambda p, n: p,
            lambda p, n: p,
            ("module_that_does_not_exist",),
        ),
        "bernoulli_0.25",
        "Q",
        NODES["Q"],
        seed=0,
    )
    assert row["status"] == "unavailable"


def test_observed_qiskit_native_grover_operator_loses_its_minus_one_too():
    # Not an aeonq artifact: Qiskit's own grover_operator carries the -1 as a
    # global phase, and the QASM round trip drops it just the same.
    case, name, program, node = next(
        s
        for s in translation.native_qiskit_sources()
        if s[0] == "bernoulli_0.25" and s[1] == "Q (grover_operator)"
    )
    row = evaluate_translation(
        PATH["qiskit>qasm2>qiskit"], case, name, node, seed=7, source_program=program
    )
    assert row["source_vs_ir_max_error"] < 1e-9
    assert row["equal_up_to_global_phase"] and not row["equal_exact"]
    assert row["equal_when_controlled"] is False and row["count_invisible_failure"]


def test_observed_grover_operator_discards_the_oracle_global_phase():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator
    from qiskit.quantum_info import Operator

    prep = QuantumCircuit(1)
    prep.ry(1.0, 0)
    plain, phased = QuantumCircuit(1), QuantumCircuit(1, global_phase=pi)
    plain.z(0)
    phased.z(0)
    assert np.allclose(
        Operator(grover_operator(phased, prep)).data,
        Operator(grover_operator(plain, prep)).data,
    )
