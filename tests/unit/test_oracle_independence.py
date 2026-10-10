"""The independent oracles must not import the code they are meant to check."""

import ast
from pathlib import Path

import numpy as np
import pytest

from aeonq.oracles import numpy_sim
from aeonq.oracles.cases import CASES, grover_ops

ORACLES = Path(__file__).resolve().parents[2] / "aeonq" / "oracles"


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("path", sorted(ORACLES.glob("*.py")), ids=lambda p: p.name)
def test_oracle_modules_import_no_semantic_core(path):
    shared = {
        m
        for m in _imported_modules(path)
        if m.split(".")[0] == "aeonq" and not m.startswith("aeonq.oracles")
    }
    assert shared == set()


def test_textbook_gates_are_unitary_and_rotations_have_half_angles():
    for name, parameters in (
        ("H", ()),
        ("RX", (0.3,)),
        ("RY", (0.3,)),
        ("RZ", (0.3,)),
        ("P", (0.3,)),
        ("U", (0.3, 0.4, 0.5)),
    ):
        u = numpy_sim.gate_matrix(name, parameters)
        assert np.allclose(u.conj().T @ u, np.eye(2))
    # Ry(2 pi) = -I: the angle is not periodic in 2 pi, only in 4 pi.
    assert np.allclose(numpy_sim.gate_matrix("RY", (2 * np.pi,)), -np.eye(2))


def test_little_endian_controls_and_controlled_global_phase():
    # X on qubit 1 controlled by qubit 0 maps |01> (x = 1) to |11> (x = 3).
    psi = numpy_sim.statevector(
        [numpy_sim.Op("X", 0), numpy_sim.Op("X", 1, controls=(0,))], 2
    )
    assert np.argmax(abs(psi)) == 3
    phase = numpy_sim.unitary(
        [numpy_sim.Op("GPHASE", None, np.pi / 3, controls=(1,))], 2
    )
    w = np.exp(1j * np.pi / 3)
    assert np.allclose(phase, np.diag([1, 1, w, w]))  # x = 2, 3 have qubit 1 set


@pytest.mark.parametrize("make", CASES, ids=lambda f: f.__name__)
def test_grover_is_the_closed_form_rotation(make):
    case = make()
    n = case.n_qubits
    q = numpy_sim.unitary(grover_ops(case), n)
    psi = numpy_sim.statevector(case.preparation, n)
    good = case.good_mask()
    assert np.isclose(np.sum(abs(psi[good]) ** 2), case.a)
    for k in range(5):
        assert np.isclose(
            np.sum(abs(psi[good]) ** 2), np.sin((2 * k + 1) * case.theta) ** 2
        )
        psi = q @ psi
