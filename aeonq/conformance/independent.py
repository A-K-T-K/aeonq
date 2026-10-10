"""Checks against oracles that share no code with the semantic core.

The battery compares adapters with ``aeonq.reference``, which evaluates the
same IR the adapters lower. A defect in the IR, its synthesis or the
reference simulator could therefore be shared by both sides and cancel. Here
every operator is also produced by sources that never touch that code
(``aeonq.oracles``), and every source is held to closed forms:

* amplitude: ||Pi_good A|0>||^2 = a, with a from the problem statement;
* rotation: on span{psi_good, psi_bad} of A|0>, Q must act as the rotation
  [[cos 2t, sin 2t], [-sin 2t, cos 2t]] with sin^2 t = a, and leave that span
  invariant. The global phase of Q is not free here: -Q fails;
* Q^2 likewise rotates by 4t;
* controlled Q: identity when the control is 0, the rotation when it is 1;
* amplification: P_good(Q^k A|0>) = sin^2((2k+1) t) for k = 0..4.

Sources, per case and operator:

* ``numpy:handwritten``: the oracle gate lists on the NumPy simulator;
* ``<sdk>:handwritten``: native circuits written by hand, SDK matrix;
* ``<sdk>:adapter``: aeonq's adapter lowering of the IR, SDK matrix;
* ``numpy:<sdk>-program``: the adapter's native program read gate by gate
  into the NumPy simulator (a second simulator for aeonq's output);
* ``ir:reference``: aeonq.reference, the semantics under test.

Every source is also compared entry by entry with ``numpy:handwritten``.
Negative controls (Q without its -1, qubit-reversed Q, A with a negated
angle) must fail, showing that the checks can detect those faults.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from aeonq.ir import ControlledOperator, GroverPower
from aeonq.ir.nodes import Operator
from aeonq.oracles import numpy_sim
from aeonq.oracles.cases import CASES, OracleCase, grover_ops
from aeonq.oracles.handwritten import HANDWRITTEN, PRECISION
from aeonq.problem import AEProblem
from aeonq.reference import grover_for, matrix, reference_amplitude

OPERATORS = ("A", "Q", "Q_power_2", "controlled_Q")
AMPLIFICATION_POWERS = range(5)


def _ir_problem(name: str) -> tuple[AEProblem, float]:
    """The aeonq problem with this name and the a_true its builder reports."""
    from aeonq.benchmarks.problems import (
        bernoulli,
        discrete_distribution,
        independent_threshold,
        lognormal_tail,
        strike_for_index,
    )

    builders: dict[str, Callable[[], Any]] = {
        "bernoulli_0.25": lambda: bernoulli(0.25),
        "uniform_threshold_n2_k1": lambda: independent_threshold(2, 1),
        "distribution_0.1_0.2_0.3_0.4_marked_1_2": lambda: discrete_distribution(
            (0.1, 0.2, 0.3, 0.4), (1, 2)
        ),
        "lognormal_tail_n2_k2_flag": lambda: lognormal_tail(
            2, strike_for_index(2, 2), objective_ancilla=True
        ),
    }
    built = builders[name]()
    return built.problem, built.a_true


def _ir_nodes(problem: AEProblem) -> dict[str, Operator]:
    q = grover_for(problem)
    n = q.n_qubits
    return {
        "A": problem.state_preparation,
        "Q": q,
        "Q_power_2": GroverPower(q, 2, n),
        "controlled_Q": ControlledOperator(q, (n,), (1,), tuple(range(n)), n + 1),
    }


def _numpy_matrices(case: OracleCase) -> dict[str, np.ndarray]:
    n = case.n_qubits
    grover = grover_ops(case)
    return {
        "A": numpy_sim.unitary(case.preparation, n),
        "Q": numpy_sim.unitary(grover, n),
        "Q_power_2": numpy_sim.unitary(grover + grover, n),
        "controlled_Q": numpy_sim.unitary(numpy_sim.controlled(grover, n), n + 1),
    }


def _adapters() -> dict[str, Any]:
    from aeonq.adapters.cirq import CirqAdapter
    from aeonq.adapters.pennylane import PennyLaneAdapter
    from aeonq.adapters.qiskit import QiskitAdapter
    from aeonq.adapters.qsharp import QSharpAdapter

    return {
        "qiskit": QiskitAdapter(),
        "cirq": CirqAdapter(),
        "pennylane": PennyLaneAdapter(),
        "qsharp": QSharpAdapter(),
    }


def _second_simulator(sdk: str, adapter: Any, node: Operator) -> np.ndarray:
    from aeonq.oracles import sdk_import

    program = adapter.lower(node)
    n = node.n_qubits
    if sdk == "qiskit":
        ops = sdk_import.from_qiskit(program)
    elif sdk == "cirq":
        import cirq

        ops = sdk_import.from_cirq(program, cirq.LineQubit.range(n))
    else:
        ops = sdk_import.from_pennylane(program)
    return numpy_sim.unitary(ops, n)


def collect_sources(case: OracleCase) -> dict[str, tuple[dict[str, np.ndarray], float]]:
    """Every source's matrices for one case, with its entry precision."""
    problem, _ = _ir_problem(case.name)
    nodes = _ir_nodes(problem)
    sources: dict[str, tuple[dict[str, np.ndarray], float]] = {
        "numpy:handwritten": (_numpy_matrices(case), 1e-9),
        "ir:reference": ({k: matrix(v) for k, v in nodes.items()}, 1e-9),
    }
    for sdk, build in HANDWRITTEN.items():
        sources[f"{sdk}:handwritten"] = (build(case), PRECISION[sdk])
    for sdk, adapter in _adapters().items():
        sources[f"{sdk}:adapter"] = (
            {k: adapter.unitary(v) for k, v in nodes.items()},
            adapter.unitary_precision,
        )
        if sdk != "qsharp":
            sources[f"numpy:{sdk}-program"] = (
                {k: _second_simulator(sdk, adapter, v) for k, v in nodes.items()},
                1e-9,
            )
    return sources


# ----------------------------------------------------------- closed forms


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], complex
    )


def _plane(case: OracleCase) -> np.ndarray:
    """Columns psi_good, psi_bad: normalized halves of A|0> from the oracle."""
    state = numpy_sim.statevector(case.preparation, case.n_qubits)
    good = case.good_mask()
    columns = np.zeros((state.size, 2), complex)
    columns[good, 0] = state[good]
    columns[~good, 1] = state[~good]
    norms = np.linalg.norm(columns, axis=0)
    if np.any(norms == 0):
        raise ValueError("rotation check needs 0 < a < 1")
    return columns / norms


def rotation_error(case: OracleCase, unitary: np.ndarray, turns: int) -> float:
    """Distance of U from the closed-form rotation by 2*turns*theta on the
    good/bad plane, plus any amplitude it leaks out of the plane."""
    plane = _plane(case)
    restricted = plane.conj().T @ unitary @ plane
    leak = unitary @ plane - plane @ restricted
    return float(
        max(
            np.max(np.abs(restricted - _rotation(2 * turns * case.theta))),
            np.max(np.abs(leak)),
        )
    )


def controlled_error(case: OracleCase, unitary: np.ndarray) -> float:
    dim = 2**case.n_qubits
    off = max(np.max(np.abs(unitary[:dim, dim:])), np.max(np.abs(unitary[dim:, :dim])))
    idle = np.max(np.abs(unitary[:dim, :dim] - np.eye(dim)))
    return float(max(off, idle, rotation_error(case, unitary[dim:, dim:], 1)))


def amplitude_error(case: OracleCase, a_matrix: np.ndarray) -> float:
    good = case.good_mask()
    return float(abs(np.sum(np.abs(a_matrix[good, 0]) ** 2) - case.a))


def amplification_error(case: OracleCase, a_matrix: np.ndarray, q: np.ndarray) -> float:
    good = case.good_mask()
    state = a_matrix[:, 0]
    worst = 0.0
    for k in AMPLIFICATION_POWERS:
        expected = np.sin((2 * k + 1) * case.theta) ** 2
        worst = max(worst, abs(float(np.sum(np.abs(state[good]) ** 2)) - expected))
        state = q @ state
    return worst


def closed_form_checks(
    case: OracleCase, matrices: dict[str, np.ndarray], precision: float
) -> list[tuple[str, str, float, float]]:
    """(operator, check, error, tolerance). Derived quantities sum up to
    2^n entries, so the tolerance scales with the dimension."""
    dim = 2**case.n_qubits
    tolerance = precision * dim
    return [
        ("A", "amplitude", amplitude_error(case, matrices["A"]), tolerance),
        ("Q", "rotation_2theta", rotation_error(case, matrices["Q"], 1), tolerance),
        (
            "Q_power_2",
            "rotation_4theta",
            rotation_error(case, matrices["Q_power_2"], 2),
            tolerance,
        ),
        (
            "controlled_Q",
            "identity_then_rotation",
            controlled_error(case, matrices["controlled_Q"]),
            2 * tolerance,
        ),
        (
            "A,Q",
            "amplification_sin2_(2k+1)theta",
            amplification_error(case, matrices["A"], matrices["Q"]),
            tolerance * (len(AMPLIFICATION_POWERS) + 1),
        ),
    ]


def negative_controls(
    case: OracleCase,
) -> dict[str, tuple[dict[str, np.ndarray], set[str]]]:
    """Faulty variants of the oracle and the operators each one changes.

    A sound set of checks must reject every variant; individual checks may
    pass (an angle sign flip in A leaves the amplitude check satisfied).
    """
    n = case.n_qubits
    good = _numpy_matrices(case)
    without_minus = tuple(op for op in grover_ops(case) if op.name != "GPHASE")
    controls = {
        "negative:Q_without_global_phase": (
            {
                **good,
                "Q": numpy_sim.unitary(without_minus, n),
                "Q_power_2": numpy_sim.unitary(without_minus * 2, n),
                "controlled_Q": numpy_sim.unitary(
                    numpy_sim.controlled(without_minus, n), n + 1
                ),
            },
            {"Q", "Q_power_2", "controlled_Q"},
        )
    }
    if n > 1:
        reverse = np.array([int(f"{x:0{n}b}"[::-1], 2) for x in range(2**n)])
        q = good["Q"][np.ix_(reverse, reverse)]
        controls["negative:Q_qubits_reversed"] = (
            {**good, "Q": q, "Q_power_2": q @ q},
            {"Q", "Q_power_2"},
        )
    negated = tuple(
        numpy_sim.Op(
            op.name, op.target, -op.parameters[0], op.controls, op.control_values
        )
        if op.name == "RY"
        else op
        for op in case.preparation
    )
    if negated != case.preparation:
        controls["negative:A_angle_negated"] = (
            {**good, "A": numpy_sim.unitary(negated, n)},
            {"A"},
        )
    return controls


def independent_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for make in CASES:
        case = make()
        problem, builder_a = _ir_problem(case.name)
        for subject, value in (
            ("aeonq problem builder a_true", builder_a),
            ("ir:reference amplitude", reference_amplitude(problem)),
        ):
            rows.append(
                {
                    "case": case.name,
                    "operator": "A",
                    "kind": "closed_form",
                    "check": "amplitude",
                    "subject": subject,
                    "compared_with": "analytic a",
                    "error": abs(value - case.a),
                    "tolerance": 1e-12,
                    "expected_to_pass": True,
                }
            )
        sources = collect_sources(case)
        oracle = sources["numpy:handwritten"][0]
        for subject, (matrices, precision) in sources.items():
            for operator, check, error, tolerance in closed_form_checks(
                case, matrices, precision
            ):
                rows.append(
                    {
                        "case": case.name,
                        "operator": operator,
                        "kind": "closed_form",
                        "check": check,
                        "subject": subject,
                        "compared_with": "closed form",
                        "error": error,
                        "tolerance": tolerance,
                        "expected_to_pass": True,
                    }
                )
            if subject == "numpy:handwritten":
                continue
            for operator in OPERATORS:
                rows.append(
                    {
                        "case": case.name,
                        "operator": operator,
                        "kind": "cross",
                        "check": "entrywise_exact",
                        "subject": subject,
                        "compared_with": "numpy:handwritten",
                        "error": float(
                            np.max(np.abs(matrices[operator] - oracle[operator]))
                        ),
                        "tolerance": max(precision, 1e-9),
                        "expected_to_pass": True,
                    }
                )
        for subject, (matrices, faulty) in negative_controls(case).items():
            if all(np.allclose(matrices[k], oracle[k], atol=1e-12) for k in faulty):
                continue  # the variant equals the oracle here, so it is no fault
            checks = [
                (operator, check, "closed form", error, tolerance)
                for operator, check, error, tolerance in closed_form_checks(
                    case, matrices, 1e-9
                )
                if not faulty.isdisjoint(operator.split(","))
            ] + [
                (
                    operator,
                    "entrywise_exact",
                    "numpy:handwritten",
                    float(np.max(np.abs(matrices[operator] - oracle[operator]))),
                    1e-9,
                )
                for operator in sorted(faulty)
            ]
            for operator, check, compared_with, error, tolerance in checks:
                rows.append(
                    {
                        "case": case.name,
                        "operator": operator,
                        "kind": "negative_control",
                        "check": check,
                        "subject": subject,
                        "compared_with": compared_with,
                        "error": error,
                        "tolerance": tolerance,
                        "expected_to_pass": None,
                    }
                )
    for row in rows:
        row["passed"] = row["error"] <= row["tolerance"]
    return rows


def undetected_negative_controls(
    rows: list[dict[str, Any]], *, closed_form_only: bool = False
) -> list[tuple[str, str]]:
    """(case, variant) pairs that no check rejected."""
    detected: dict[tuple[str, str], bool] = {}
    for row in rows:
        if row["kind"] != "negative_control":
            continue
        if closed_form_only and row["compared_with"] != "closed form":
            continue
        key = (row["case"], row["subject"])
        detected[key] = detected.get(key, False) or not row["passed"]
    return [key for key, caught in detected.items() if not caught]
