"""A reusable conformance battery that any backend adapter can be run against.

Three levels, each compared with the exact reference semantics:

* operator: the adapter's matrix for A, A†, S0, Sχ, Q, Q², controlled Q and a
  controlled global phase must match *exactly* per entry, within the
  adapter's declared ``unitary_precision``. No free global phase is allowed,
  because a phase that is global for U becomes relative once U is controlled.
* measurement: asymmetric computational basis states must normalize to the
  same logical bit tuple, which exposes bit-order (endianness) errors.
* sampling: counts from A·Q^k must fit the exact outcome distribution
  (likelihood-ratio G-test) and return exactly the requested shots.

Seeds are fixed, so a battery run is deterministic for a given adapter.
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.stats import power_divergence

from aeonq.adapters.base import BackendAdapter
from aeonq.benchmarks.problems import (
    bernoulli,
    discrete_distribution,
    independent_threshold,
    lognormal_tail,
    strike_for_index,
)
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    Gate,
    GlobalPhase,
    GroverPower,
    ObjectivePredicate,
    RegisterMap,
    Sequence,
    StatePreparation,
)
from aeonq.ir.nodes import Operator
from aeonq.problem import AEProblem
from aeonq.reference import grover_for, matrix, statevector

SAMPLING_SHOTS = 4000
SAMPLING_P_THRESHOLD = 1e-3
IMPOSSIBLE_PROBABILITY = 1e-12


@dataclass(frozen=True)
class CheckResult:
    level: str
    case: str
    check: str
    passed: bool | None
    measure: float | None
    threshold: float | None
    detail: str = ""

    @property
    def status(self) -> str:
        return (
            "unsupported"
            if self.passed is None
            else "passed"
            if self.passed
            else "failed"
        )


def _entangled_case() -> tuple[str, AEProblem]:
    # CX makes control/target direction observable; the three rotation axes
    # with asymmetric angles expose sign, half-angle and phase conventions.
    operations = (
        Gate("H", (0,), 2),
        Gate("CX", (0, 1), 2),
        Gate("RY", (1,), 2, 0.9),
        Gate("RZ", (0,), 2, 0.4),
        Gate("RX", (1,), 2, 0.3),
    )
    problem = AEProblem(
        StatePreparation(Sequence(operations, 2), 2),
        ObjectivePredicate("equal", qubits=(0, 1), value=0b01),
        (0, 1),
    )
    return "entangled_cx_n2", problem


def battery_cases() -> list[tuple[str, AEProblem]]:
    """Small problems covering rotations, CX direction, zero-valued controls,
    threshold predicates and an objective ancilla computed by A."""
    cases = [
        bernoulli(0.25),
        independent_threshold(2, 1),
        discrete_distribution((0.1, 0.2, 0.3, 0.4), (1, 2)),
        lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True),
    ]
    # Noncommuting rotations expose sign errors through probabilities, not
    # just matrices: P(1) changes from .09759 to .17173 under sign negation.
    asymmetric = AEProblem(
        StatePreparation(
            Sequence(
                (
                    Gate("RY", (0,), 1, 0.7),
                    Gate("RZ", (0,), 1, 0.4),
                    Gate("RX", (0,), 1, 0.3),
                ),
                1,
            ),
            1,
        ),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    return [(case.case_id, case.problem) for case in cases] + [
        _entangled_case(),
        ("asymmetric_rotations_n1", asymmetric),
    ]


def _operator_nodes(problem: AEProblem) -> dict[str, Operator]:
    q = grover_for(problem)
    n = q.n_qubits
    nodes: dict[str, Operator] = {
        "A": problem.state_preparation,
        "A_adjoint": Adjoint(problem.state_preparation, n),
        "S0": q.zero_reflection,
        "Schi": q.objective_reflection,
        "Q": q,
        "Q_power_2": GroverPower(q, 2, n),
        "controlled_Q": ControlledOperator(q, (n,), (1,), tuple(range(n)), n + 1),
    }
    return nodes


def _operator_checks(adapter: BackendAdapter) -> list[CheckResult]:
    results = []
    nodes = [
        (case, name, node)
        for case, problem in battery_cases()
        for name, node in _operator_nodes(problem).items()
    ]
    nodes.append(
        (
            "phase",
            "controlled_global_phase",
            ControlledOperator(GlobalPhase(pi / 3, 1), (1,), (1,), (0,), 2),
        )
    )
    for case, name, node in nodes:
        try:
            native = adapter.unitary(node)
        except NotImplementedError:
            results.append(
                CheckResult(
                    "operator", case, name, None, None, None, "no matrix export"
                )
            )
            continue
        except Exception as exc:
            results.append(
                CheckResult(
                    "operator", case, name, False, None, None, type(exc).__name__
                )
            )
            continue
        reference = matrix(node)
        if native.shape != reference.shape:
            results.append(
                CheckResult("operator", case, name, False, None, None, "shape")
            )
            continue
        error = float(np.max(np.abs(native - reference)))
        results.append(
            CheckResult(
                "operator",
                case,
                name,
                error <= adapter.unitary_precision,
                error,
                adapter.unitary_precision,
            )
        )
    return results


def _asymmetric_pattern(n: int) -> int:
    # Each pattern differs from its own bit reversal for n >= 2.
    return {1: 0b1, 2: 0b01, 3: 0b011, 4: 0b1101, 5: 0b10011}[n]


def _run(adapter: BackendAdapter, ir: Operator, shots: int, seed: int) -> dict:
    compiled = adapter.compile(adapter.lower(ir))
    n = ir.n_qubits
    raw = adapter.execute(compiled, shots, seed=seed)
    result = adapter.normalize_result(
        raw, RegisterMap(tuple(range(n)), tuple(range(n)))
    )
    counts = {
        sum(bit << i for i, bit in enumerate(bits)): c
        for bits, c in result.counts.items()
    }
    return {"counts": counts, "shots": result.shots}


def _measurement_checks(adapter: BackendAdapter) -> list[CheckResult]:
    results = []
    for n in range(1, 6):
        pattern = _asymmetric_pattern(n)
        gates = tuple(Gate("X", (q,), n) for q in range(n) if (pattern >> q) & 1)
        label = f"basis_{pattern:0{n}b}"
        try:
            observed = _run(adapter, Sequence(gates, n), 32, seed=5)
            passed = observed["shots"] == 32 and observed["counts"] == {pattern: 32}
            detail = "" if passed else f"observed {observed['counts']}"
        except Exception as exc:
            passed, detail = False, type(exc).__name__
        results.append(
            CheckResult("measurement", f"n{n}", label, passed, None, None, detail)
        )
    return results


def _goodness_of_fit(
    counts: dict[int, int], probabilities: np.ndarray
) -> tuple[float, str]:
    """G-test p-value; outcomes with expected count below five are pooled."""
    shots = sum(counts.values())
    impossible = [x for x in counts if probabilities[x] < IMPOSSIBLE_PROBABILITY]
    if impossible:
        return 0.0, f"impossible outcomes {impossible}"
    observed = np.array([counts.get(x, 0) for x in range(len(probabilities))], float)
    expected = probabilities * shots
    large = expected >= 5
    obs = list(observed[large])
    exp = list(expected[large])
    if (~large).any() and expected[~large].sum() > 0:
        obs.append(observed[~large].sum())
        exp.append(expected[~large].sum())
    if len(obs) < 2:
        return 1.0, "single outcome"
    return float(power_divergence(obs, exp, lambda_="log-likelihood").pvalue), ""


def _sampling_checks(adapter: BackendAdapter) -> list[CheckResult]:
    results = []
    for index, (case, problem) in enumerate(battery_cases()):
        q = grover_for(problem)
        n = q.n_qubits
        for power in (0, 1, 2):
            ir = Sequence((problem.state_preparation, GroverPower(q, power, n)), n)
            probabilities = np.abs(statevector(ir)) ** 2
            try:
                observed = _run(
                    adapter, ir, SAMPLING_SHOTS, seed=1009 + 31 * index + power
                )
                if observed["shots"] != SAMPLING_SHOTS:
                    passed, p_value, detail = False, None, "wrong shot total"
                else:
                    p_value, detail = _goodness_of_fit(
                        observed["counts"], probabilities
                    )
                    passed = p_value >= SAMPLING_P_THRESHOLD
            except Exception as exc:
                passed, p_value, detail = False, None, type(exc).__name__
            results.append(
                CheckResult(
                    "sampling",
                    case,
                    f"A_Q^{power}",
                    passed,
                    p_value,
                    SAMPLING_P_THRESHOLD,
                    detail,
                )
            )
    return results


def run_battery(adapter: BackendAdapter) -> list[CheckResult]:
    """Run every conformance level; a correct adapter passes every check."""
    return (
        _operator_checks(adapter)
        + _measurement_checks(adapter)
        + _sampling_checks(adapter)
    )
