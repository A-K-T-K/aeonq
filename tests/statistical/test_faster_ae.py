from math import asin, sin, sqrt

import numpy as np
import pytest

from aeonq import AEProblem
from aeonq.adapters.base import BackendAdapter, BackendCapabilities, NormalizedResult
from aeonq.algorithms.faster_ae import FasterAE, attenuated_problem
from aeonq.ir import Gate, ObjectivePredicate, RegisterMap, StatePreparation
from aeonq.ir.nodes import GroverPower, Sequence
from aeonq.reference import reference_amplitude


class IdealBernoulliBackend(BackendAdapter):
    """Fast fake backend sampling the exact semantic objective probability."""

    def capabilities(self):
        return BackendCapabilities(
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
            None,
            metadata={"ideal": "test"},
        )

    def validate(self, ir):
        pass

    def lower(self, ir):
        return ir

    def compile(self, native_program, options=None):
        return native_program

    def execute(self, compiled_program, shots, *, seed=None):
        assert isinstance(compiled_program, Sequence)
        prep, powered = compiled_program.children
        assert isinstance(powered, GroverPower)
        problem = AEProblem(
            prep,
            powered.grover.objective_reflection.predicate,
            tuple(range(prep.n_qubits)),
        )
        amplitude = reference_amplitude(problem)
        theta = asin(sqrt(amplitude))
        p = sin((2 * powered.power + 1) * theta) ** 2
        successes = int(np.random.default_rng(seed).binomial(shots, p))
        n = prep.n_qubits
        good = next(i for i in range(2**n) if problem.objective.evaluate(i, n))
        good_bits = f"{good:0{n}b}"
        bad_bits = "0" * n if good != 0 else "1" * n
        return {good_bits: successes, bad_bits: shots - successes}

    def normalize_result(self, raw, register_map: RegisterMap):
        counts = {register_map.normalize_bitstring(k): v for k, v in raw.items()}
        return NormalizedResult(counts, sum(counts.values()))

    def resource_metrics(self, compiled_program):
        return {"qubits": compiled_program.n_qubits}


def problem(a: float) -> AEProblem:
    prep = StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1)
    return AEProblem(prep, ObjectivePredicate("bit", qubits=(0,)), (0,))


@pytest.mark.parametrize("a", [0.0, 0.01, 0.25, 0.75, 1.0])
def test_attenuation_exact(a: float) -> None:
    assert reference_amplitude(attenuated_problem(problem(a))) == pytest.approx(
        a / 25, abs=1e-13
    )


@pytest.mark.parametrize("a", [0.0, 0.01, 0.25, 0.75, 1.0])
def test_faster_ae_two_stage_seeded(a: float) -> None:
    result = FasterAE(epsilon=0.08, max_iterations=4, seed=17).run(
        problem(a), IdealBernoulliBackend()
    )
    lo, hi = result.confidence_or_credible_interval
    assert 0 <= lo <= result.estimate <= hi <= 1
    assert lo - 1e-12 <= a <= hi + 1e-12
    assert result.total_shots > 0
    assert result.total_oracle_queries == sum(
        result.shots_per_power[k] * (2 * k + 1) for k in result.shots_per_power
    )
    if result.guarantee_status == "THEORETICAL_GUARANTEE_SATISFIED":
        assert hi - lo <= 0.16 + 1e-12
        # Only the midpoint keeps the epsilon error bound on a 2*epsilon interval.
        assert result.estimate == pytest.approx((lo + hi) / 2)


def test_faster_ae_reaches_second_stage() -> None:
    result = FasterAE(epsilon=0.001, max_iterations=4, seed=12).run(
        problem(1.0), IdealBernoulliBackend()
    )
    assert any(row["stage"] == "second" for row in result.diagnostics["stage_history"])
    assert result.diagnostics["switch_iteration"] is not None


@pytest.mark.parametrize("a", [0.5, 0.9, 0.999, 1.0])
def test_faster_ae_point_estimate_is_last_second_stage_centre(a: float) -> None:
    # Near a = 1 the midpoint of a wide intersected interval is biased; an
    # unconverged run reports Nakaji's centre of the last second-stage interval.
    errors = []
    for seed in range(20):
        result = FasterAE(epsilon=0.01, max_iterations=4, seed=seed).run(
            problem(a), IdealBernoulliBackend()
        )
        last = result.diagnostics["stage_history"][-1]
        assert last["stage"] == "second"
        lo, hi = result.diagnostics["theta_interval"]
        assert lo <= result.diagnostics["theta_estimate"] <= hi
        errors.append(abs(result.estimate - a))
    assert sum(errors) / len(errors) < 0.01


def test_faster_ae_budget_and_invalid_parameters() -> None:
    result = FasterAE(epsilon=1e-6, max_iterations=3, max_total_shots=1).run(
        problem(0.5), IdealBernoulliBackend()
    )
    assert result.guarantee_status == "INSUFFICIENT_BUDGET"
    assert result.total_shots == 0
    with pytest.raises(ValueError):
        FasterAE(epsilon=0)
    with pytest.raises(ValueError):
        FasterAE(epsilon=0.1, alpha=1)
