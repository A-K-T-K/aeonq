from math import asin, pi, sqrt

import numpy as np
import pytest

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.inference.fisher import (
    fisher_information,
    success_derivative,
    success_probability,
)
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation
from aeonq.scheduling import (
    CostAwareSchedule,
    ExecutionCost,
    ExpectedInformationSchedule,
    FisherInformationSchedule,
    FixedSchedule,
    ScheduleState,
)
from aeonq.scheduling.information import expected_information


@pytest.mark.parametrize("a", [1e-8, 1e-4, 0.01, 0.1, 0.4, 0.75, 0.99, 1 - 1e-8])
@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_success_derivative_matches_finite_difference(a: float, k: int) -> None:
    step = min(1e-7, a / 10, (1 - a) / 10)
    finite_difference = (
        success_probability(a + step, k) - success_probability(a - step, k)
    ) / (2 * step)
    assert success_derivative(a, k) == pytest.approx(
        finite_difference, rel=2e-4, abs=1e-5
    )


def test_fisher_limits_and_nonregular_points() -> None:
    for k in range(4):
        m = 2 * k + 1
        assert success_derivative(0, k) == m * m
        assert success_derivative(1, k) == m * m
        assert np.isinf(fisher_information(0, k).value)
        assert not fisher_information(1, k).regular
        assert fisher_information(0.3, k).value == pytest.approx(m * m / (0.3 * 0.7))
    # p_1(a)=1 at a=1/4; ordinary score is nonregular there.
    info = fisher_information(0.25, 1)
    assert not info.regular
    assert np.isfinite(info.value)


def test_schedulers_reject_aliases_and_honor_budget() -> None:
    wide = ScheduleState((0, pi / 2), 0, 8, (16, 32), remaining_shots=20)
    assert wide.candidates() == ((0, 16, 0),)
    for scheduler in (
        FisherInformationSchedule(),
        ExpectedInformationSchedule(),
        CostAwareSchedule(),
    ):
        chosen = scheduler.select(wide)
        assert chosen.power == 0 and chosen.shots == 16
    with pytest.raises(ValueError, match="crosses phase branches"):
        FixedSchedule(((8, 16),)).select(wide)
    narrow = ScheduleState((0.05, 0.08), 0, 8, (16, 32), remaining_queries=160)
    for scheduler in (
        FisherInformationSchedule(),
        ExpectedInformationSchedule(),
        CostAwareSchedule(),
    ):
        chosen = scheduler.select(narrow)
        assert chosen.branch is not None
        assert chosen.shots * (2 * chosen.power + 1) <= 160
        assert chosen.score is not None and chosen.score >= 0


def test_expected_information_and_cost_tradeoff() -> None:
    region = (0.1, 0.15)
    assert expected_information(region, 1, 16) > expected_information(region, 0, 16)
    assert expected_information((0.1, 0.1), 1, 16) == 0
    state = ScheduleState(
        region,
        0,
        2,
        (8, 32),
        resource_metrics_by_power={
            0: {"depth": 2, "two_qubit_gates": 0},
            1: {"depth": 10, "two_qubit_gates": 2},
            2: {"depth": 20, "two_qubit_gates": 4},
        },
    )
    low_overhead = CostAwareSchedule(ExecutionCost(oracle_weight=0, shot_weight=1))
    high_overhead = CostAwareSchedule(
        ExecutionCost(oracle_weight=0, shot_weight=1, submission_overhead=1000)
    )
    assert low_overhead.select(state).shots in (8, 32)
    assert high_overhead.select(state).shots == 32
    depth_cost = ExecutionCost(oracle_weight=0, depth_weight=1)
    assert depth_cost.estimate(1, 8, state.resource_metrics_by_power[1]) == 80
    with pytest.raises(ValueError, match="depth metric"):
        depth_cost.estimate(1, 8)


def test_iqae_uses_supplied_scheduler() -> None:
    a = 0.1
    problem = AEProblem(
        StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(a))), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    schedule = FixedSchedule(((0, 128), (0, 64)))
    result = IQAE(
        epsilon=0.001, max_rounds=2, scheduler=schedule, shot_options=(64, 128), seed=7
    ).run(problem, QiskitAdapter())
    assert result.powers_used == (0, 0)
    assert result.total_shots == 192
