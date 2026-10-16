import json

import pytest

from aeonq import estimate
from aeonq.__main__ import main
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.base import Experiment
from aeonq.benchmarks.problems import bernoulli, lognormal_tail, strike_for_index
from aeonq.resources import (
    circuit_metrics,
    estimate_resources,
    format_circuit_metrics,
)

FLAG_CASE = lognormal_tail(2, strike_for_index(2, 2), objective_ancilla=True).problem


def _totals(item):
    if hasattr(item, "total_shots"):
        return (
            item.total_shots,
            item.total_oracle_queries,
            item.total_circuit_evaluations,
        )
    return (item.shots, item.oracle_queries, item.circuit_evaluations)


@pytest.mark.parametrize(
    "algorithm, options",
    [
        ("mlqae", {"powers": (0, 1, 2, 4, 8), "shots_per_power": 64}),
        ("mlqae", {"powers": (0, 2, 2, 5), "shots_per_power": 32}),
        ("canonical_qae", {"evaluation_qubits": 3, "shots": 128}),
    ],
)
@pytest.mark.parametrize("a", [0.0, 0.3, 1.0])
def test_fixed_schedule_budget_is_exact(algorithm, options, a):
    problem = bernoulli(a).problem
    report = estimate_resources(
        problem, algorithm=algorithm, backend=ReferenceAdapter(), **options
    )
    run = estimate(
        problem, algorithm=algorithm, backend=ReferenceAdapter(), seed=3, **options
    )
    assert report.exact
    assert _totals(report) == _totals(run)
    assert report.max_power == run.max_power


@pytest.mark.parametrize("a", [0.0, 1e-6, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adaptive_budget_bounds_every_run(a, seed):
    problem = bernoulli(a).problem
    for algorithm, options in (
        ("iqae", {"shots_per_round": 64, "max_rounds": 6, "max_power": 8}),
        ("faster_ae", {"max_iterations": 4}),
    ):
        report = estimate_resources(
            problem, algorithm=algorithm, backend=ReferenceAdapter(), **options
        )
        run = estimate(
            problem,
            algorithm=algorithm,
            backend=ReferenceAdapter(),
            seed=seed,
            **options,
        )
        assert not report.exact
        assert all(b >= r for b, r in zip(_totals(report), _totals(run)))
        assert report.max_power >= run.max_power
        listed = {c.power for c in report.circuits}
        if algorithm == "faster_ae":
            assert set(run.powers_used) <= listed


def test_faster_ae_bound_is_attained_when_budget_runs_out():
    # a = 0.5 does not converge in four iterations, so the run uses the bound.
    problem = bernoulli(0.5).problem
    report = estimate_resources(
        problem, algorithm="faster_ae", backend=ReferenceAdapter(), max_iterations=4
    )
    run = estimate(
        problem,
        algorithm="faster_ae",
        backend=ReferenceAdapter(),
        seed=1,
        max_iterations=4,
    )
    assert run.convergence_status == "INSUFFICIENT_BUDGET"
    assert _totals(report) == _totals(run)


def test_circuit_metrics_match_compiled_programs():
    backend = QiskitAdapter()
    table = circuit_metrics(FLAG_CASE, backend, powers=(4, 0, 1, 1))
    assert [c.power for c in table] == [0, 1, 4]
    experiment = Experiment(FLAG_CASE, backend)
    for row in table:
        assert row.metrics == experiment.metrics(row.power)
        assert row.logical_qubits == FLAG_CASE.state_preparation.n_qubits
    assert table[2].metrics["depth"] > table[0].metrics["depth"]
    with pytest.raises(ValueError):
        circuit_metrics(FLAG_CASE, backend, powers=(-1,))


def test_tables_mark_unreported_metrics():
    text = format_circuit_metrics(
        circuit_metrics(bernoulli(0.2).problem, ReferenceAdapter(), (0, 2))
    )
    header, rule, *rows = text.splitlines()
    assert "depth" in header and "2q gates" in header
    assert len(rows) == 2 and all("n/a" in row for row in rows)


def test_summaries_report_resources_and_guarantee():
    problem = bernoulli(0.3).problem
    report = estimate_resources(
        problem, algorithm="iqae", backend=ReferenceAdapter(), max_rounds=5
    )
    assert "worst-case upper bound" in report.summary()
    assert f"{report.oracle_queries:,}" in str(report)
    run = estimate(problem, algorithm="iqae", backend=ReferenceAdapter(), seed=4)
    text = run.summary()
    for fragment in (
        run.guarantee_status,
        run.interval_type,
        f"{run.total_oracle_queries:,}",
        f"{run.total_shots:,}",
    ):
        assert fragment in text


def test_unsupported_algorithm_and_invalid_options_are_rejected():
    problem = bernoulli(0.3).problem
    with pytest.raises(ValueError, match="unsupported algorithm"):
        estimate_resources(problem, algorithm="qpe", backend=ReferenceAdapter())
    with pytest.raises(ValueError):
        estimate_resources(
            problem, algorithm="mlqae", backend=ReferenceAdapter(), shots_per_power=0
        )


def test_command_line(capsys):
    assert (
        main(
            [
                "resources",
                "--problem",
                "bernoulli:0.25",
                "--algorithm",
                "mlqae",
                "--set",
                "shots_per_power=512",
            ]
        )
        == 0
    )
    assert "17,920" in capsys.readouterr().out

    assert (
        main(
            ["circuits", "--problem", "lognormal:2:2:flag", "--powers", "0,2", "--json"]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert [row["power"] for row in rows] == [0, 2]

    assert (
        main(["run", "--problem", "threshold:2:2", "--algorithm", "canonical_qae"]) == 0
    )
    assert "Resources used" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["circuits", "--problem", "bernoulli:2"])
    with pytest.raises(SystemExit):
        main(["resources", "--problem", "nope:1", "--algorithm", "mlqae"])
