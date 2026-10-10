"""Small executable checks for independent benchmark truth and artifact processing."""

import json
from math import isclose

import pytest

from aeonq.benchmarks.problems import (
    discrete_distribution,
    independent_failures,
    uniform_marked,
)
from aeonq.benchmarks.runner import build_problem, map_trials, run_manifest
from aeonq.reference import reference_amplitude
from aeonq.conformance.operator_equivalence import measure_operator_conformance
from benchmarks.pipeline.process_results import scheduler_comparison_rows


@pytest.mark.parametrize(
    "case",
    [
        uniform_marked(3, (1, 3, 7)),
        discrete_distribution((0.1, 0.2, 0.3, 0.4), (1, 3)),
        discrete_distribution((0.0, 0.0, 0.0, 1.0), (3,)),
        independent_failures((0.1, 0.2, 0.3)),
        independent_failures((0.0, 0.0)),
    ],
)
def test_classical_truth_agrees_with_independent_statevector(case):
    assert isclose(reference_amplitude(case.problem), case.a_true, abs_tol=1e-12)


def test_random_problem_is_reproducible():
    spec = {"kind": "random_distribution", "problem_seed": 42, "marked_count": 2}
    assert build_problem(spec) == build_problem(spec)
    assert 0 <= build_problem(spec).a_true <= 1


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "distribution", "probabilities": [0.1, 0.2, 0.3, 0.3], "marked": [1]},
        {"kind": "uniform_marked", "n_qubits": 2, "marked": [4]},
        {"kind": "reliability", "failure_probabilities": [1.2, 0.3]},
    ],
)
def test_invalid_benchmark_problem_rejected(spec):
    with pytest.raises(ValueError):
        build_problem(spec)


def test_manifest_results_are_saved_with_traceable_seed_and_truth(tmp_path):
    manifest = {
        "schema_version": 1,
        "profile": "test",
        "base_seed": 341,
        "repetitions": 2,
        "backends": ["reference"],
        "problems": [{"kind": "reliability", "failure_probabilities": [0.1, 0.2]}],
        "algorithms": {
            "mlqae": {
                "method": "mlqae",
                "options": {
                    "powers": [0, 1],
                    "shots_per_power": 16,
                    "epsilon": 0.2,
                    "alpha": 0.05,
                },
            }
        },
    }
    target = tmp_path / "raw.jsonl"
    rows = run_manifest(manifest, target)
    stored = [json.loads(line) for line in target.read_text().splitlines()]
    assert stored == json.loads(json.dumps(rows))
    assert len({row["seed"] for row in rows}) == 2
    assert all(row["environment"]["source_snapshot_sha256"] for row in rows)
    assert all(row["error"] is None and row["result"] is not None for row in rows)
    assert all(isclose(row["a_true"], 0.28, abs_tol=1e-12) for row in rows)
    assert all(row["environment"]["python"] for row in rows)


def test_threaded_reference_trials_keep_order_and_seeded_results(tmp_path):
    manifest = {
        "schema_version": 1,
        "profile": "thread-check",
        "base_seed": 712,
        "repetitions": 12,
        "backends": ["reference"],
        "problems": [{"kind": "bernoulli", "amplitude": 0.25}],
        "algorithms": {
            "mlqae": {
                "method": "mlqae",
                "options": {
                    "powers": [0, 1, 2],
                    "shots_per_power": 32,
                    "epsilon": 0.1,
                    "alpha": 0.05,
                },
            }
        },
    }
    serial = run_manifest(manifest, tmp_path / "serial.jsonl", max_workers=1)
    threaded = run_manifest(manifest, tmp_path / "threaded.jsonl", max_workers=4)
    assert [row["seed"] for row in serial] == [row["seed"] for row in threaded]
    for left, right in zip(serial, threaded):
        assert left["result"]["estimate"] == right["result"]["estimate"]
        assert (
            left["result"]["diagnostics"]["observations"]
            == (right["result"]["diagnostics"]["observations"])
        )
    with pytest.raises(ValueError):
        run_manifest(manifest, tmp_path / "bad.jsonl", max_workers=0)


def test_resumed_manifest_run_matches_an_uninterrupted_one(tmp_path):
    manifest = {
        "schema_version": 1,
        "profile": "resume-check",
        "base_seed": 523,
        "repetitions": 6,
        "backends": ["reference"],
        "problems": [{"kind": "bernoulli", "amplitude": 0.25}],
        "algorithms": {
            "mlqae": {
                "method": "mlqae",
                "options": {
                    "powers": [0, 1, 2],
                    "shots_per_power": 32,
                    "epsilon": 0.1,
                    "alpha": 0.05,
                },
            }
        },
    }
    complete = run_manifest(manifest, tmp_path / "complete.jsonl")
    target = tmp_path / "resumed.jsonl"
    run_manifest(manifest, target)
    lines = target.read_text().splitlines()
    # Interrupted after three trials, partway through writing the fourth.
    target.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40])
    added = run_manifest(manifest, target, resume=True)
    stored = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["trial"] for row in added] == [3, 4, 5]
    assert [row["trial"] for row in stored] == list(range(6))
    assert [row["result"]["estimate"] for row in stored] == [
        row["result"]["estimate"] for row in complete
    ]
    assert run_manifest(manifest, target, resume=True) == []
    assert len(run_manifest(manifest, target)) == 6
    in_processes = run_manifest(manifest, tmp_path / "processes.jsonl", processes=3)
    assert [(row["trial"], row["result"]["estimate"]) for row in in_processes] == [
        (row["trial"], row["result"]["estimate"]) for row in complete
    ]


def test_map_trials_keeps_task_order_across_worker_processes():
    tasks = [(3, 1), (0, 5), (2, 2), (7, 0), (1, 9)]
    assert list(map_trials(sum, tasks, workers=3)) == [sum(t) for t in tasks]
    assert list(map_trials(sum, tasks)) == [sum(t) for t in tasks]
    with pytest.raises(ValueError):
        list(map_trials(sum, tasks, workers=0))


def test_manifest_reuses_each_local_sdk_across_seeded_trials(tmp_path):
    manifest = {
        "schema_version": 1,
        "profile": "sdk-reuse",
        "base_seed": 991,
        "repetitions": 2,
        "backends": ["qiskit", "cirq", "pennylane", "qsharp"],
        "problems": [{"kind": "bernoulli", "amplitude": 0.25}],
        "algorithms": {
            "mlqae": {
                "method": "mlqae",
                "options": {
                    "powers": [0],
                    "shots_per_power": 16,
                    "epsilon": 0.2,
                    "alpha": 0.05,
                },
            }
        },
    }
    rows = run_manifest(manifest, tmp_path / "sdk.jsonl")
    assert len(rows) == 8
    assert all(row["error"] is None for row in rows)
    assert {row["backend"] for row in rows} == set(manifest["backends"])
    for row in rows:
        metrics = row["result"]["resource_metrics"]
        assert "total_compiled_depth" in metrics
        if metrics["depth"] is not None:
            assert metrics["total_compiled_depth"] == metrics["depth"]


def test_operator_conformance_export_reports_all_four_sdk_distances():
    rows = measure_operator_conformance()
    assert {row["backend"] for row in rows} == {"qiskit", "cirq", "pennylane", "qsharp"}
    assert {row["operator"] for row in rows} >= {
        "A",
        "A_adjoint",
        "S0",
        "Schi",
        "Q",
        "Q_power_2",
        "controlled_Q",
    }
    assert all(row["exact_pass"] for row in rows)
    assert all(row["exact_distance"] < row["tolerance"] for row in rows)


def test_independent_scheduler_report_keeps_profiles_separate_and_counts_depth():
    def row(profile, method, estimate, queries, depth):
        return {
            "profile": profile,
            "backend": "reference",
            "case_id": "same-case",
            "trial": 0,
            "seed": 42,
            "method_label": method,
            "a_true": 0.25,
            "result": {
                "estimate": estimate,
                "total_oracle_queries": queries,
                "total_shots": 10,
                "max_power": 1,
                "total_circuit_evaluations": 1,
                "wall_clock_time": 0.01,
                "interval_type": "confidence interval",
                "confidence_or_credible_interval": [0.2, 0.3],
                "resource_metrics": {"total_compiled_depth": depth},
            },
        }

    rows = [
        row("first", "iqae_fixed", 0.25, 10, 2),
        row("first", "iqae_fisher", 0.26, 6, 3),
        row("second", "iqae_fixed", 0.25, 100, 20),
        row("second", "iqae_fisher", 0.27, 80, 30),
    ]
    report = scheduler_comparison_rows(rows)
    first = next(
        r for r in report if r["profile"] == "first" and r["method"] == "iqae_fisher"
    )
    second = next(
        r for r in report if r["profile"] == "second" and r["method"] == "iqae_fisher"
    )
    assert first["independent_n"] == second["independent_n"] == 1
    assert first["mean_query_difference"] == -4
    assert second["mean_query_difference"] == -20
    assert first["mean_total_compiled_depth"] == 3
    assert second["mean_total_compiled_depth"] == 30
