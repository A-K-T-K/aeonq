from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aeonq.benchmarks.evidence import source_snapshot, trial_seed, verify_source_archive
from aeonq.benchmarks.runner import environment_record, open_trial_log
from aeonq.benchmarks.validation import validate_records


def test_extended_validation_works_outside_workspace(tmp_path):
    # Benchmark CLIs start with benchmarks/ on sys.path, not the project root.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aeonq.benchmarks.validation import extended_expected; "
            "keys, rows = extended_expected('noise_study', 1); "
            "assert len(rows) == 30; assert len(keys) == 4",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_full_analysis_rejects_absent_studies(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "pipeline")
    )
    from process_extended import write_extended

    emitted = []
    with pytest.raises(ValueError, match="missing studies"):
        write_extended(
            tmp_path,
            tmp_path,
            lambda *args: emitted.append(args),
            require_complete=True,
        )
    assert emitted == []


def test_seeds_separate_every_condition_and_reproduce_across_order():
    identities = [
        ("study", case, method, backend, trial)
        for case in ("a", "b")
        for method in ("iqae", "mlqae")
        for backend in ("reference", "cirq")
        for trial in range(50)
    ]
    seeds = [trial_seed(*identity) for identity in identities]
    assert len(set(seeds)) == len(seeds)
    assert seeds[::-1] == [trial_seed(*identity) for identity in reversed(identities)]
    assert trial_seed(*identities[0], base_seed=1) != seeds[0]
    assert trial_seed(*identities[0], condition=1e-3) != seeds[0]


def test_source_archive_preserves_hashed_bytes():
    verify_source_archive(source_snapshot())
    with pytest.raises(ValueError, match="provenance"):
        verify_source_archive({"source_snapshot_sha256": "fake"})


def population():
    rows = [
        {
            "trial": i,
            "seed": 50 + i,
            "configuration": {"shots": 20},
            "a_true": 0.25,
            "error": None,
            "result": {"estimate": 0.24},
            "environment": environment_record(),
        }
        for i in range(2)
    ]
    expected = {
        (i,): {"seed": 50 + i, "configuration": {"shots": 20}, "a_true": 0.25}
        for i in range(2)
    }
    return rows, expected


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "unexpected",
        "failure",
        "configuration",
        "seed",
        "truth",
        "provenance",
        "nan",
    ],
)
def test_invalid_population_is_rejected_before_analysis(defect):
    rows, expected = population()
    rows = deepcopy(rows)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif defect == "unexpected":
        rows[0]["trial"] = 99
    elif defect == "failure":
        rows[0]["error"] = "backend failed"
    elif defect == "configuration":
        rows[0]["configuration"]["shots"] = 5
    elif defect == "seed":
        rows[0]["seed"] = 999
    elif defect == "truth":
        rows[0]["a_true"] = 0.4
    elif defect == "provenance":
        rows[0]["environment"].pop("source_snapshot_sha256")
    elif defect == "nan":
        rows[0]["result"]["estimate"] = float("nan")
    with pytest.raises(ValueError):
        validate_records(rows, ("trial",), expected)


def test_complete_population_and_partial_resume_are_distinct():
    rows, expected = population()
    validate_records(rows, ("trial",), expected)
    validate_records(rows[:1], ("trial",), expected, complete=False)


def test_resume_rejects_corrupt_interior_and_duplicate_records(tmp_path):
    path = tmp_path / "raw.jsonl"
    row = {"trial": 0, "error": None, "environment": environment_record()}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        open_trial_log(path, ("trial",), resume=True)
    path.write_text("broken\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="malformed"):
        open_trial_log(path, ("trial",), resume=True)
