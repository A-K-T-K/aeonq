"""Fault rejections must be retained without reporting survivor-only accuracy."""

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from aeonq.benchmarks.validation import (
    read_records,
    validate_mutation_impact,
    validate_mutation_battery,
)

RAW = Path(__file__).resolve().parents[2] / "benchmarks/results/raw"


def evidence():
    rows = read_records(RAW / "adapter_mutation_impact.jsonl")
    with (RAW / "adapter_mutation_impact.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        return rows, list(csv.DictReader(stream))


def test_mutation_accuracy_is_suppressed_when_a_group_has_rejections():
    rows, summaries = evidence()
    validate_mutation_impact(rows, summaries)
    rejected = next(row for row in summaries if int(row["crashes"]) > 0)
    assert rejected["mean_abs_error"] == ""
    bad = deepcopy(summaries)
    next(row for row in bad if row == rejected)["mean_abs_error"] = 0.1
    with pytest.raises(ValueError, match="only successful runs"):
        validate_mutation_impact(rows, bad)


@pytest.mark.parametrize("duplicate", [False, True])
def test_mutation_analysis_requires_complete_unique_trials(duplicate):
    rows, summaries = evidence()
    rows = rows + [rows[-1]] if duplicate else rows[:-1]
    with pytest.raises(ValueError, match="duplicate|missing"):
        validate_mutation_impact(rows, summaries)


@pytest.mark.parametrize("duplicate", [False, True])
def test_mutation_battery_requires_every_sdk_and_fault(duplicate):
    with (RAW / "adapter_mutation.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_mutation_battery(rows)
    rows = rows + [rows[-1]] if duplicate else rows[:-1]
    with pytest.raises(ValueError, match="duplicate|missing"):
        validate_mutation_battery(rows)
