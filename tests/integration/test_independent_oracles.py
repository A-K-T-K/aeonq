"""Every operator source meets the closed forms and the hand-written oracle."""

import pytest

from aeonq.conformance.independent import (
    independent_rows,
    undetected_negative_controls,
)


@pytest.fixture(scope="module")
def rows():
    return independent_rows()


def test_every_source_passes_every_independent_check(rows):
    failures = [
        (r["case"], r["subject"], r["operator"], r["check"], r["error"])
        for r in rows
        if r["kind"] != "negative_control" and not r["passed"]
    ]
    assert failures == []


def test_all_sources_are_covered(rows):
    subjects = {r["subject"] for r in rows}
    for sdk in ("qiskit", "cirq", "pennylane", "qsharp"):
        assert f"{sdk}:handwritten" in subjects
        assert f"{sdk}:adapter" in subjects
    for sdk in ("qiskit", "cirq", "pennylane"):
        assert f"numpy:{sdk}-program" in subjects
    assert "ir:reference" in subjects


def test_every_negative_control_is_rejected(rows):
    assert undetected_negative_controls(rows) == []


def test_a_dropped_global_phase_survives_count_level_laws(rows):
    # -Q gives the same A·Q^k distributions and the same Q², so only the
    # operator-level checks on Q and controlled Q can see it.
    dropped = [
        r
        for r in rows
        if r["subject"] == "negative:Q_without_global_phase"
        and r["compared_with"] == "closed form"
    ]
    by_check = {}
    for r in dropped:
        by_check.setdefault(r["check"], set()).add(r["passed"])
    assert by_check["amplification_sin2_(2k+1)theta"] == {True}
    assert by_check["rotation_4theta"] == {True}
    assert by_check["rotation_2theta"] == {False}
    assert by_check["identity_then_rotation"] == {False}
