"""The native canonical MLE comparison reuses the original trial evidence."""

from pathlib import Path


def test_native_mle_view_changes_only_point_estimate_and_library(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/pipeline")
    )
    from process_extended import native_point_estimate_rows

    row = {
        "method": "canonical_qae",
        "library": "qiskit_algorithms",
        "error": None,
        "estimate": 0.22,
        "mle": 0.25,
        "interval": [0.24, 0.26],
        "queries": 31,
        "seed": 12,
    }
    output = native_point_estimate_rows([row])
    assert output[0] is row
    assert output[1] == {**row, "library": "qiskit_algorithms_mle", "estimate": 0.25}
    assert row["estimate"] == 0.22
