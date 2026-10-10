import csv

from benchmarks.compiler_cross_sdk import run


def test_cross_sdk_compiler_benchmark_records_supported_and_unsupported(tmp_path):
    output = tmp_path / "compiler.csv"
    rows = run(output, powers=(0, 1))
    assert len(rows) == 36
    assert len(list(csv.DictReader(output.open(newline="", encoding="utf-8")))) == 36
    assert all(row["executable_success"] for row in rows if row["sdk"] != "qsharp")
    qsharp = [row for row in rows if row["sdk"] == "qsharp"]
    assert all(
        row["executable_success"] for row in qsharp if row["pipeline"] != "ae_aware"
    )
    assert all(
        not row["executable_success"] and "depth metric required" in row["error"]
        for row in qsharp
        if row["pipeline"] == "ae_aware"
    )
