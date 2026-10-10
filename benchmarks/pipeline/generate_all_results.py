"""Regenerate measured raw data and processed tables.

The default `smoke` profile is a small run for quick checks. The full
`publication_reference` profile is selected explicitly.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.benchmarks.runner import environment_record, run_manifest
from aeonq.conformance.operator_equivalence import measure_operator_conformance

from alias_stress import generate as generate_alias_stress
from process_extended import write_extended
from process_results import (
    failure_recovery_rows,
    scheduler_comparison_rows,
    read_jsonl,
    summary_rows,
    write_validation_reports,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]


def _capability_rows(conformance: list[dict], raw: list[dict]) -> list[dict]:
    rows = []
    for name, adapter in (
        ("qiskit", QiskitAdapter()),
        ("cirq", CirqAdapter()),
        ("pennylane", PennyLaneAdapter()),
        ("qsharp", QSharpAdapter()),
    ):
        capability = asdict(adapter.capabilities())
        operator_rows = [row for row in conformance if row["backend"] == name]
        execution_rows = [row for row in raw if row["backend"] == name]

        def operator_pass(label: str) -> bool:
            selected = [row for row in operator_rows if row["operator"] == label]
            return bool(selected) and all(row["exact_pass"] for row in selected)

        def algorithm_pass(label: str) -> bool:
            selected = [row for row in execution_rows if row["algorithm"] == label]
            return bool(selected) and all(row["error"] is None for row in selected)

        rows.append(
            {
                "backend": name,
                "sdk_version": next(iter(capability["metadata"].values())),
                "semantic_A": operator_pass("A"),
                "exact_adjoint": operator_pass("A_adjoint"),
                "S0": operator_pass("S0"),
                "Schi": operator_pass("Schi"),
                "Q": operator_pass("Q"),
                "Q_power": operator_pass("Q_power_2"),
                "controlled_Q": operator_pass("controlled_Q"),
                "canonical_qae_local": algorithm_pass("canonical_qae"),
                "iqae_local": algorithm_pass("iqae"),
                "mlqae_local": algorithm_pass("mlqae"),
                "faster_ae_local": algorithm_pass("faster_ae"),
                "checkpointing": "algorithm layer; not tested on this SDK",
                "resource_extraction": "partial"
                if name == "qsharp"
                else "local metrics",
                **{
                    key: value
                    for key, value in capability.items()
                    if key.startswith("supports_") or key in ("max_qubits", "max_shots")
                },
            }
        )
    return rows


def _edge_rows(raw: list[dict], alias: dict) -> list[dict]:
    categories = [
        ("amplitude endpoints", lambda r: r["a_true"] in (0, 1)),
        (
            "near endpoints",
            lambda r: 0 < r["a_true"] <= 1e-6 or 1 - 1e-6 <= r["a_true"] < 1,
        ),
        ("interior amplitude", lambda r: 0.01 <= r["a_true"] <= 0.99),
        (
            "high Grover power",
            lambda r: r["result"] is not None and r["result"]["max_power"] >= 8,
        ),
        (
            "alias resampling",
            lambda r: r["result"] is not None
            and r["result"]["diagnostics"].get("resampling_triggered", False),
        ),
    ]
    rows = []
    for category, predicate in categories:
        matches = [r for r in raw if predicate(r)]
        rows.append(
            {
                "category": category,
                "trial_count": len(matches),
                "successful_runs": sum(r["error"] is None for r in matches),
                "status": "measured" if matches else "not measured in this profile",
            }
        )
    rows.append(
        {
            "category": "injected alias inconsistency and resampling",
            "trial_count": 1,
            "successful_runs": int(bool(alias["resampling_triggered"])),
            "status": "measured separately in alias_stress.json",
        }
    )
    noise_path = ROOT / "results" / "raw" / "noise_study.jsonl"
    if noise_path.exists():
        noise = read_jsonl(noise_path)
        rows.append(
            {
                "category": "noise-model coverage",
                "trial_count": len(noise),
                "successful_runs": sum(r["error"] is None for r in noise),
                "status": "measured separately in noise_study.jsonl",
            }
        )
    for category in (
        "dirty ancilla contract",
        "hardware timeout",
    ) + (() if noise_path.exists() else ("noise-model coverage",)):
        rows.append(
            {
                "category": category,
                "trial_count": 0,
                "successful_runs": 0,
                "status": "not measured in this profile",
            }
        )
    return rows


def run(profile: str, *, reuse_raw: bool = False, workers: int = 1) -> None:
    manifest = json.loads(
        (ROOT / "experiment_manifest.yaml").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported experiment manifest schema")
    names = (
        ("smoke_reference", "smoke_cross_sdk")
        if profile == "smoke"
        else (
            "publication_reference",
            "publication_cross_sdk",
            "publication_scheduler_qiskit",
            "publication_case_study",
            "publication_case_study_sdk",
            "publication_case_study_qsharp",
        )
        if profile == "publication_reference"
        else (profile,)
    )
    raw_paths = []
    for name in names:
        if name not in manifest["profiles"]:
            raise ValueError(f"unknown experiment profile: {name}")
        path = ROOT / "results" / "raw" / f"{name}.jsonl"
        if not reuse_raw or not path.exists():
            run_manifest(manifest["profiles"][name], path, max_workers=workers)
        raw_paths.append(path)
    from aeonq.benchmarks.validation import validate_manifest

    raw = []
    for name, path in zip(names, raw_paths):
        rows = read_jsonl(path)
        validate_manifest(rows, manifest["profiles"][name])
        raw.extend(rows)
    summary = summary_rows(raw)
    comparisons = scheduler_comparison_rows(raw)
    conformance = measure_operator_conformance()
    alias = generate_alias_stress()
    alias_path = ROOT / "results" / "raw" / "alias_stress.json"
    alias_path.write_text(json.dumps(alias, indent=2), encoding="utf-8")
    compiler_path = ROOT / "results" / "raw" / "phase10_compiler.csv"
    if not compiler_path.exists():
        raise FileNotFoundError(
            "run benchmarks/compiler_comparison.py before generating figures"
        )
    cross_sdk_compiler_path = (
        ROOT / "results" / "raw" / "phase12_cross_sdk_compiler.csv"
    )
    if not cross_sdk_compiler_path.exists():
        raise FileNotFoundError(
            "run benchmarks/compiler_cross_sdk.py before generating tables"
        )

    processed = ROOT / "results" / "processed"
    tables = ROOT / "results" / "tables"
    write_csv(processed / "summary.csv", summary)
    if comparisons:
        write_csv(processed / "scheduler_comparison.csv", comparisons)
    write_csv(processed / "operator_conformance.csv", conformance)
    write_validation_reports(summary, conformance, processed)
    write_csv(tables / "table_1_capabilities.csv", _capability_rows(conformance, raw))
    write_csv(tables / "table_2_semantic_conformance.csv", conformance)
    write_csv(tables / "table_3_edge_cases.csv", _edge_rows(raw, alias))
    write_csv(tables / "table_4_algorithm_accuracy.csv", summary)
    with compiler_path.open(encoding="utf-8") as stream:
        import csv

        compiler_rows = list(csv.DictReader(stream))
    write_csv(tables / "table_5_compiler_resources.csv", compiler_rows)
    with cross_sdk_compiler_path.open(newline="", encoding="utf-8") as stream:
        cross_sdk_compiler_rows = list(csv.DictReader(stream))
    write_csv(tables / "table_5b_cross_sdk_compiler.csv", cross_sdk_compiler_rows)
    if comparisons:
        write_csv(tables / "table_6_scheduler_comparison.csv", comparisons)
    cross_sdk = [
        row
        for row in summary
        if row["backend"] in {"qiskit", "cirq", "pennylane", "qsharp"}
        and not row["profile"].startswith("publication_case_study")
    ]
    if cross_sdk:
        write_csv(tables / "table_7_cross_sdk.csv", cross_sdk)
    write_csv(tables / "table_8_failure_recovery.csv", failure_recovery_rows())
    case_study = [
        row for row in summary if row["profile"].startswith("publication_case_study")
    ]
    if case_study:
        write_csv(tables / "table_16_case_study.csv", case_study)
    write_extended(
        ROOT, tables, write_csv, require_complete=profile == "publication_reference"
    )

    environment = environment_record()
    (ROOT / "results" / "environment").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "environment" / "last_run.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "raw_files": [str(p.relative_to(ROOT)) for p in raw_paths],
                "trial_count": len(raw),
                "failed_trials": sum(row["error"] is not None for row in raw),
                "environment": environment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="reference-trial worker threads; SDK trials stay serial",
    )
    arguments = parser.parse_args()
    run(arguments.profile, reuse_raw=arguments.reuse_raw, workers=arguments.workers)
