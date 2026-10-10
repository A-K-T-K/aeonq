"""Translator-fidelity study: real SDK converters on amplitude-estimation circuits.

Every operator of every conformance case (A, A†, S0, Sχ, Q, Q², controlled Q
and the measured circuits A·Q^k) is lowered natively in Qiskit, Cirq or
PennyLane and pushed through each existing conversion path listed in
``aeonq.conformance.translation.PATHS``. The translated program is evaluated
with the target SDK alone. A one-gate probe of each OpenQASM importer then
locates the gates responsible for any difference.

Writes:
  benchmarks/results/raw/translator_fidelity.csv          one row per program
  benchmarks/results/raw/importer_gate_probe.csv          one row per gate
  benchmarks/results/raw/translator_fidelity.environment.json
  benchmarks/results/tables/table_17_translator_fidelity.csv   per path
  benchmarks/results/processed/translator_fidelity.md     narrative summary

Run from the repository root:  python benchmarks/translator_fidelity.py
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import warnings

from aeonq.benchmarks.evidence import trial_seed
from aeonq.benchmarks.runner import environment_record
from aeonq.reference import matrix
from aeonq.conformance.translation import (
    COUNT_SHOTS,
    PATHS,
    UNAVAILABLE_PATHS,
    converter_versions,
    evaluate_translation,
    importer_gate_probe,
    native_qiskit_sources,
    study_cases,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks" / "results" / "raw"
TABLES = ROOT / "benchmarks" / "results" / "tables"
PROCESSED = ROOT / "benchmarks" / "results" / "processed"


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_study(path_filter: str | None = None) -> list[dict]:
    rows = []
    for case, nodes in study_cases():
        for node_name, node in nodes.items():
            for path in PATHS:
                if path_filter and path_filter not in path.name:
                    continue
                seed = trial_seed("translator_fidelity", case, node_name, path.name, 0)
                started = perf_counter()
                row = evaluate_translation(
                    path,
                    case,
                    node_name,
                    node,
                    seed=seed,
                    prepared_state=(
                        matrix(nodes["A"])[:, 0] if node_name == "Q" else None
                    ),
                )
                row["seconds"] = perf_counter() - started
                rows.append(row)
            for name, reason in UNAVAILABLE_PATHS.items():
                if path_filter and path_filter not in name:
                    continue
                source, intermediate, target = name.split(">")
                rows.append(
                    {
                        "case": case,
                        "node": node_name,
                        "qubits": node.n_qubits,
                        "path": name,
                        "source": source,
                        "intermediate": intermediate,
                        "target": target,
                        "status": "unavailable",
                        "error": reason,
                    }
                )
        print(f"case {case} done", flush=True)
    return rows


def run_native() -> list[dict]:
    """Qiskit-native Grover operators (no aeonq lowering) on the Qiskit paths."""
    rows = []
    for case, name, program, node in native_qiskit_sources():
        for path in PATHS:
            if path.source != "qiskit":
                continue
            seed = trial_seed("translator_fidelity_native", case, name, path.name, 0)
            row = evaluate_translation(
                path, case, name, node, seed=seed, source_program=program
            )
            row["source_matches_reference"] = (
                row.get("source_vs_ir_max_error", 1.0) <= 1e-9
            )
            rows.append(row)
    print("native Qiskit sources done", flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per-path counts. Rates are over programs the path could evaluate."""
    table = []
    for name in dict.fromkeys(row["path"] for row in rows):
        group = [row for row in rows if row["path"] == name]
        evaluated = [row for row in group if row["status"] == "evaluated"]
        statuses = Counter(row["status"] for row in group)
        table.append(
            {
                "path": name,
                "programs": len(group),
                "evaluated": len(evaluated),
                "export_failed": statuses["export_failed"],
                "import_failed": statuses["import_failed"],
                "unitary_failed": statuses["unitary_failed"],
                "unavailable": statuses["unavailable"],
                "equal_exact": sum(bool(r.get("equal_exact")) for r in evaluated),
                "equal_up_to_global_phase_only": sum(
                    bool(r.get("equal_up_to_global_phase")) and not r.get("equal_exact")
                    for r in evaluated
                ),
                "not_equal_up_to_phase": sum(
                    not r.get("equal_up_to_global_phase") for r in evaluated
                ),
                "qubit_order_changed": sum(
                    r.get("qubit_order") not in ("identity", None) for r in evaluated
                ),
                "controlled_checked": sum(
                    r.get("equal_when_controlled") is not None for r in evaluated
                ),
                "controlled_failed": sum(
                    r.get("equal_when_controlled") is False for r in evaluated
                ),
                "controlled_unsupported": sum(
                    r.get("equal_when_controlled") is None for r in evaluated
                ),
                "count_pass": sum(r.get("count_pass") is True for r in evaluated),
                "operator_pass": sum(bool(r.get("operator_pass")) for r in evaluated),
                "count_invisible_failures": sum(
                    bool(r.get("count_invisible_failure")) for r in evaluated
                ),
                "worst_data_angle_residual": max(
                    (
                        r["data_angle_max_residual"]
                        for r in evaluated
                        if isinstance(r.get("data_angle_max_residual"), float)
                        and r.get("data_angles_reexpressed", 0)
                        < r.get("data_angles_checked", 0)
                    ),
                    default=None,
                ),
            }
        )
    return table


def _first_errors(rows: list[dict], status: str) -> dict[str, str]:
    seen: dict[str, str] = {}
    for row in rows:
        if row["status"] == status and row["path"] not in seen:
            seen[row["path"]] = row["error"]
    return seen


def report(
    rows: list[dict], table: list[dict], probe: list[dict], native: list[dict]
) -> str:
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    invisible = [r for r in evaluated if r.get("count_invisible_failure")]
    lines = [
        "# Translator fidelity on amplitude-estimation circuits",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by "
        "`benchmarks/translator_fidelity.py`. Every program was lowered by a "
        "verified aeonq adapter and then converted only by third-party code; "
        "the translated program was evaluated only with the target SDK. "
        "Operator comparisons are against the source program's own matrix, "
        f"entry by entry; counts use {COUNT_SHOTS} shots of U|0...0> and a "
        "G-test at p >= 1e-3.",
        "",
        f"* programs attempted: {len(rows)} "
        f"({sum(r['status'] == 'unavailable' for r in rows)} on unavailable paths)",
        f"* programs translated and evaluated: {len(evaluated)}",
        f"* exactly equal: {sum(bool(r.get('equal_exact')) for r in evaluated)}",
        "* equal only up to global phase: "
        + str(
            sum(
                bool(r.get("equal_up_to_global_phase")) and not r.get("equal_exact")
                for r in evaluated
            )
        ),
        "* not equal even up to global phase: "
        + str(sum(not r.get("equal_up_to_global_phase") for r in evaluated)),
        "* failed once controlled in the target SDK: "
        + str(sum(r.get("equal_when_controlled") is False for r in evaluated)),
        f"* count check passed while an operator check failed: {len(invisible)}",
        "",
        "## Per path",
        "",
        "| path | evaluated / attempted | exact | phase only | not equal | "
        "controlled failed | counts pass | count-invisible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(
            f"| `{t['path']}` | {t['evaluated']} / {t['programs']} | "
            f"{t['equal_exact']} | {t['equal_up_to_global_phase_only']} | "
            f"{t['not_equal_up_to_phase']} | {t['controlled_failed']} | "
            f"{t['count_pass']} | {t['count_invisible_failures']} |"
        )
    lines += ["", "## Conversion failures (first message per path)", ""]
    for status in ("export_failed", "import_failed", "unitary_failed", "unavailable"):
        for path, error in _first_errors(rows, status).items():
            lines.append(f"* `{path}` {status}: {error}")
    lines += [
        "",
        "## Count-invisible failures",
        "",
        "| case | node | path | max entry error | global phase offset | "
        "controlled equal |",
        "|---|---|---|---|---|---|",
    ]
    for r in invisible:
        offset = r.get("global_phase_offset")
        lines.append(
            f"| {r['case']} | {r['node']} | `{r['path']}` | "
            f"{r['max_abs_error']:.3g} | "
            f"{'' if offset is None else f'{offset:+.4f}'} | "
            f"{r.get('equal_when_controlled')} |"
        )
    canonical = [
        r
        for r in evaluated
        if "canonical_qae_translated" in r
        and abs(r["canonical_qae_translated"] - r["canonical_qae_source"]) > 1e-9
    ]
    lines += [
        "",
        "## Consequence for canonical QAE",
        "",
        "Most likely canonical-QAE estimate (5 evaluation qubits, exact phase "
        "estimation on the eigenphases of Q, global phase included) from the "
        "source Q and from the translated Q. Rows where they differ:",
        "",
        "| case | path | estimate from source Q | estimate from translated Q |",
        "|---|---|---|---|",
    ]
    for r in canonical:
        lines.append(
            f"| {r['case']} | `{r['path']}` | {r['canonical_qae_source']:.4f} | "
            f"{r['canonical_qae_translated']:.4f} |"
        )
    native_evaluated = [r for r in native if r["status"] == "evaluated"]
    lines += [
        "",
        "## Qiskit-native Grover operators",
        "",
        "Grover operators built by Qiskit's `grover_operator` (hand-written A "
        "and oracle) and by `qiskit-algorithms`' `EstimationProblem`, with no "
        "aeonq lowering, sent through the Qiskit-source paths. "
        f"Evaluated {len(native_evaluated)} of {len(native)}; phase only "
        + str(
            sum(
                bool(r.get("equal_up_to_global_phase")) and not r.get("equal_exact")
                for r in native_evaluated
            )
        )
        + "; failed once controlled "
        + str(sum(r.get("equal_when_controlled") is False for r in native_evaluated))
        + "; count-invisible "
        + str(sum(bool(r.get("count_invisible_failure")) for r in native_evaluated))
        + ". Sources that already differ from the reference before conversion: "
        + (
            ", ".join(
                sorted(
                    {
                        f"{r['case']} {r['node']}"
                        for r in native
                        if r.get("source_matches_reference") is False
                    }
                )
            )
            or "none"
        )
        + ".",
        "",
        "| case | program | path | status | exact | phase only | controlled equal "
        "| counts pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in native:
        lines.append(
            f"| {r['case']} | {r['node']} | `{r['path']}` | {r['status']} | "
            f"{r.get('equal_exact', '')} | "
            f"{bool(r.get('equal_up_to_global_phase')) and not r.get('equal_exact')} | "
            f"{r.get('equal_when_controlled', '')} | {r.get('count_pass', '')} |"
        )
    deviations = [
        p for p in probe if p["status"] != "imported" or not p.get("equal_exact")
    ]
    lines += [
        "",
        "## Importer gate probe: deviations from the gate formula",
        "",
        f"{len(probe)} one-gate programs; {len(deviations)} deviate or are "
        "rejected. All others match the formula exactly.",
        "",
        "| importer | statement | status | equal up to phase | phase offset |",
        "|---|---|---|---|---|",
    ]
    for p in deviations:
        offset = p.get("global_phase_offset")
        lines.append(
            f"| {p['importer']} | `{p['statement']}` | {p['status']} | "
            f"{p.get('equal_up_to_global_phase', '')} | "
            f"{'' if offset is None else f'{offset:+.4f}'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", help="only paths whose name contains this text")
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore")
    started = perf_counter()
    rows = run_study(arguments.paths)
    native = run_native()
    probe = importer_gate_probe()
    table = summarize(rows)
    _write_csv(RAW / "translator_fidelity.csv", rows)
    _write_csv(RAW / "importer_gate_probe.csv", probe)
    _write_csv(RAW / "translator_native_grover.csv", native)
    _write_csv(TABLES / "table_17_translator_fidelity.csv", table)
    environment = environment_record()
    environment["converters"] = converter_versions()
    environment["seconds"] = perf_counter() - started
    (RAW / "translator_fidelity.environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "translator_fidelity.md").write_text(
        report(rows, table, probe, native), encoding="utf-8"
    )
    print(f"{len(rows)} programs, {len(probe)} probe gates", flush=True)


if __name__ == "__main__":
    main()
