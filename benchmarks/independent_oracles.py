"""Independent oracle checks: closed forms, hand-written circuits, NumPy simulator.

See ``aeonq.conformance.independent`` for what is compared with what.

Writes:
  benchmarks/results/raw/independent_oracles.csv           one row per check
  benchmarks/results/raw/independent_oracles.environment.json
  benchmarks/results/tables/table_18_independent_oracles.csv   per source
  benchmarks/results/processed/independent_oracles.md      narrative summary

Run from the repository root:  python benchmarks/independent_oracles.py
"""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import warnings

from aeonq.benchmarks.runner import environment_record
from aeonq.conformance.independent import (
    independent_rows,
    undetected_negative_controls,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks" / "results" / "raw"
TABLES = ROOT / "benchmarks" / "results" / "tables"
PROCESSED = ROOT / "benchmarks" / "results" / "processed"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["kind"] != "negative_control":
            groups[row["subject"]].append(row)
    table = []
    for subject, group in groups.items():
        closed = [r for r in group if r["kind"] == "closed_form"]
        cross = [r for r in group if r["kind"] == "cross"]
        table.append(
            {
                "subject": subject,
                "closed_form_checks": len(closed),
                "closed_form_passed": sum(r["passed"] for r in closed),
                "entrywise_vs_numpy_oracle": len(cross),
                "entrywise_passed": sum(r["passed"] for r in cross),
                "worst_closed_form_error": max(
                    (r["error"] for r in closed), default=None
                ),
                "worst_entrywise_error": max((r["error"] for r in cross), default=None),
            }
        )
    return table


def report(rows: list[dict], table: list[dict]) -> str:
    negatives = [r for r in rows if r["kind"] == "negative_control"]
    variants = sorted({(r["case"], r["subject"]) for r in negatives})
    missed_closed = undetected_negative_controls(rows, closed_form_only=True)
    missed = undetected_negative_controls(rows)
    lines = [
        "# Independent oracle checks",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by "
        "`benchmarks/independent_oracles.py`. Oracles come from "
        "`aeonq.oracles`, which imports no aeonq IR, synthesis, reference or "
        "adapter code. Closed forms: amplitude a, Q as a rotation by 2θ on "
        "the good/bad plane (global phase included), Q² by 4θ, controlled Q "
        "as identity ⊕ rotation, and P_good(Q^k A|0>) = sin²((2k+1)θ) for "
        "k = 0..4.",
        "",
        "| source | closed-form passed | entrywise vs NumPy oracle passed | "
        "worst closed-form error | worst entrywise error |",
        "|---|---|---|---|---|",
    ]
    for t in table:
        cross = (
            f"{t['entrywise_passed']} / {t['entrywise_vs_numpy_oracle']}"
            if t["entrywise_vs_numpy_oracle"]
            else "(is the oracle)"
            if t["subject"] == "numpy:handwritten"
            else "n/a"
        )
        entry = t["worst_entrywise_error"]
        lines.append(
            f"| `{t['subject']}` | {t['closed_form_passed']} / "
            f"{t['closed_form_checks']} | {cross} | "
            f"{t['worst_closed_form_error']:.2e} | "
            f"{'' if entry is None else f'{entry:.2e}'} |"
        )
    lines += [
        "",
        "## Negative controls",
        "",
        f"{len(variants)} faulty variants of the oracle (variants identical "
        "to the oracle for a case are omitted). Rejected by some check: "
        f"{len(variants) - len(missed)}; rejected by a closed form alone: "
        f"{len(variants) - len(missed_closed)}.",
        "",
        "| case | variant | check | result |",
        "|---|---|---|---|",
    ]
    for r in negatives:
        lines.append(
            f"| {r['case']} | {r['subject'].removeprefix('negative:')} | "
            f"{r['check']} ({r['operator']}) | "
            f"{'passed' if r['passed'] else 'rejected'} |"
        )
    if missed_closed:
        lines += [
            "",
            "Variants that no closed form rejects: "
            + ", ".join(f"{v} on {c}" for c, v in missed_closed)
            + ". The amplitude and amplification laws depend on A only through "
            "P_good, and at a = 1/2 every sin²((2k±1)π/4) equals 1/2, so a sign "
            "change in A is invisible to them there; the entrywise comparison "
            "with the hand-written oracle rejects it.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("ignore")
    started = perf_counter()
    rows = independent_rows()
    table = summarize(rows)
    _write_csv(RAW / "independent_oracles.csv", rows)
    _write_csv(TABLES / "table_18_independent_oracles.csv", table)
    environment = environment_record()
    environment["seconds"] = perf_counter() - started
    (RAW / "independent_oracles.environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "independent_oracles.md").write_text(
        report(rows, table), encoding="utf-8"
    )
    failed = [r for r in rows if r["kind"] != "negative_control" and not r["passed"]]
    print(f"{len(rows)} checks, {len(failed)} unexpected failures", flush=True)


if __name__ == "__main__":
    main()
