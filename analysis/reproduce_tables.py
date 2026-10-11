"""Compute the evaluation tables and summary quantities from the stored results.

Every table and quantity is computed from the per-trial records in
``benchmarks/results/raw``. The conformance battery, the operator-equivalence
check, the disjoint-cube bound and the Faster AE shot counts are evaluated
directly. Study parameters (equivalence margins, estimator settings, check
counts and SDK versions) are read from the benchmark pipeline, the trial
records and the battery itself.

Usage, from the repository root:

    python analysis/reproduce_tables.py
    python analysis/reproduce_tables.py --no-live      # omit the SDK conformance battery
    python analysis/reproduce_tables.py --regenerate   # rerun all studies first
"""

import argparse
import csv
import importlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import version
from math import log, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks" / "results" / "raw"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "benchmarks" / "pipeline"))

# Equivalence margins as defined by the benchmark pipeline.
_pipeline = importlib.import_module("process_extended")
COVERAGE_MARGIN: float = _pipeline.COVERAGE_MARGIN
EQUIVALENCE_ALPHA: float = _pipeline.EQUIVALENCE_ALPHA
MAE_MARGIN: float = _pipeline.MAE_MARGIN
TVD_MARGIN: float = _pipeline.TVD_MARGIN

Z95 = float(norm.ppf(0.975))

# Labels and display order -------------------------------------------------

SDK_NAMES = {
    "reference": "Reference simulator",
    "qiskit": "Qiskit",
    "cirq": "Cirq",
    "pennylane": "PennyLane",
    "qsharp": "Microsoft Q#",
}
SDK_PACKAGES = {
    "qiskit": "qiskit",
    "cirq": "cirq-core",
    "pennylane": "pennylane",
    "qsharp": "qdk",
}
SCALING_BACKENDS = (
    ("reference", "Reference simulator"),
    ("qiskit", "Qiskit"),
    ("cirq", "Cirq"),
    ("pennylane", "PennyLane"),
    ("qsharp", "Microsoft Q# (QDK)"),
)
PATH_NAMES = {
    "qiskit": "Qiskit",
    "cirq": "Cirq",
    "pennylane": "PennyLane",
    "qsharp": "QDK",
    "qasm2": "QASM 2",
    "qasm3": "QASM 3",
    "qasm3-noinclude": "QASM 3 (include line removed)",
    "pytket": "pytket",
}
FAILURE_STAGES = {
    "export_failed": "export",
    "import_failed": "import",
    "unitary_failed": "unitary evaluation",
}
METHODS = (
    ("iqae", "Iterative"),
    ("mlqae", "Maximum likelihood"),
    ("faster_ae", "Faster"),
    ("canonical_qae", "Canonical"),
)
LIBRARIES = (
    ("aeonq", "aeonq"),
    ("qiskit_algorithms", "qiskit-algorithms"),
    ("qiskit_algorithms_lr", "Likelihood-ratio interval, same trials"),
)
MUTANTS = (
    ("drop_global_phase", "Drop global phase"),
    ("global_phase_sign", "Invert global phase sign"),
    ("drop_phase_under_control", "Omit phase under control"),
    ("rotation_sign", "Negate rotation parameters"),
    ("rotation_half_angle", "Halve rotation angles"),
    ("rz_as_phase_gate", "Lower R_Z as phase gate"),
    ("swap_cx_direction", "Swap CNOT direction"),
    ("invert_first_control", "Invert control polarity"),
    ("adjoint_ignored", "Omit adjoint"),
    ("grover_power_off_by_one", "Shift power index (Q^(k+1) for Q^k)"),
    ("zero_reflection_about_ones", "Reflect about |1...1>"),
    ("objective_reflection_negated", "Negate objective reflection S_chi"),
    ("reverse_qubit_order", "Reverse qubit order"),
    ("reverse_result_bits", "Reverse readout bit order"),
    ("drop_one_shot", "Omit one shot record"),
)
PHASE_FAULTS = (
    "drop_global_phase",
    "drop_phase_under_control",
    "objective_reflection_negated",
)
CHECK_LEVELS = ("operator", "measurement", "sampling")
SCALING_COLUMNS = (4, 8, 12)  # register widths shown in the scaling table
CUBE_WIDTHS = range(1, 9)  # register widths of the exhaustive cube-count check

# Data model ---------------------------------------------------------------


@dataclass
class Table:
    title: str
    header: tuple[str, ...]
    rows: list[tuple[str, list[str]]]
    footer: str = ""


@dataclass
class Section:
    title: str
    tables: list[Table] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    quantities: list[tuple[str, str]] = field(default_factory=list)

    def add(self, label: str, value: Any, spec: str) -> None:
        self.quantities.append((label, show(value, spec)))


def show(value: Any, spec: str) -> str:
    """Format a value; spec is int, str, bool, f<n>, g<n> or e<n>."""
    if value is None:
        return "--"
    if spec == "bool":
        return "yes" if value else "no"
    if spec == "str":
        return str(value)
    if spec == "int":
        return f"{value:,.0f}"
    digits = int(spec[1:])
    if spec[0] == "f":
        return f"{value:.{digits}f}"
    if spec[0] == "e":
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


# Statistics ---------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(k1: int, n1: int, k2: int, n2: int, level: float) -> tuple[float, float]:
    """Newcombe hybrid score interval (method 10) for p1 - p2."""
    z = float(norm.ppf(1 - (1 - level) / 2))
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return (
        d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def welch(a: np.ndarray, b: np.ndarray, level: float) -> tuple[float, float]:
    """Welch interval for mean(a) - mean(b)."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = sqrt(va + vb)
    d = float(a.mean() - b.mean())
    if se == 0:
        return d, d
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = float(t.ppf(1 - (1 - level) / 2, df)) * se
    return d - half, d + half


# Data loaders, also used by reproduce_figures.py --------------------------


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def recorded_versions(path: Path) -> dict[str, str]:
    """Package versions recorded with a study's results."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("environment", {}).get("dependencies", {})


def sdk_label(sdk: str, versions: dict[str, str]) -> str:
    if sdk == "reference":
        return SDK_NAMES[sdk]
    release = versions.get(SDK_PACKAGES[sdk], "?")
    if sdk == "qsharp":
        return f"{SDK_NAMES[sdk]} (QDK {release})"
    return f"{SDK_NAMES[sdk]} {release}"


def path_label(path: str) -> str:
    return " -> ".join(PATH_NAMES.get(part, part) for part in path.split(">"))


@lru_cache(maxsize=None)
def check_totals() -> dict[str, int]:
    """Number of checks per level in the conformance battery."""
    from aeonq.adapters.reference import ReferenceAdapter
    from aeonq.conformance.battery import run_battery

    totals: dict[str, int] = defaultdict(int)
    for result in run_battery(ReferenceAdapter()):
        totals[result.level] += 1
    return dict(totals)


@lru_cache(maxsize=None)
def baseline_cells() -> dict[tuple[str, str, str], dict]:
    """Summarize the baseline trials for each (method, library, case).

    Returns the number of trials, coverage with its Wilson interval, mean
    absolute error, mean interval width, mean query count, whether every trial
    stopped after one round, the fraction of converged trials and the
    estimator configuration.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(RAW / "native_baselines.jsonl"):
        if row["error"] is None:
            groups[(row["method"], row["library"], row["case_id"])].append(row)
    cells = {}
    for key, rows in groups.items():
        covered = sum(bool(r["covered"]) for r in rows)
        cells[key] = {
            "a_true": rows[0]["a_true"],
            "configuration": rows[0]["configuration"],
            "n": len(rows),
            "coverage": covered / len(rows),
            "wilson": wilson(covered, len(rows)),
            "mae": mean(abs(r["estimate"] - r["a_true"]) for r in rows),
            "width": mean(r["interval"][1] - r["interval"][0] for r in rows),
            "queries": mean(r["queries"] for r in rows),
            "single_round": all(r["queries"] == r["shots"] for r in rows),
            "converged": mean(r["status"] == "CONVERGED" for r in rows),
        }
    return cells


def baseline_case_order(cells: dict) -> list[str]:
    cases = {case for _, _, case in cells}

    def key(case: str) -> tuple:
        if case.startswith("bernoulli_"):
            return (0, float(case.split("_", 1)[1]), case)
        return (1, 0.0, case)

    return sorted(cases, key=key)


@lru_cache(maxsize=None)
def noise_cells() -> dict[tuple[float, str, str], dict]:
    """Summarize the noise trials for each (one-qubit rate, case, method)."""
    groups: dict[tuple[float, str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(RAW / "noise_study.jsonl"):
        if row["error"] is None:
            groups[(row["one_qubit_error"], row["case_id"], row["method"])].append(row)
    cells = {}
    for key, rows in groups.items():
        covered = sum(bool(r["covered"]) for r in rows)
        cells[key] = {
            "n": len(rows),
            "two_qubit_error": rows[0]["two_qubit_error"],
            "coverage": covered / len(rows),
            "wilson": wilson(covered, len(rows)),
            "statuses": {r["guarantee_status"] for r in rows},
            "warned": mean(bool(r["warnings"]) for r in rows),
        }
    return cells


@lru_cache(maxsize=None)
def mutation_profiles() -> dict[str, dict]:
    """Summarize the mutation battery for each injected fault.

    Returns the failing (operator, measurement, sampling) check counts, whether
    they agree across SDKs, and the number of SDKs on which the fault was
    detected.
    """
    by_mutant: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(RAW / "adapter_mutation.csv"):
        by_mutant[row["mutant"]].append(row)
    profiles = {}
    for mutant, rows in by_mutant.items():
        shapes = {
            tuple(int(r[f"failed_{level}"]) for level in CHECK_LEVELS) for r in rows
        }
        profiles[mutant] = {
            "failed": next(iter(shapes)),
            "identical": len(shapes) == 1,
            "killed": sum(r["killed"] == "True" for r in rows),
            "sdks": len(rows),
        }
    return profiles


def scaling_rows() -> list[dict]:
    return json.loads((RAW / "scaling_study.json").read_text(encoding="utf-8"))["rows"]


def scaling_seconds(row: dict) -> float | None:
    """Seconds to lower, compile and sample; recorded compile time includes lowering."""
    if row.get("status") != "ok":
        return None
    return row["compile_seconds"] + row["execute_seconds"]


def compiler_rows() -> list[dict]:
    return read_csv(RAW / "phase10_compiler.csv")


# Sections -----------------------------------------------------------------


def conformance_section() -> Section:
    from aeonq.adapters.cirq import CirqAdapter
    from aeonq.adapters.pennylane import PennyLaneAdapter
    from aeonq.adapters.qiskit import QiskitAdapter
    from aeonq.adapters.qsharp import QSharpAdapter
    from aeonq.adapters.reference import ReferenceAdapter
    from aeonq.conformance.battery import run_battery
    from aeonq.conformance.operator_equivalence import measure_operator_conformance

    section = Section("Conformance")
    adapters = {
        "reference": ReferenceAdapter,
        "qiskit": QiskitAdapter,
        "cirq": CirqAdapter,
        "pennylane": PennyLaneAdapter,
        "qsharp": QSharpAdapter,
    }
    installed = {package: version(package) for package in SDK_PACKAGES.values()}
    equivalence: dict[str, float] = defaultdict(float)
    for row in measure_operator_conformance():
        equivalence[row["backend"]] = max(
            equivalence[row["backend"]], float(row["exact_distance"])
        )
    rows, battery_worst = [], {}
    for sdk, factory in adapters.items():
        adapter = factory()
        passed: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        results = run_battery(adapter)
        for result in results:
            total[result.level] += 1
            passed[result.level] += bool(result.passed)
        battery_worst[sdk] = max(
            (r.measure for r in results if r.level == "operator" and r.measure),
            default=0.0,
        )
        rows.append(
            (
                sdk_label(sdk, installed),
                [
                    show(adapter.unitary_precision, "e0"),
                    *(f"{passed[level]}/{total[level]}" for level in CHECK_LEVELS),
                    show(battery_worst[sdk], "e1"),
                    show(equivalence.get(sdk), "e1"),
                ],
            )
        )
    section.tables.append(
        Table(
            "Adapter targets, declared entry-wise unitary tolerance and conformance",
            (
                "Adapter target",
                "Tolerance",
                "Operator",
                "Measurement",
                "Sampling",
                "Max deviation (battery)",
                "Max deviation (equivalence check)",
            ),
            rows,
        )
    )
    others = ("qiskit", "cirq", "pennylane")
    section.add(
        "Largest entry-wise deviation, Qiskit/Cirq/PennyLane, equivalence check",
        max(equivalence[k] for k in others),
        "e1",
    )
    section.add(
        "Largest entry-wise deviation, Q#, equivalence check",
        equivalence["qsharp"],
        "e1",
    )
    section.add(
        "Largest entry-wise deviation, Qiskit/Cirq/PennyLane, battery",
        max(battery_worst[k] for k in others),
        "e1",
    )
    section.add(
        "Largest entry-wise deviation, Q#, battery", battery_worst["qsharp"], "e1"
    )
    return section


def mutation_section() -> Section:
    section = Section("Mutation analysis")
    profiles = mutation_profiles()
    totals = check_totals()
    rows = [
        (
            label,
            [
                *(show(value, "int") for value in profiles[mutant]["failed"]),
                f"{profiles[mutant]['killed']}/{profiles[mutant]['sdks']}",
            ],
        )
        for mutant, label in MUTANTS
    ]
    section.tables.append(
        Table(
            "Failing conformance checks per injected fault ("
            + ", ".join(f"{totals[level]} {level}" for level in CHECK_LEVELS)
            + ")",
            ("Lowering fault", "Operator", "Measurement", "Sampling", "SDKs detecting"),
            rows,
        )
    )
    section.add("Checks in the battery", sum(totals.values()), "int")
    for level in CHECK_LEVELS:
        section.add(f"  {level} checks", totals[level], "int")
    section.add("Injected faults", len(MUTANTS), "int")
    section.add(
        "Faults detected on every SDK",
        sum(p["killed"] == p["sdks"] for m, p in profiles.items() if m != "none"),
        "int",
    )
    section.add(
        "Unmutated adapters detected as faulty", profiles["none"]["killed"], "int"
    )
    section.add(
        "Failure profiles identical across SDKs",
        all(p["identical"] for p in profiles.values()),
        "bool",
    )
    section.add(
        "Faults with no measurement or sampling failure",
        sum(profiles[m]["failed"][1:] == (0, 0) for m, _ in MUTANTS),
        "int",
    )

    impact_rows = read_csv(RAW / "adapter_mutation_impact.csv")
    impact = {
        (r["mutant"], r["case_id"], r["method"]): float(r["mean_abs_error"])
        for r in impact_rows
        if r["mean_abs_error"] not in ("", "None")
    }
    sdk = SDK_NAMES.get(impact_rows[0]["sdk"], impact_rows[0]["sdk"])
    trials = impact_rows[0]["repetitions"]
    cases = sorted({r["case_id"] for r in impact_rows})
    for case in cases:
        section.add(
            f"{sdk} canonical MAE, {case}, unmutated ({trials} trials)",
            impact[("none", case, "canonical_qae")],
            "f3",
        )
        for fault in PHASE_FAULTS:
            section.add(f"  with {fault}", impact[(fault, case, "canonical_qae")], "f2")
    section.add(
        "Iterative MAE unchanged by the three phase faults",
        all(
            impact[(fault, case, "iqae")] == impact[("none", case, "iqae")]
            for fault in PHASE_FAULTS
            for case in cases
        ),
        "bool",
    )
    section.add(
        "Estimation error unchanged by phase-sign inversion and R_Z as phase gate",
        all(
            impact[(fault, case, method)] == impact[("none", case, method)]
            for fault in ("global_phase_sign", "rz_as_phase_gate")
            for case in cases
            for method in ("iqae", "canonical_qae")
        ),
        "bool",
    )
    return section


def translator_section() -> Section:
    section = Section("Translator fidelity and independent oracles")
    raw = read_csv(RAW / "translator_fidelity.csv")
    by_path: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        by_path[row["path"]].append(row)
    available = [
        path
        for path, rows in by_path.items()
        if any(r["status"] != "unavailable" for r in rows)
    ]
    rows, totals = [], [0] * 6
    for path in available:
        evaluated = [r for r in by_path[path] if r["status"] == "evaluated"]
        controllable = [r for r in evaluated if r["equal_when_controlled"] != ""]
        values: list[int | None] = [len(evaluated)]
        if evaluated:
            values += [
                sum(r["equal_exact"] == "True" for r in evaluated),
                sum(
                    r["equal_up_to_global_phase"] == "True"
                    and r["equal_exact"] != "True"
                    for r in evaluated
                ),
                sum(r["equal_up_to_global_phase"] != "True" for r in evaluated),
                sum(r["equal_when_controlled"] == "False" for r in controllable)
                if controllable
                else None,
                sum(r["count_invisible_failure"] == "True" for r in evaluated),
            ]
        else:
            values += [None] * 5
        for i, value in enumerate(values):
            totals[i] += value or 0
        rows.append((path_label(path), [show(v, "int") for v in values]))
    rows.append(("Total", [show(v, "int") for v in totals]))
    sizes = {len(by_path[p]) for p in available}
    section.tables.append(
        Table(
            f"Translator fidelity ({', '.join(map(str, sorted(sizes)))} programs per path)",
            (
                "Path",
                "Evaluated",
                "Exact",
                "Phase only",
                "Not equal",
                "Controlled",
                "Count-invisible",
            ),
            rows,
        )
    )

    attempted = [r for p in available for r in by_path[p]]
    evaluated = [r for r in attempted if r["status"] == "evaluated"]
    section.add("Conversion paths evaluated", len(available), "int")
    section.add("Programs on available paths", len(attempted), "int")
    section.add("Programs translated", len(evaluated), "int")
    section.add("Programs not translated", len(attempted) - len(evaluated), "int")
    stages: dict[str, int] = defaultdict(int)
    for r in attempted:
        if r["status"] != "evaluated":
            stages[r["status"]] += 1
    for stage, count in sorted(stages.items()):
        section.add(f"  failing at {FAILURE_STAGES.get(stage, stage)}", count, "int")
    section.add(
        "Translated programs with permuted qubits",
        sum(r["qubit_order"] not in ("identity", "none", "") for r in evaluated),
        "int",
    )
    section.add(
        "Exactly equal programs failing once controlled",
        sum(
            r["equal_exact"] == "True" and r["equal_when_controlled"] == "False"
            for r in evaluated
        ),
        "int",
    )
    grover = [r for r in evaluated if r["node"] == "Q"]
    changed = [
        r
        for r in grover
        if abs(float(r["canonical_qae_source"]) - float(r["canonical_qae_translated"]))
        > 1e-9
    ]
    section.add("Adapter-lowered Grover operators translated", len(grover), "int")
    section.add("  whose canonical estimate changes", len(changed), "int")
    shifts: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in changed:
        before = show(float(r["canonical_qae_source"]), "f2")
        after = show(float(r["canonical_qae_translated"]), "f2")
        shifts[(r["case"], before, after)] += 1
    for (case, before, after), count in sorted(shifts.items()):
        section.lines.append(
            f"  Canonical estimate {before} -> {after} on {case} ({count} path{'' if count == 1 else 's'})"
        )

    native = [
        r
        for r in read_csv(RAW / "translator_native_grover.csv")
        if r["status"] == "evaluated" and r["node"].startswith("Q (")
    ]
    kept = [r for r in native if r["equal_exact"] == "True"]
    section.add("Native Grover operator translations", len(native), "int")
    section.add("  that lose the -1", len(native) - len(kept), "int")
    section.add(
        "  targets of those that keep it",
        ", ".join(sorted({r["target"] for r in kept})),
        "str",
    )

    oracle = read_csv(RAW / "independent_oracles.csv")
    positive = [r for r in oracle if r["expected_to_pass"] == "True"]
    section.add("Independent oracle checks", len(positive), "int")
    section.add("  passed", sum(r["passed"] == "True" for r in positive), "int")
    worst: dict[str, float] = defaultdict(float)
    for r in positive:
        if r["kind"] != "closed_form":
            sdk = "qsharp" if r["subject"].startswith("qsharp") else "other"
            worst[sdk] = max(worst[sdk], float(r["error"]))
    section.add(
        "  largest entry-wise difference from the NumPy oracle, non-Q#",
        worst["other"],
        "e1",
    )
    section.add(
        "  largest entry-wise difference from the NumPy oracle, Q#",
        worst["qsharp"],
        "e1",
    )
    variants: dict[tuple, list[dict]] = defaultdict(list)
    for r in oracle:
        if r["expected_to_pass"] != "True":
            variants[(r["case"], r["subject"])].append(r)
    section.add("Faulty oracle variants", len(variants), "int")
    section.add(
        "  rejected",
        sum(any(r["passed"] != "True" for r in v) for v in variants.values()),
        "int",
    )
    return section


def equivalence_footer(section: Section) -> str:
    """Cross-SDK equivalence: sampling (Weissman L1 bound) and estimators (TOST)."""
    sampling = json.loads(
        (RAW / "equivalence_sampling.json").read_text(encoding="utf-8")
    )["rows"]
    bounds: dict[str, list[float]] = {"reference": [], "sdk": []}
    for r in sampling:
        counts = np.asarray(r["counts"], dtype=float)
        exact = np.asarray(r["exact_probabilities"], dtype=float)
        shots, m = counts.sum(), len(counts)
        tvd = 0.5 * float(np.abs(counts / shots - exact).sum())
        radius = sqrt(2 / shots * log((2**m - 2) / 0.05)) if m > 1 else 0.0
        bounds["reference" if r["backend"] == "reference" else "sdk"].append(
            tvd + radius / 2
        )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in read_jsonl(RAW / "equivalence_estimators.jsonl"):
        if r["error"] is None:
            groups[(r["case_id"], r["method"], r["backend"])].append(r)
    level = 1 - 2 * EQUIVALENCE_ALPHA
    coverage_ok = error_ok = comparisons = 0
    for (case, method, backend), trials in groups.items():
        if backend == "reference":
            continue
        reference = groups[(case, method, "reference")]
        comparisons += 1
        low, high = newcombe(
            sum(bool(r["covered"]) for r in trials),
            len(trials),
            sum(bool(r["covered"]) for r in reference),
            len(reference),
            level,
        )
        coverage_ok += -COVERAGE_MARGIN < low and high < COVERAGE_MARGIN
        low, high = welch(
            np.array([abs(r["estimate"] - r["a_true"]) for r in trials]),
            np.array([abs(r["estimate"] - r["a_true"]) for r in reference]),
            level,
        )
        error_ok += -MAE_MARGIN < low and high < MAE_MARGIN
    every = bounds["reference"] + bounds["sdk"]
    section.add(
        "Sampling comparisons within the TVD margin, SDK adapters",
        f"{sum(b < TVD_MARGIN for b in bounds['sdk'])}/{len(bounds['sdk'])}",
        "str",
    )
    section.add(
        "  reference simulator",
        f"{sum(b < TVD_MARGIN for b in bounds['reference'])}/{len(bounds['reference'])}",
        "str",
    )
    section.add("Largest SDK 95% upper bound on TVD", max(bounds["sdk"]), "f4")
    section.add(
        "Equivalence margins: TVD, coverage, MAE",
        f"{TVD_MARGIN}, {COVERAGE_MARGIN}, {MAE_MARGIN}",
        "str",
    )
    return (
        f"Equivalence (margins met): sampling {sum(b < TVD_MARGIN for b in every)}"
        f"/{len(every)}, coverage {coverage_ok}/{comparisons}, "
        f"error {error_ok}/{comparisons}"
    )


def baseline_section() -> Section:
    section = Section("Baseline comparison and equivalence")
    cells = baseline_cells()
    cases = baseline_case_order(cells)
    rows = []
    for method, method_name in METHODS:
        for library, library_name in LIBRARIES:
            if (method, library, cases[0]) not in cells:
                continue
            selected = [cells[(method, library, c)] for c in cases]
            label = (
                f"  {library_name}"
                if library == "qiskit_algorithms_lr"
                else f"{method_name}, {library_name}"
            )
            rows.append(
                (
                    label,
                    [
                        show(mean(c["queries"] for c in selected), "int"),
                        show(mean(c["mae"] for c in selected), "g2"),
                        show(mean(c["width"] for c in selected), "g2"),
                        show(min(c["coverage"] for c in selected), "f3"),
                        show(sum(c["wilson"][1] < 0.95 for c in selected), "int"),
                    ],
                )
            )
    trials = cells[(METHODS[0][0], "aeonq", cases[0])]["n"]
    section.tables.append(
        Table(
            f"Comparison with qiskit-algorithms ({len(cases)} problems, "
            f"{trials} trials each) and cross-SDK equivalence",
            (
                "Estimator",
                "Mean queries",
                "Mean abs. error",
                "Mean int. width",
                "Lowest coverage",
                "Cases sig. below 0.95",
            ),
            rows,
            equivalence_footer(section),
        )
    )

    first_round = [c for c in cases if cells[("iqae", "aeonq", c)]["single_round"]]
    section.lines.append(
        "  Problems on which iterative aeonq stops after one round: "
        + ", ".join(first_round)
    )
    section.add(
        "Iterative aeonq queries on those problems",
        mean(cells[("iqae", "aeonq", c)]["queries"] for c in first_round),
        "int",
    )
    section.add(
        "  mean absolute error",
        mean(cells[("iqae", "aeonq", c)]["mae"] for c in first_round),
        "f4",
    )
    section.add(
        "  qiskit-algorithms queries",
        mean(cells[("iqae", "qiskit_algorithms", c)]["queries"] for c in first_round),
        "int",
    )
    section.add(
        "  qiskit-algorithms mean absolute error",
        mean(cells[("iqae", "qiskit_algorithms", c)]["mae"] for c in first_round),
        "e0",
    )
    for case in cases:
        if case not in first_round:
            ratio = (
                cells[("iqae", "aeonq", case)]["queries"]
                / cells[("iqae", "qiskit_algorithms", case)]["queries"]
            )
            section.add(
                f"Iterative query ratio aeonq/qiskit-algorithms, {case}", ratio, "f1"
            )

    for case in cases:
        if cells[("mlqae", "qiskit_algorithms", case)]["wilson"][1] < 0.95:
            section.add(
                f"Fisher-interval coverage significantly below nominal, {case}",
                cells[("mlqae", "qiskit_algorithms", case)]["coverage"],
                "f3",
            )
            section.add(
                "  likelihood-ratio coverage, same trials",
                cells[("mlqae", "qiskit_algorithms_lr", case)]["coverage"],
                "f3",
            )
    ranked = sorted(
        cases, key=lambda c: cells[("mlqae", "qiskit_algorithms_lr", c)]["coverage"]
    )
    for case in ranked[:2]:
        section.add(
            f"Likelihood-ratio coverage, {case}",
            cells[("mlqae", "qiskit_algorithms_lr", case)]["coverage"],
            "f3",
        )
    section.add(
        "Clopper-Pearson / Fisher mean width ratio",
        mean(cells[("mlqae", "aeonq", c)]["width"] for c in cases)
        / mean(cells[("mlqae", "qiskit_algorithms", c)]["width"] for c in cases),
        "f1",
    )
    section.add(
        "Maximum-likelihood problems reaching the requested width in every trial",
        sum(cells[("mlqae", "aeonq", c)]["converged"] == 1.0 for c in cases),
        "int",
    )
    section.add(
        "  in at least one trial",
        sum(cells[("mlqae", "aeonq", c)]["converged"] > 0 for c in cases),
        "int",
    )
    faster = mean(cells[("faster_ae", "aeonq", c)]["queries"] for c in cases) / mean(
        cells[("faster_ae", "qiskit_algorithms", c)]["queries"] for c in cases
    )
    section.add(
        "Faster AE additional queries, aeonq vs qiskit-algorithms (%)",
        100 * (faster - 1),
        "f0",
    )

    from aeonq.algorithms.faster_ae import FasterAE

    configuration = cells[("faster_ae", "aeonq", cases[0])]["configuration"]
    alpha, iterations = configuration["alpha"], configuration["max_iterations"]
    first, second = FasterAE(
        epsilon=configuration["epsilon"], alpha=alpha, max_iterations=iterations
    ).required_shots
    section.add(
        "Faster AE per-cosine failure budget alpha/(2J)", alpha / (2 * iterations), "f5"
    )
    section.add("Faster AE first-stage shots, aeonq", first, "int")
    section.add("Faster AE second-stage shots, aeonq", second, "int")
    section.add(
        "Faster AE first-stage shots, qiskit-algorithms int(1944 ln(2/alpha))",
        int(1944 * log(2 / alpha)),
        "int",
    )
    section.add(
        "Faster AE second-stage shots, qiskit-algorithms int(972 ln(2/alpha))",
        int(972 * log(2 / alpha)),
        "int",
    )
    bounds = [1 - (2 * iterations - j0) * alpha for j0 in range(1, iterations + 1)]
    section.add(
        "qiskit-algorithms success bound 1-(2J-j0)alpha, smallest", min(bounds), "f2"
    )
    section.add(
        "qiskit-algorithms success bound 1-(2J-j0)alpha, largest", max(bounds), "f2"
    )
    return section


def noise_section() -> Section:
    section = Section("Noise study")
    cells = noise_cells()
    rates = sorted({rate for rate, _, _ in cells})
    cases = sorted({case for _, case, _ in cells})
    methods = sorted({method for _, _, method in cells})
    rows = [
        (
            f"{method}, {case}",
            [show(cells[(rate, case, method)]["coverage"], "f3") for rate in rates],
        )
        for method in methods
        for case in cases
    ]
    ratio = cells[(rates[-1], cases[0], methods[0])]["two_qubit_error"] / rates[-1]
    section.tables.append(
        Table(
            "Interval coverage by single-qubit depolarizing rate "
            f"(two-qubit rate {ratio:g} times higher; rate 0 is ideal simulation)",
            ("Estimator, case", *(f"{rate:g}" for rate in rates)),
            rows,
        )
    )
    noisy = [c for (rate, _, _), c in cells.items() if rate > 0]
    section.add(
        "Every noisy result carries the non-ideal-sampling warning",
        all(c["warned"] == 1 for c in noisy),
        "bool",
    )
    section.add(
        "No noisy result reports a satisfied guarantee",
        all("THEORETICAL_GUARANTEE_SATISFIED" not in c["statuses"] for c in noisy),
        "bool",
    )
    return section


def overhead_section() -> Section:
    section = Section("Framework overhead")
    breakdown = json.loads((RAW / "overhead_study.json").read_text(encoding="utf-8"))[
        "estimator_breakdown"
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in breakdown:
        groups[(r["sdk"], r["case_id"], r["method"])].append(r)
    rows = [
        (
            ", ".join(key),
            [
                show(len(trials), "int"),
                show(1000 * median(r["wall_seconds"] for r in trials), "f1"),
                show(100 * median(r["aeonq_other_fraction"] for r in trials), "g2"),
            ],
        )
        for key, trials in sorted(groups.items())
    ]
    section.tables.append(
        Table(
            "Median wall-clock time and share spent in aeonq outside SDK "
            "compilation and execution",
            (
                "SDK, case, estimator",
                "Evaluations",
                "Median wall (ms)",
                "Median share (%)",
            ),
            rows,
        )
    )
    return section


def compiler_section() -> Section:
    section = Section("Optimization pipeline")
    compiler = compiler_rows()
    configs = sorted({(r["case_id"], int(r["power"])) for r in compiler})
    pick = {(r["case_id"], int(r["power"]), r["pipeline"]): r for r in compiler}

    def resources(case: str, power: int, pipeline: str) -> tuple[int, int]:
        row = pick[(case, power, pipeline)]
        return int(row["depth"]), int(row["two_qubit_gates"])

    optimized = [r for r in compiler if r["pipeline"] != "naive_materialization"]
    section.add("Configurations compared", len(configs), "int")
    section.add(
        "  where the pipeline matches level 3 in depth and two-qubit gates",
        sum(
            resources(c, k, "ae_aware") == resources(c, k, "generic_sdk")
            for c, k in configs
        ),
        "int",
    )
    section.add(
        "  where the pipeline improves on level 3",
        sum(
            resources(c, k, "ae_aware") < resources(c, k, "generic_sdk")
            for c, k in configs
        ),
        "int",
    )
    section.add(
        "Largest optimized depth", max(int(r["depth"]) for r in optimized), "int"
    )
    section.add(
        "Largest optimized two-qubit count",
        max(int(r["two_qubit_gates"]) for r in optimized),
        "int",
    )
    section.add(
        "Largest naive depth",
        max(
            int(r["depth"])
            for r in compiler
            if r["pipeline"] == "naive_materialization"
        ),
        "int",
    )
    return section


def scaling_section() -> Section:
    section = Section("Scaling")
    rows_raw = scaling_rows()
    versions = recorded_versions(RAW / "scaling_study.json")
    power = max(r["power"] for r in rows_raw)
    shots = sorted({r["shots_returned"] for r in rows_raw if r.get("status") == "ok"})
    rows = []
    for backend, _ in SCALING_BACKENDS:
        cells = []
        for qubits in SCALING_COLUMNS:
            match = [
                r
                for r in rows_raw
                if r["backend"] == backend
                and r["qubits"] == qubits
                and r["power"] == power
            ]
            cells.append(show(scaling_seconds(match[0]) if match else None, "f3"))
        rows.append((sdk_label(backend, versions), cells))
    section.tables.append(
        Table(
            f"Seconds to lower, compile and execute {shots[0]} shots of Q^{power} A "
            "on a uniform threshold problem (one run per cell)",
            ("Backend", *(f"{q} qubits" for q in SCALING_COLUMNS)),
            rows,
        )
    )
    return section


def synthesis_section() -> Section:
    from aeonq.synthesis.reflection import threshold_cubes

    section = Section("Phase-oracle synthesis")
    excess = max(
        len(threshold_cubes(tuple(range(m)), k)) - m
        for m in CUBE_WIDTHS
        for k in range(2**m)
    )
    section.add(
        f"Largest cube count minus register width m, all thresholds on "
        f"{CUBE_WIDTHS[0]} to {CUBE_WIDTHS[-1]} qubits",
        excess,
        "int",
    )
    return section


# Output -------------------------------------------------------------------


def print_table(table: Table) -> None:
    rows = [(label, *cells) for label, cells in table.rows]
    widths = [
        max(len(str(row[i])) for row in [table.header, *rows])
        for i in range(len(table.header))
    ]
    rule = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(f"\n{table.title}\n{rule}")
    for i, row in enumerate([table.header, *rows]):
        print(
            "  ".join(
                str(cell).ljust(widths[j]) if j == 0 else str(cell).rjust(widths[j])
                for j, cell in enumerate(row)
            )
        )
        if i == 0:
            print(rule)
    print(rule)
    if table.footer:
        print(table.footer)


def print_section(section: Section) -> None:
    print(f"\n{section.title}\n" + "=" * len(section.title))
    for table in section.tables:
        print_table(table)
    if section.lines or section.quantities:
        print()
    for line in section.lines:
        print(line)
    for label, value in section.quantities:
        print(f"  {label:<78s} {value:>14s}")


def regenerate() -> None:
    python = sys.executable
    for command in (
        ["benchmarks/pipeline/regenerate_everything.py", "--fresh"],
        ["benchmarks/translator_fidelity.py"],
        ["benchmarks/independent_oracles.py"],
    ):
        print("running", " ".join(command), flush=True)
        subprocess.run([python, *command], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="omit the SDK conformance battery and operator-equivalence check",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rerun every study before the analysis (about 90 minutes)",
    )
    arguments = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if arguments.regenerate:
        regenerate()
    builders = [
        mutation_section,
        translator_section,
        baseline_section,
        noise_section,
        overhead_section,
        compiler_section,
        scaling_section,
        synthesis_section,
    ]
    if not arguments.no_live:
        builders.insert(0, conformance_section)
    for build in builders:
        print_section(build())


if __name__ == "__main__":
    main()
