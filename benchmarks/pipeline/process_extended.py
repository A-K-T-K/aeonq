"""Tables for the extended evaluation, derived only from raw result files.

Covers the native baselines, the case study, overhead, scaling, adapter
mutation analysis, noise, cross-SDK equivalence and extensibility. Every
statistic is computed here from files written by the benchmark scripts.
"""

import csv
import json
from math import log, sqrt
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.stats import t

from process_results import read_jsonl, wilson_interval

COVERAGE_MARGIN = 0.05
MAE_MARGIN = 0.002
TVD_MARGIN = 0.01
EQUIVALENCE_ALPHA = 0.05  # TOST: 90% intervals inside the margins


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _coverage(rows: list[dict]) -> dict:
    with_interval = [r for r in rows if r.get("covered") is not None]
    covered = sum(bool(r["covered"]) for r in with_interval)
    low, high = wilson_interval(covered, len(with_interval))
    return {
        "coverage_n": len(with_interval),
        "coverage": covered / len(with_interval) if with_interval else None,
        "coverage_wilson_lower": low if with_interval else None,
        "coverage_wilson_upper": high if with_interval else None,
    }


def _errors(rows: list[dict]) -> dict:
    errors = np.array([r["estimate"] - r["a_true"] for r in rows], dtype=float)
    return {
        "bias": float(errors.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def native_point_estimate_rows(raw: list[dict]) -> list[dict]:
    """Include Qiskit's canonical MLE without inventing additional trials.

    Its interval, samples, seed, queries and timing are identical to the native
    grid row; only the reported point estimate changes to the saved MLE.
    """
    derived = [
        {**row, "library": "qiskit_algorithms_mle", "estimate": row["mle"]}
        for row in raw
        if row["method"] == "canonical_qae"
        and row["library"] == "qiskit_algorithms"
        and row["error"] is None
    ]
    return [*raw, *derived]


def native_baseline_rows(raw: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in native_point_estimate_rows(raw):
        groups.setdefault((row["case_id"], row["method"], row["library"]), []).append(
            row
        )
    output = []
    for (case, method, library), rows in sorted(groups.items()):
        ok = [r for r in rows if r["error"] is None]
        entry = {
            "case_id": case,
            "a_true": rows[0]["a_true"],
            "method": method,
            "library": library,
            "trials": len(rows),
            "failures": len(rows) - len(ok),
        }
        if ok:
            entry.update(_errors(ok))
            entry.update(_coverage(ok))
            entry.update(
                interval_kind=ok[0]["interval_kind"],
                mean_queries=mean(r["queries"] for r in ok),
                median_seconds=median(r["seconds"] for r in ok),
                converged_fraction=(
                    mean(r["status"] == "CONVERGED" for r in ok)
                    if library == "aeonq"
                    else None
                ),
            )
        output.append(entry)
    return output


def native_summary_rows(detail: list[dict]) -> list[dict]:
    """Per method and library: worst case, and cases significantly under nominal."""
    output = []
    for method in sorted({r["method"] for r in detail}):
        # The likelihood-ratio MLAE series is a sensitivity check on one method.
        for library in (
            "aeonq",
            "qiskit_algorithms",
            "qiskit_algorithms_lr",
            "qiskit_algorithms_mle",
        ):
            if not any(
                r["method"] == method and r["library"] == library for r in detail
            ):
                continue
            rows = [
                r
                for r in detail
                if r["method"] == method
                and r["library"] == library
                and r.get("coverage") is not None
            ]
            all_rows = [
                r for r in detail if r["method"] == method and r["library"] == library
            ]
            entry = {
                "method": method,
                "library": library,
                "cases": len(all_rows),
                "failed_trials": sum(r["failures"] for r in all_rows),
                "mean_queries": mean(
                    r["mean_queries"] for r in all_rows if "mean_queries" in r
                ),
                "median_seconds": median(
                    r["median_seconds"] for r in all_rows if "median_seconds" in r
                ),
                "mean_mae": mean(r["mae"] for r in all_rows if "mae" in r),
            }
            if rows:
                worst = min(rows, key=lambda r: r["coverage"])
                entry.update(
                    worst_coverage_case=worst["case_id"],
                    worst_coverage=worst["coverage"],
                    worst_coverage_wilson_upper=worst["coverage_wilson_upper"],
                    cases_significantly_below_nominal=sum(
                        r["coverage_wilson_upper"] < 0.95 for r in rows
                    ),
                )
            else:
                entry.update(worst_coverage_case="no interval reported")
            output.append(entry)
    return output


def noise_rows(raw: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in raw:
        key = (row["one_qubit_error"], row["case_id"], row["method"])
        groups.setdefault(key, []).append(row)
    output = []
    for (rate, case, method), rows in sorted(groups.items()):
        ok = [r for r in rows if r["error"] is None]
        statuses = sorted({r["guarantee_status"] for r in ok})
        output.append(
            {
                "one_qubit_error": rate,
                "two_qubit_error": rows[0]["two_qubit_error"],
                "case_id": case,
                "method": method,
                "trials": len(rows),
                "failures": len(rows) - len(ok),
                **_errors(ok),
                **_coverage(ok),
                "converged_fraction": mean(
                    r["convergence_status"] == "CONVERGED" for r in ok
                ),
                "reported_guarantee_statuses": " ".join(statuses),
                "non_ideal_warning_fraction": mean(bool(r["warnings"]) for r in ok),
            }
        )
    return output


def _newcombe(
    p1: float, n1: int, p2: float, n2: int, alpha: float
) -> tuple[float, float]:
    """Newcombe hybrid score interval for p1 - p2 (method 10)."""
    l1, u1 = wilson_interval(round(p1 * n1), n1, alpha)
    l2, u2 = wilson_interval(round(p2 * n2), n2, alpha)
    d = p1 - p2
    return (
        d - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def _welch(a: np.ndarray, b: np.ndarray, alpha: float) -> tuple[float, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = sqrt(va + vb)
    if se == 0:
        difference = float(a.mean() - b.mean())
        return difference, difference
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = t.ppf(1 - alpha, df) * se
    d = float(a.mean() - b.mean())
    return float(d - half), float(d + half)


def _verdict(low: float, high: float, margin: float) -> str:
    if -margin < low and high < margin:
        return "equivalent"
    if low > margin or high < -margin:
        return "different"
    return "inconclusive"


def equivalence_estimator_rows(raw: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in raw:
        if row["error"] is None:
            groups.setdefault(
                (row["case_id"], row["method"], row["backend"]), []
            ).append(row)
    output = []
    for (case, method, backend), rows in sorted(groups.items()):
        if backend == "reference":
            continue
        reference = groups[(case, method, "reference")]
        cov = mean(bool(r["covered"]) for r in rows)
        ref_cov = mean(bool(r["covered"]) for r in reference)
        c_low, c_high = _newcombe(
            cov, len(rows), ref_cov, len(reference), 2 * EQUIVALENCE_ALPHA
        )
        abs_err = np.array([abs(r["estimate"] - r["a_true"]) for r in rows])
        ref_err = np.array([abs(r["estimate"] - r["a_true"]) for r in reference])
        e_low, e_high = _welch(abs_err, ref_err, EQUIVALENCE_ALPHA)
        output.append(
            {
                "case_id": case,
                "method": method,
                "backend": backend,
                "trials": len(rows),
                "reference_trials": len(reference),
                "coverage": cov,
                "reference_coverage": ref_cov,
                "coverage_difference_90ci_lower": c_low,
                "coverage_difference_90ci_upper": c_high,
                "coverage_margin": COVERAGE_MARGIN,
                "coverage_verdict": _verdict(c_low, c_high, COVERAGE_MARGIN),
                "mae": float(abs_err.mean()),
                "reference_mae": float(ref_err.mean()),
                "mae_difference_90ci_lower": e_low,
                "mae_difference_90ci_upper": e_high,
                "mae_margin": MAE_MARGIN,
                "mae_verdict": _verdict(e_low, e_high, MAE_MARGIN),
            }
        )
    return output


def equivalence_sampling_rows(document: dict) -> list[dict]:
    """Distribution-free TVD upper bound (Weissman et al. 2003 L1 deviation)."""
    output = []
    for row in document["rows"]:
        counts = np.array(row["counts"], dtype=float)
        exact = np.array(row["exact_probabilities"], dtype=float)
        shots = counts.sum()
        m = len(counts)
        tvd = 0.5 * float(np.abs(counts / shots - exact).sum())
        # P(||p_hat - p||_1 >= e) <= (2^m - 2) exp(-N e^2 / 2), so with
        # probability 0.95 the true distribution is within this L1 radius.
        radius = sqrt(2 / shots * log((2**m - 2) / 0.05)) if m > 1 else 0.0
        upper = tvd + radius / 2
        output.append(
            {
                "case_id": row["case_id"],
                "power": row["power"],
                "backend": row["backend"],
                "shots": int(shots),
                "outcomes": m,
                "empirical_tvd": tvd,
                "tvd_95_upper_bound": upper,
                "tvd_margin": TVD_MARGIN,
                "verdict": "equivalent" if upper < TVD_MARGIN else "inconclusive",
            }
        )
    return output


def mutation_rows(kill: list[dict], impact: list[dict]) -> list[dict]:
    output = []
    for mutant in dict.fromkeys(r["mutant"] for r in kill):
        rows = [r for r in kill if r["mutant"] == mutant]
        effects = [r for r in impact if r["mutant"] == mutant]
        baseline = {
            (r["case_id"], r["method"]): r for r in impact if r["mutant"] == "none"
        }

        def degraded(method: str) -> str:
            changes = []
            for r in effects:
                if r["method"] != method:
                    continue
                if int(r["crashes"]):
                    changes.append("rejected")
                    continue
                base = float(baseline[(r["case_id"], method)]["mean_abs_error"])
                changes.append(f"{float(r['mean_abs_error']) - base:+.3f}")
            return " / ".join(changes)

        output.append(
            {
                "mutant": mutant,
                "description": rows[0]["description"],
                "sdks_killed": f"{sum(r['killed'] == 'True' for r in rows)}/{len(rows)}",
                "failed_operator_checks": rows[0]["failed_operator"],
                "failed_measurement_checks": rows[0]["failed_measurement"],
                "failed_sampling_checks": rows[0]["failed_sampling"],
                "detected_by_shots": int(rows[0]["failed_measurement"])
                + int(rows[0]["failed_sampling"])
                > 0,
                "iqae_mae_change": degraded("iqae"),
                "canonical_qae_mae_change": degraded("canonical_qae"),
            }
        )
    return output


def overhead_rows(document: dict) -> tuple[list[dict], list[dict]]:
    construction = [
        {
            **{k: v for k, v in row.items()},
            "aeonq_total_ms": 1e3
            * (row["aeonq_lower_seconds"] + row["aeonq_compile_seconds"]),
            "native_total_ms": 1e3
            * (row["native_build_seconds"] + row["native_compile_seconds"]),
        }
        for row in document["construction"]
    ]
    groups: dict[tuple, list[dict]] = {}
    for row in document["estimator_breakdown"]:
        groups.setdefault((row["sdk"], row["case_id"], row["method"]), []).append(row)
    breakdown = [
        {
            "sdk": sdk,
            "case_id": case,
            "method": method,
            "trials": len(rows),
            "median_wall_seconds": median(r["wall_seconds"] for r in rows),
            "median_sdk_compile_seconds": median(
                r["sdk_compile_seconds"] for r in rows
            ),
            "median_sdk_execute_seconds": median(
                r["sdk_execute_seconds"] for r in rows
            ),
            "median_aeonq_other_seconds": median(
                r["aeonq_other_seconds"] for r in rows
            ),
            "median_aeonq_other_fraction": median(
                r["aeonq_other_fraction"] for r in rows
            ),
        }
        for (sdk, case, method), rows in sorted(groups.items())
    ]
    return construction, breakdown


def write_extended(
    root: Path, tables: Path, write_csv, *, require_complete: bool = False
) -> list[str]:
    """Validate inputs before analysis; full builds require every study."""
    raw = root / "results" / "raw"
    written = []
    if require_complete:
        required = (
            "native_baselines.jsonl",
            "noise_study.jsonl",
            "equivalence_estimators.jsonl",
            "equivalence_sampling.json",
            "adapter_mutation.csv",
            "adapter_mutation_impact.csv",
            "adapter_mutation_impact.jsonl",
            "extensibility.csv",
            "overhead_study.json",
            "scaling_study.json",
        )
        missing = [name for name in required if not (raw / name).is_file()]
        if missing:
            raise ValueError(f"missing studies: {', '.join(missing)}")
    # Validate the entire expected populations before writing any statistics.
    from aeonq.benchmarks.validation import (
        validate_extended,
        validate_sampling,
        validate_mutation_impact,
        validate_mutation_battery,
    )

    for study in ("native_baselines", "noise_study", "equivalence_estimators"):
        path = raw / f"{study}.jsonl"
        if path.exists():
            metadata = json.loads(
                path.with_suffix(".environment.json").read_text(encoding="utf-8")
            )
            planned = json.loads(
                (root / "experiment_manifest.yaml").read_text(encoding="utf-8")
            )["extended_studies"][study]["repetitions"]
            if metadata.get("repetitions") != planned:
                raise ValueError(f"{study}: repetitions disagree with plan")
            if (
                not isinstance(metadata.get("repetitions"), int)
                or metadata["repetitions"] < 1
            ):
                raise ValueError(f"{study}: missing planned repetition count")
            validate_extended(read_jsonl(path), study, metadata["repetitions"])
    if (raw / "equivalence_sampling.json").exists():
        validate_sampling(
            json.loads((raw / "equivalence_sampling.json").read_text(encoding="utf-8"))
        )
    if (raw / "adapter_mutation_impact.jsonl").exists():
        validate_mutation_impact(
            read_jsonl(raw / "adapter_mutation_impact.jsonl"),
            _read_csv(raw / "adapter_mutation_impact.csv"),
        )
    if (raw / "adapter_mutation.csv").exists():
        validate_mutation_battery(_read_csv(raw / "adapter_mutation.csv"))
        from aeonq.benchmarks.evidence import verify_source_archive

        verify_source_archive(
            json.loads(
                (raw / "adapter_mutation.environment.json").read_text(encoding="utf-8")
            )
        )

    def emit(name: str, rows: list[dict]) -> None:
        write_csv(tables / name, rows)
        written.append(name)

    if (raw / "native_baselines.jsonl").exists():
        detail = native_baseline_rows(read_jsonl(raw / "native_baselines.jsonl"))
        emit("table_9_native_baselines.csv", detail)
        emit("table_9b_native_baseline_summary.csv", native_summary_rows(detail))
    if (raw / "overhead_study.json").exists():
        construction, breakdown = overhead_rows(
            json.loads((raw / "overhead_study.json").read_text(encoding="utf-8"))
        )
        emit("table_10_construction_overhead.csv", construction)
        emit("table_10b_estimator_overhead.csv", breakdown)
    if (raw / "scaling_study.json").exists():
        emit(
            "table_11_scaling.csv",
            json.loads((raw / "scaling_study.json").read_text(encoding="utf-8"))[
                "rows"
            ],
        )
    if (raw / "adapter_mutation.csv").exists() and (
        raw / "adapter_mutation_impact.csv"
    ).exists():
        emit(
            "table_12_adapter_mutation.csv",
            mutation_rows(
                _read_csv(raw / "adapter_mutation.csv"),
                _read_csv(raw / "adapter_mutation_impact.csv"),
            ),
        )
    if (raw / "noise_study.jsonl").exists():
        emit("table_13_noise.csv", noise_rows(read_jsonl(raw / "noise_study.jsonl")))
    if (raw / "equivalence_sampling.json").exists():
        emit(
            "table_14_equivalence_sampling.csv",
            equivalence_sampling_rows(
                json.loads(
                    (raw / "equivalence_sampling.json").read_text(encoding="utf-8")
                )
            ),
        )
    if (raw / "equivalence_estimators.jsonl").exists():
        emit(
            "table_14b_equivalence_estimators.csv",
            equivalence_estimator_rows(
                read_jsonl(raw / "equivalence_estimators.jsonl")
            ),
        )
    if (raw / "extensibility.csv").exists():
        emit("table_15_extensibility.csv", _read_csv(raw / "extensibility.csv"))
    return written
