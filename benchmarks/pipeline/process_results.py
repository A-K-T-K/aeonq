"""Derive result tables from raw experiment files only."""

import csv
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.stats import norm, t


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(
    successes: int, total: int, alpha: float = 0.05
) -> tuple[float, float]:
    if total < 1:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (float(max(0, center - half)), float(min(1, center + half)))


def summary_rows(raw: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in raw:
        key = (row["profile"], row["case_id"], row["method_label"], row["backend"])
        groups.setdefault(key, []).append(row)
    output = []
    for (profile, case_id, method, backend), rows in sorted(groups.items()):
        successful = [r for r in rows if r["result"] is not None]
        if not successful:
            output.append(
                {
                    "profile": profile,
                    "case_id": case_id,
                    "method": method,
                    "backend": backend,
                    "n": 0,
                    "failures": len(rows),
                }
            )
            continue
        truth = float(successful[0]["a_true"])
        errors = np.array(
            [r["result"]["estimate"] - truth for r in successful], dtype=float
        )
        shots = [r["result"]["total_shots"] for r in successful]
        queries = [r["result"]["total_oracle_queries"] for r in successful]
        evaluations = [r["result"]["total_circuit_evaluations"] for r in successful]
        powers = [r["result"]["max_power"] for r in successful]
        compile_times = [r["result"]["compile_time"] for r in successful]
        execution_times = [r["result"]["execution_time"] for r in successful]
        elapsed = [r["result"]["wall_clock_time"] for r in successful]

        def mean_resource(name: str) -> float | None:
            values = [r["result"]["resource_metrics"].get(name) for r in successful]
            return (
                mean(values)
                if all(isinstance(value, (int, float)) for value in values)
                else None
            )

        interval_rows = [
            r
            for r in successful
            if r["result"]["interval_type"] != "no inferential interval"
        ]
        coverage_n = len(interval_rows)
        covered = sum(
            r["result"]["confidence_or_credible_interval"][0]
            <= truth
            <= r["result"]["confidence_or_credible_interval"][1]
            for r in interval_rows
        )
        cover_low, cover_high = wilson_interval(covered, coverage_n)
        output.append(
            {
                "profile": profile,
                "case_id": case_id,
                "method": method,
                "backend": backend,
                "a_true": truth,
                "n": len(successful),
                "failures": len(rows) - len(successful),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "sd_error": float(np.std(errors, ddof=1))
                if len(errors) > 1
                else float("nan"),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "coverage_n": coverage_n,
                "coverage_count": covered,
                "empirical_coverage": covered / coverage_n
                if coverage_n
                else float("nan"),
                "coverage_wilson_lower": cover_low,
                "coverage_wilson_upper": cover_high,
                "nominal_coverage": 1
                - float(successful[0]["result"]["requested_alpha"]),
                "convergence_rate": sum(
                    r["result"]["convergence_status"] == "CONVERGED" for r in successful
                )
                / len(successful),
                "mean_shots": mean(shots),
                "median_shots": median(shots),
                "mean_queries": mean(queries),
                "median_queries": median(queries),
                "mean_circuit_evaluations": mean(evaluations),
                "mean_max_power": mean(powers),
                "mean_total_compiled_depth": mean_resource("total_compiled_depth"),
                "mean_total_two_qubit_gates": mean_resource("total_two_qubit_gates"),
                "mean_compile_seconds": mean(compile_times),
                "mean_execution_seconds": mean(execution_times),
                "mean_wall_seconds": mean(elapsed),
            }
        )
    return output


def scheduler_comparison_rows(raw: list[dict]) -> list[dict]:
    """Independent-condition scheduler contrasts, stratified by problem.

    Repetitions are independent across methods; trial numbers are not pairs.
    Equal-weight case means are contrasted using a Welch/Satterthwaite interval.
    """
    groups = {}
    for row in raw:
        if (
            row["method_label"].startswith("iqae_")
            or row["method_label"] == "canonical_qae"
        ):
            if row.get("error") is not None or row["result"] is None:
                raise ValueError("failed scheduler execution")
            key = (row["profile"], row["backend"], row["method_label"])
            groups.setdefault(key, {}).setdefault(row["case_id"], []).append(row)
    output = []
    for (profile, backend, method), cases in sorted(groups.items()):
        baseline = groups.get((profile, backend, "iqae_fixed"))
        if baseline is None:
            continue
        if set(cases) != set(baseline):
            raise ValueError("scheduler problem populations differ")
        matched = [r for rows in cases.values() for r in rows]
        parts, deltas, error_deltas, denominators = [], [], [], []
        for case, rows in cases.items():
            other = baseline[case]
            a = np.array([r["result"]["total_oracle_queries"] for r in rows], float)
            b = np.array([r["result"]["total_oracle_queries"] for r in other], float)
            deltas.append(float(a.mean() - b.mean()))
            error_deltas.append(
                mean(abs(r["result"]["estimate"] - r["a_true"]) for r in rows)
                - mean(abs(r["result"]["estimate"] - r["a_true"]) for r in other)
            )
            if len(a) > 1 and len(b) > 1:
                va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                parts.append(va + vb)
                denominators.append(va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        mu = mean(deltas)
        variance = sum(parts) / len(cases) ** 2
        denominator = sum(denominators) / len(cases) ** 4
        df = variance**2 / denominator if denominator > 0 else float("inf")
        half = (
            float(t.ppf(0.975, df) * np.sqrt(variance))
            if len(parts) == len(cases)
            else float("nan")
        )
        if method == "iqae_fixed":
            mu, half = 0.0, 0.0  # same series, an identity contrast
        intervals = [
            r
            for r in matched
            if r["result"]["interval_type"] != "no inferential interval"
        ]
        depths = [
            r["result"]["resource_metrics"].get("total_compiled_depth") for r in matched
        ]
        output.append(
            {
                "profile": profile,
                "backend": backend,
                "method": method,
                "baseline": "iqae_fixed",
                "independent_n": len(matched),
                "baseline_n": sum(map(len, baseline.values())),
                "mae": mean(
                    abs(r["result"]["estimate"] - r["a_true"]) for r in matched
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["result"]["estimate"] - r["a_true"]) ** 2
                                for r in matched
                            ]
                        )
                    )
                ),
                "empirical_coverage": mean(
                    r["result"]["confidence_or_credible_interval"][0]
                    <= r["a_true"]
                    <= r["result"]["confidence_or_credible_interval"][1]
                    for r in intervals
                )
                if intervals
                else None,
                **{
                    output_key: mean(r["result"][input_key] for r in matched)
                    for output_key, input_key in (
                        ("mean_shots", "total_shots"),
                        ("mean_queries", "total_oracle_queries"),
                        ("mean_max_power", "max_power"),
                        ("mean_circuit_evaluations", "total_circuit_evaluations"),
                        ("mean_wall_seconds", "wall_clock_time"),
                    )
                },
                "mean_total_compiled_depth": mean(depths)
                if all(isinstance(d, (int, float)) for d in depths)
                else None,
                "mean_absolute_error_difference": mean(error_deltas),
                "mean_query_difference": mu,
                "independent_95ci_lower": mu - half,
                "independent_95ci_upper": mu + half,
            }
        )
    return output


def write_validation_reports(
    summary: list[dict], conformance: list[dict], directory: Path
) -> None:
    """Derive short audit reports from the same processed rows as the tables."""
    directory.mkdir(parents=True, exist_ok=True)
    latest_profile = (
        "publication_reference"
        if any(row["profile"] == "publication_reference" for row in summary)
        else "smoke_reference"
    )
    reference = [
        row
        for row in summary
        if row["profile"] == latest_profile
        and row["backend"] == "reference"
        and row.get("coverage_n", 0)
    ]
    methods = sorted({row["method"] for row in reference})
    lines = [
        "# Statistical validation report",
        "",
        f"Profile: `{latest_profile}`. Values below come from `summary.csv`.",
        "Each row identifies the lowest observed case coverage for that method;",
        "Wilson bounds describe trial uncertainty for that case.",
        "",
        "| Method | Lowest-coverage case | Covered / trials | Observed | 95% Wilson interval | Nominal |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for method in methods:
        row = min(
            (r for r in reference if r["method"] == method),
            key=lambda r: r["empirical_coverage"],
        )
        lines.append(
            f"| {method} | {row['case_id']} | {row['coverage_count']} / "
            f"{row['coverage_n']} | {row['empirical_coverage']:.3f} | "
            f"[{row['coverage_wilson_lower']:.3f}, "
            f"{row['coverage_wilson_upper']:.3f}] | "
            f"{row['nominal_coverage']:.3f} |"
        )
    lines.append("")
    (directory / "statistical_validation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    report = [
        "# Local operator conformance report",
        "",
        "The complete measured matrix distances are in `operator_conformance.csv`.",
        "The QDK matrix-dump tolerance is 1e-6; other SDK tolerances are 1e-9.",
        "",
        "| Backend | Passing rows | Maximum exact matrix distance |",
        "| --- | ---: | ---: |",
    ]
    for backend in sorted({row["backend"] for row in conformance}):
        rows = [row for row in conformance if row["backend"] == backend]
        report.append(
            f"| {backend} | {sum(bool(row['exact_pass']) for row in rows)} / "
            f"{len(rows)} | {max(float(row['exact_distance']) for row in rows):.3g} |"
        )
    report.append("")
    (directory / "conformance_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def failure_recovery_rows() -> list[dict]:
    """Execute each fake failure mode and record the actual observed outcome."""
    from aeonq.adapters.fake import FakeBackendAdapter
    from aeonq.ir import Gate, RegisterMap
    from aeonq.runtime.async_executor import AsyncExecutor
    from aeonq.runtime.retry import RetryPolicy
    import asyncio

    rows = []
    for mode in (
        "compilation_failure",
        "timeout",
        "delayed_result",
        "dropped_connection",
        "job_cancellation",
        "interrupted_job",
        "backend_quota",
        "unsupported_circuit",
        "malformed_result",
        "partial_result",
        "missing_shots",
    ):
        backend = FakeBackendAdapter((mode,), delay_seconds=0.001)
        observed = "success"
        attempts = 1
        recovery_action = "none"
        try:
            circuit = backend.compile(backend.lower(Gate("X", (0,), 1)))
            if mode == "timeout":
                recovery_action = "timeout without automatic resubmission"
                raw = asyncio.run(
                    AsyncExecutor(backend, RetryPolicy(max_attempts=2)).execute(
                        circuit, 4, timeout_seconds=0.0001, allow_resubmit=True
                    )
                ).raw
            elif mode == "dropped_connection":
                recovery_action = "explicit idempotent retry"
                execution = asyncio.run(
                    AsyncExecutor(backend, RetryPolicy(max_attempts=2)).execute(
                        circuit, 4, allow_resubmit=True
                    )
                )
                raw, attempts = execution.raw, execution.attempts
            else:
                raw = backend.execute(circuit, 4)
            result = backend.normalize_result(raw, RegisterMap((0,), (0,)))
            if result.shots != 4:
                observed = "shot_count_mismatch"
        except Exception as exc:
            observed = type(exc).__name__
        rows.append(
            {
                "fault": mode,
                "observed_outcome": observed,
                "recovery_action": recovery_action,
                "attempts": attempts,
                "recovered": observed == "success" and mode == "dropped_connection",
                "injected": True,
            }
        )
    return rows
