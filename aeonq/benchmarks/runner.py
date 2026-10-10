"""Run manifest-defined trials and retain every result or failure as raw JSONL."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache, partial
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
from random import Random
import subprocess
from typing import Any, TextIO

from aeonq import estimate
from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.benchmarks.evidence import source_snapshot, trial_seed
from aeonq.benchmarks.problems import (
    BenchmarkProblem,
    bernoulli,
    discrete_distribution,
    independent_failures,
    independent_threshold,
    lognormal_tail,
    strike_for_index,
    uniform_marked,
)
from aeonq.scheduling import (
    CostAwareSchedule,
    ExecutionCost,
    ExpectedInformationSchedule,
    FisherInformationSchedule,
    FixedSchedule,
)


# Adapters import their SDK at module load, and SDKs are optional extras, so
# only the backends a manifest names are imported.
BACKENDS = {
    "reference": ("aeonq.adapters.reference", "ReferenceAdapter"),
    "qiskit": ("aeonq.adapters.qiskit", "QiskitAdapter"),
    "cirq": ("aeonq.adapters.cirq", "CirqAdapter"),
    "pennylane": ("aeonq.adapters.pennylane", "PennyLaneAdapter"),
    "qsharp": ("aeonq.adapters.qsharp", "QSharpAdapter"),
}


def _load_backend(name: str) -> BackendAdapter:
    module, class_name = BACKENDS[name]
    adapter: BackendAdapter = getattr(import_module(module), class_name)()
    return adapter


def _installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:  # git is not installed, e.g. in the slim Docker image
        return None
    return git.stdout.strip() if git.returncode == 0 else None


@lru_cache(maxsize=1)
def environment_record() -> dict[str, Any]:
    names = (
        "aeonq",
        "numpy",
        "scipy",
        "qiskit",
        "qiskit-aer",
        "cirq-core",
        "pennylane",
        "qdk",
        "qiskit-algorithms",
    )
    lock_path = Path(__file__).resolve().parents[2] / "requirements-lock.txt"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        **source_snapshot(),
        "requirements_lock_sha256": (
            sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
        ),
        # None records an optional SDK that is not installed.
        "dependencies": {name: _installed_version(name) for name in names},
    }


def open_trial_log(
    output: Path, key_fields: tuple[str, ...], *, resume: bool
) -> tuple[TextIO, set[tuple[Any, ...]]]:
    """Open a JSONL trial log and return it with the keys already recorded.

    Without ``resume`` the file is truncated. With it, complete rows are kept,
    a partial last line left by an interrupted run is dropped, and new rows
    are appended; callers skip every trial whose key is in the returned set.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[Any, ...]] = set()
    if not resume or not output.exists():
        return output.open("w", encoding="utf-8"), done
    kept: list[str] = []
    lines = output.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("malformed record before final line")
            break  # interrupted write of final record only
        kept.append(line)
        key = tuple(row[field] for field in key_fields)
        if key in done:
            raise ValueError("duplicate trial identity in resume log")
        if row.get("error") is not None:
            raise ValueError("cannot resume a log containing execution failures")
        if row.get("environment", {}).get("seed_scheme") != "sha256-condition-v1":
            raise ValueError("cannot resume a log from a different seed scheme")
        done.add(key)
    output.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return output.open("a", encoding="utf-8"), done


def map_trials(
    function: Callable[[Any], Any], tasks: Iterable[Any], *, workers: int = 1
) -> Iterator[Any]:
    """Yield ``function(task)`` for every task, in task order.

    With ``workers > 1`` the calls run in separate processes. Trials are
    seeded from their own index, so the results do not depend on how they are
    distributed; only the per-trial wall times change, since they are then
    measured under load. ``function`` must be a picklable top-level function.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    tasks = list(tasks)
    if workers == 1 or len(tasks) < 2:
        yield from map(function, tasks)
        return
    # Small chunks keep workers balanced when trial costs differ by case.
    chunksize = max(1, min(8, len(tasks) // (4 * workers)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(function, tasks, chunksize=chunksize)


def build_problem(spec: dict[str, Any]) -> BenchmarkProblem:
    if spec.get("kind") == "bernoulli":
        return bernoulli(float(spec["amplitude"]))
    if spec.get("kind") == "uniform_threshold":
        return independent_threshold(int(spec["n_qubits"]), int(spec["threshold"]))
    if spec.get("kind") == "uniform_marked":
        return uniform_marked(
            int(spec["n_qubits"]), tuple(int(x) for x in spec["marked"])
        )
    if spec.get("kind") == "distribution":
        return discrete_distribution(
            tuple(float(x) for x in spec["probabilities"]),
            tuple(int(x) for x in spec["marked"]),
        )
    if spec.get("kind") == "reliability":
        return independent_failures(
            tuple(float(x) for x in spec["failure_probabilities"])
        )
    if spec.get("kind") == "lognormal_tail":
        n_qubits = int(spec["n_qubits"])
        return lognormal_tail(
            n_qubits,
            strike_for_index(n_qubits, int(spec["strike_index"])),
            objective_ancilla=bool(spec.get("objective_ancilla", False)),
        )
    if spec.get("kind") == "random_distribution":
        random = Random(int(spec["problem_seed"]))
        weights = [random.random() for _ in range(4)]
        total = sum(weights)
        probabilities = tuple(weight / total for weight in weights)
        marked = tuple(
            sorted(random.sample(range(4), int(spec.get("marked_count", 2))))
        )
        case = discrete_distribution(probabilities, marked)
        return BenchmarkProblem(
            f"random_seed_{spec['problem_seed']}_{case.case_id}",
            case.problem,
            case.a_true,
            case.reference_method,
        )
    raise ValueError(f"unsupported benchmark problem: {spec.get('kind')}")


def _scheduler(name: str, options: dict[str, Any]) -> Any:
    if name == "fixed":
        return FixedSchedule(
            tuple(
                (0, int(options.get("shots_per_round", 128)))
                # One entry per IQAE round, so the schedule cannot run out.
                for _ in range(int(options.get("max_rounds", IQAE.max_rounds)))
            )
        )
    if name == "fisher":
        return FisherInformationSchedule()
    if name == "expected":
        return ExpectedInformationSchedule(grid_points=65)
    if name == "cost_aware":
        return CostAwareSchedule(
            ExecutionCost(oracle_weight=1, submission_overhead=100), grid_points=65
        )
    raise ValueError(f"unsupported benchmark scheduler: {name}")


def _run_trial(
    trial: int,
    *,
    manifest: dict[str, Any],
    environment: dict[str, Any],
    case: BenchmarkProblem,
    label: str,
    algorithm: str,
    options: dict[str, Any],
    scheduler_name: str | None,
    backend_name: str,
    backend: BackendAdapter,
) -> dict[str, Any]:
    seed = trial_seed(
        manifest["profile"],
        case.case_id,
        label,
        backend_name,
        trial,
        int(manifest["base_seed"]),
    )
    row: dict[str, Any] = {
        "profile": manifest["profile"],
        "case_id": case.case_id,
        "a_true": case.a_true,
        "reference_method": case.reference_method,
        "algorithm": algorithm,
        "method_label": label,
        "scheduler": scheduler_name or "default",
        "backend": backend_name,
        "trial": trial,
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "configuration": options,
    }
    try:
        run_options = dict(options)
        if scheduler_name:
            run_options["scheduler"] = _scheduler(scheduler_name, options)
        result = estimate(
            case.problem,
            algorithm=algorithm,
            backend=backend,
            seed=seed,
            **run_options,
        )
        row["result"] = asdict(result)
        row["error"] = None
    except Exception as exc:
        row["result"] = None
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return row


@lru_cache(maxsize=None)
def _process_backend(name: str) -> BackendAdapter:
    """One adapter per backend and worker process, as in a serial run."""
    return _load_backend(name)


@lru_cache(maxsize=64)
def _process_problem(spec_json: str) -> BenchmarkProblem:
    return build_problem(json.loads(spec_json))


def _manifest_task(
    task: tuple[dict[str, Any], dict[str, Any], str, str, str, int],
) -> dict[str, Any]:
    """Run one manifest trial in a worker process; see ``run_manifest``."""
    manifest, environment, spec_json, label, backend_name, trial = task
    algorithm_spec = manifest["algorithms"][label]
    return _run_trial(
        trial,
        manifest=manifest,
        environment=environment,
        case=_process_problem(spec_json),
        label=label,
        algorithm=algorithm_spec.get("method", label),
        options=dict(algorithm_spec.get("options", algorithm_spec)),
        scheduler_name=algorithm_spec.get("scheduler"),
        backend_name=backend_name,
        backend=_process_backend(backend_name),
    )


def run_manifest(
    manifest: dict[str, Any],
    output: Path,
    *,
    max_workers: int = 1,
    processes: int = 1,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run every manifest trial; return the rows written by this call.

    ``max_workers`` threads serve reference-backend trials only. ``processes``
    runs every trial in worker processes instead; rows are written in the
    same order either way. With ``resume``, trials already in ``output`` are
    kept and skipped.
    """
    if not isinstance(processes, int) or processes < 1:
        raise ValueError("processes must be a positive integer")
    from aeonq.benchmarks.validation import validate_manifest, resume_records

    if resume and output.exists():
        validate_manifest(resume_records(output), manifest, complete=False)
    if processes > 1:
        return _run_manifest_in_processes(
            manifest, output, processes=processes, resume=resume
        )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported benchmark manifest schema")
    if int(manifest["repetitions"]) < 1:
        raise ValueError("benchmark repetitions must be positive")
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = environment_record()
    if any(name not in BACKENDS for name in manifest["backends"]):
        raise ValueError("unknown benchmark backend")
    adapters = {name: _load_backend(name) for name in manifest["backends"]}
    rows: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    stream, done = open_trial_log(
        output, ("case_id", "method_label", "backend", "trial"), resume=resume
    )
    try:
        with stream:
            for spec in manifest["problems"]:
                case = build_problem(spec)
                for label, algorithm_spec in manifest["algorithms"].items():
                    algorithm = algorithm_spec.get("method", label)
                    options = dict(algorithm_spec.get("options", algorithm_spec))
                    scheduler_name = algorithm_spec.get("scheduler")
                    for backend_name in manifest["backends"]:
                        execute = partial(
                            _run_trial,
                            manifest=manifest,
                            environment=environment,
                            case=case,
                            label=label,
                            algorithm=algorithm,
                            options=options,
                            scheduler_name=scheduler_name,
                            backend_name=backend_name,
                            backend=adapters[backend_name],
                        )
                        trials = [
                            trial
                            for trial in range(int(manifest["repetitions"]))
                            if (case.case_id, label, backend_name, trial) not in done
                        ]
                        if pool is not None and backend_name == "reference":
                            trial_rows = pool.map(execute, trials)
                        else:
                            trial_rows = map(execute, trials)
                        for row in trial_rows:
                            rows.append(row)
                            stream.write(
                                json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
                            )
                            stream.flush()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return rows


def _run_manifest_in_processes(
    manifest: dict[str, Any], output: Path, *, processes: int, resume: bool
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported benchmark manifest schema")
    if int(manifest["repetitions"]) < 1:
        raise ValueError("benchmark repetitions must be positive")
    if any(name not in BACKENDS for name in manifest["backends"]):
        raise ValueError("unknown benchmark backend")
    environment = environment_record()
    stream, done = open_trial_log(
        output, ("case_id", "method_label", "backend", "trial"), resume=resume
    )
    tasks: list[tuple[dict[str, Any], dict[str, Any], str, str, str, int]] = []
    for spec in manifest["problems"]:
        case_id = build_problem(spec).case_id
        spec_json = json.dumps(spec, sort_keys=True)
        for label in manifest["algorithms"]:
            for backend_name in manifest["backends"]:
                tasks.extend(
                    (manifest, environment, spec_json, label, backend_name, trial)
                    for trial in range(int(manifest["repetitions"]))
                    if (case_id, label, backend_name, trial) not in done
                )
    rows: list[dict[str, Any]] = []
    with stream:
        for row in map_trials(_manifest_task, tasks, workers=processes):
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
    return rows
