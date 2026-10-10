"""Regenerate every raw result, table and figure from the current code.

Phase A runs the timing-sensitive studies one at a time on an otherwise idle
process. Phase B runs the statistical studies concurrently, each spreading
its seeded trials over single-threaded worker processes (``--max-workers``,
default CPU count, shared between the studies); per-trial timings from
phase B are therefore measured under load, and isolated timings come from
phase A. Phase C derives all tables and figures
from the raw files and runs the reproducibility audit. A log of what ran,
when, and with which concurrency is written to environment/regeneration_log.json.

The run can be interrupted and restarted with the same command. Finished
steps are recorded in environment/regeneration_state.json and skipped on
restart; unfinished statistical studies keep the trials already on disk and
run only the missing ones (trial seeds do not depend on the order of
execution). The state file is removed when a run completes, so the next run
starts from scratch. ``--fresh`` discards an interrupted run and starts over.

Run from the repository root:  python benchmarks/pipeline/regenerate_everything.py
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
STATE = ROOT / "benchmarks" / "results" / "environment" / "regeneration_state.json"
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

PHASE_A = [
    ["benchmarks/performance_comparison.py"],
    ["benchmarks/compiler_comparison.py"],
    ["benchmarks/compiler_cross_sdk.py"],
    ["benchmarks/overhead_study.py"],
    ["benchmarks/scaling_study.py"],
    ["benchmarks/extensibility_report.py"],
]

PROFILES_SCRIPT = """
import argparse, json
from pathlib import Path
from aeonq.benchmarks.runner import run_manifest
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('names', nargs='+')
    arguments = parser.parse_args()
    manifest = json.loads(Path('benchmarks/experiment_manifest.yaml').read_text(encoding='utf-8'))
    for name in arguments.names:
        run_manifest(manifest['profiles'][name], Path('benchmarks/results/raw') / f'{name}.jsonl', processes=arguments.workers, resume=arguments.resume)
        print('profile done', name, flush=True)
"""

PROFILES = [
    "publication_reference",
    "publication_cross_sdk",
    "publication_scheduler_qiskit",
    "publication_case_study",
    "publication_case_study_sdk",
    "publication_case_study_qsharp",
]


# Share of the worker budget per concurrent slot, roughly proportional to its
# serial cost.
WORKER_WEIGHTS = {
    "publication_profiles": 1,
    "native_baselines": 7,
    "noise_and_mutation": 2,
    "equivalence": 2,
}


def _worker_counts(budget: int) -> dict[str, int]:
    total = sum(WORKER_WEIGHTS.values())
    return {
        name: max(1, round(budget * weight / total))
        for name, weight in WORKER_WEIGHTS.items()
    }


def _phase_b(resume: bool, workers: dict[str, int]) -> dict[str, list[list[str]]]:
    """Each slot is a sequence of commands; slots run concurrently."""
    flag = ["--resume"] if resume else []

    def parallel(name: str) -> list[str]:
        return [*flag, "--workers", str(workers[name])]

    return {
        "publication_profiles": [
            ["-c", PROFILES_SCRIPT, *parallel("publication_profiles"), *PROFILES]
        ],
        "native_baselines": [
            ["benchmarks/native_baselines.py", *parallel("native_baselines")]
        ],
        # Mutation analysis is deterministic and short; it follows the noise
        # study in the same slot as soon as that finishes.
        "noise_and_mutation": [
            ["benchmarks/noise_study.py", *parallel("noise_and_mutation")],
            ["benchmarks/adapter_mutation.py"],
        ],
        "equivalence": [["benchmarks/equivalence_study.py", *parallel("equivalence")]],
    }


PHASE_C = [
    [
        "benchmarks/pipeline/generate_all_results.py",
        "--profile",
        "publication_reference",
        "--reuse-raw",
    ],
    ["benchmarks/pipeline/audit_reproducibility.py"],
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(fresh: bool) -> dict:
    if fresh or not STATE.exists():
        return {"started_utc": _now(), "sessions": [], "done": {}}
    return json.loads(STATE.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def _run(arguments: list[str], state: dict, phase: str) -> None:
    step = f"{phase}|{arguments[0]}"
    if step in state["done"]:
        print(f"[{phase}] skip {arguments[0]} (finished earlier)", flush=True)
        return
    started, clock = _now(), perf_counter()
    subprocess.run([PYTHON, *arguments], cwd=ROOT, check=True)
    state["done"][step] = {
        "phase": phase,
        "command": arguments[0],
        "started_utc": started,
        "seconds": perf_counter() - clock,
        "concurrent_processes": 1,
    }
    _save_state(state)
    print(
        f"[{phase}] done {arguments[0]} in {state['done'][step]['seconds']:.0f}s",
        flush=True,
    )


def _keep_awake() -> None:
    """Prevent Windows idle sleep for the rest of this process.

    Idle sleep would suspend every running study. The request ends with the
    process; an explicit sleep request still applies.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard an interrupted run and regenerate everything",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="worker processes shared by the parallel statistical studies "
        "(default: CPU count)",
    )
    arguments = parser.parse_args()
    if arguments.max_workers is not None and arguments.max_workers < 1:
        parser.error("--max-workers must be positive")
    _keep_awake()
    resuming = STATE.exists() and not arguments.fresh
    state = _load_state(arguments.fresh)
    state["sessions"].append({"started_utc": _now(), "resumed": resuming})
    _save_state(state)
    if resuming:
        print(f"resuming: {len(state['done'])} steps finished earlier", flush=True)

    for command in PHASE_A:
        _run(command, state, "A: isolated timing")

    phase = "B: concurrent statistics"
    budget = arguments.max_workers or max(len(WORKER_WEIGHTS), os.cpu_count() or 4)
    workers = _worker_counts(budget)
    slots = _phase_b(resuming, workers)
    pending = {
        name: commands
        for name, commands in slots.items()
        if f"{phase}|{name}" not in state["done"]
    }
    for name in slots.keys() - pending.keys():
        print(f"[B] skip {name} (finished earlier)", flush=True)
    env = {**os.environ, **SINGLE_THREAD}
    logs_dir = ROOT / "benchmarks" / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failed: list[str] = []

    def run_slot(name: str, commands: list[list[str]]) -> None:
        # Commands in a slot run in order; slots do not wait for each other.
        started, clock = _now(), perf_counter()
        # Append on resume so the log of the interrupted attempt is kept.
        mode = "a" if resuming else "w"
        with (logs_dir / f"regeneration_{name}.log").open(
            mode, encoding="utf-8"
        ) as stream:
            for command in commands:
                code = subprocess.run(
                    [PYTHON, *command],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                ).returncode
                if code != 0:
                    with lock:
                        failed.append(f"{name} (exit code {code})")
                    return
        with lock:
            state["done"][f"{phase}|{name}"] = {
                "phase": phase,
                "command": name,
                "started_utc": started,
                "seconds": perf_counter() - clock,
                "concurrent_processes": len(pending),
                "worker_processes": workers.get(name, 1),
                "exit_code": 0,
            }
            _save_state(state)
        print(f"[B] done {name} in {perf_counter() - clock:.0f}s", flush=True)

    # Every slot is awaited before a failure is reported, so that the studies
    # that did finish are recorded and are not rerun on restart.
    threads = [threading.Thread(target=run_slot, args=item) for item in pending.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        raise SystemExit(
            f"failed: {', '.join(failed)}; see benchmarks/results/logs. "
            "Rerun the same command to resume."
        )

    for command in PHASE_C:
        _run(command, state, "C: derive and audit")

    destination = (
        ROOT / "benchmarks" / "results" / "environment" / "regeneration_log.json"
    )
    destination.write_text(
        json.dumps(
            {
                "started_utc": state["started_utc"],
                "finished_utc": _now(),
                "cpu_count": os.cpu_count(),
                # More than one session means the run was interrupted and resumed.
                "sessions": state["sessions"],
                "steps": list(state["done"].values()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    STATE.unlink()


if __name__ == "__main__":
    main()
