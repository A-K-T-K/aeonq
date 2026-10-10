import asyncio
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import pytest

import aeonq.benchmarks.runner as runner
from aeonq.adapters.fake import FakeBackendAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.ir import Gate
from aeonq.runtime.async_executor import AsyncExecutor, BackendTimeout


def test_timed_out_backend_does_not_block_event_loop_shutdown():
    # asyncio.to_thread uses the default executor, which asyncio.run() joins
    # at shutdown; a timeout must not wait for the whole backend call.
    backend = FakeBackendAdapter(("timeout",), delay_seconds=2.0)
    start = perf_counter()
    with pytest.raises(BackendTimeout):
        asyncio.run(
            AsyncExecutor(backend).execute(Gate("X", (0,), 1), 4, timeout_seconds=0.01)
        )
    assert perf_counter() - start < 1.0


def test_runner_import_does_not_require_optional_sdks():
    probe = (
        "import sys, aeonq.benchmarks.runner; "
        "sdks = {'qiskit', 'qiskit_aer', 'cirq', 'pennylane', 'qdk', 'qsharp'}; "
        "print(sorted(sdks & {m.split('.')[0] for m in sys.modules}))"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout
    assert output.strip() == "[]"


def test_environment_record_tolerates_missing_optional_sdk(monkeypatch):
    real_version = runner.version

    def version(name: str) -> str:
        if name == "qdk":
            raise PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(runner, "version", version)
    runner.environment_record.cache_clear()
    try:
        assert runner.environment_record()["dependencies"]["qdk"] is None
    finally:
        runner.environment_record.cache_clear()


def test_environment_record_without_git_executable(monkeypatch):
    # python:3.13-slim (the project Docker image) ships without git.
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    assert runner.environment_record()["git_commit"] is None


def test_fixed_benchmark_schedule_covers_every_default_iqae_round():
    schedule = runner._scheduler("fixed", {"shots_per_round": 64})
    assert len(schedule.entries) == IQAE.max_rounds
