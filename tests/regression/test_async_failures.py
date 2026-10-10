import asyncio

import pytest

from aeonq.adapters.fake import (
    BackendQuotaError,
    FakeBackendAdapter,
    UnsupportedCircuitError,
)
from aeonq.ir import Gate, RegisterMap
from aeonq.runtime.async_executor import AsyncExecutor, BackendTimeout
from aeonq.runtime.retry import RetryPolicy
from aeonq.algorithms.base import Experiment
from aeonq import AEProblem
from aeonq.ir import ObjectivePredicate, StatePreparation


def test_async_local_result_and_guarded_retry() -> None:
    backend = FakeBackendAdapter(("dropped_connection", "none"))
    executor = AsyncExecutor(backend, RetryPolicy(max_attempts=2))
    result = asyncio.run(executor.execute(Gate("X", (0,), 1), 7, allow_resubmit=True))
    assert result.attempts == 2
    assert backend.normalize_result(result.raw, RegisterMap((0,), (0,))).shots == 7
    backend = FakeBackendAdapter(("dropped_connection", "none"))
    with pytest.raises(ConnectionError):
        asyncio.run(
            AsyncExecutor(backend, RetryPolicy(max_attempts=2)).execute(
                Gate("X", (0,), 1), 7
            )
        )
    assert backend.calls == 1


def test_timeout_is_never_automatically_retried() -> None:
    backend = FakeBackendAdapter(("timeout", "none"), delay_seconds=0.1)
    with pytest.raises(BackendTimeout):
        asyncio.run(
            AsyncExecutor(backend, RetryPolicy(max_attempts=2)).execute(
                Gate("X", (0,), 1), 4, timeout_seconds=0.01, allow_resubmit=True
            )
        )
    assert backend.calls == 1


def test_delayed_result_and_caller_cancellation() -> None:
    backend = FakeBackendAdapter(("delayed_result",), delay_seconds=0.03)
    result = asyncio.run(
        AsyncExecutor(backend).execute(Gate("X", (0,), 1), 3, timeout_seconds=0.3)
    )
    assert result.attempts == 1

    async def cancel_pending() -> None:
        slow = FakeBackendAdapter(("delayed_result",), delay_seconds=0.1)
        task = asyncio.create_task(AsyncExecutor(slow).execute(Gate("X", (0,), 1), 3))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_pending())


@pytest.mark.parametrize(
    "fault,error",
    [
        ("backend_quota", BackendQuotaError),
        ("unsupported_circuit", UnsupportedCircuitError),
        ("job_cancellation", RuntimeError),
        ("interrupted_job", InterruptedError),
    ],
)
def test_backend_failures_surface(fault: str, error: type) -> None:
    backend = FakeBackendAdapter((fault,))
    with pytest.raises(error):
        asyncio.run(AsyncExecutor(backend).execute(Gate("X", (0,), 1), 3))


@pytest.mark.parametrize(
    "fault", ["malformed_result", "partial_result", "missing_shots"]
)
def test_faulty_results_are_not_silently_accepted(fault: str) -> None:
    backend = FakeBackendAdapter((fault,))
    raw = backend.execute(Gate("X", (0,), 1), 5)
    if fault == "malformed_result":
        with pytest.raises(ValueError):
            backend.normalize_result(raw, RegisterMap((0,), (0,)))
    else:
        assert backend.normalize_result(raw, RegisterMap((0,), (0,))).shots != 5


def test_compilation_failure_surfaces() -> None:
    backend = FakeBackendAdapter(("compilation_failure",))
    with pytest.raises(RuntimeError, match="compilation failure"):
        backend.compile(backend.lower(Gate("X", (0,), 1)))


@pytest.mark.parametrize(
    "fault", ["partial_result", "missing_shots", "malformed_result"]
)
def test_algorithm_experiment_rejects_faulty_counts(fault: str) -> None:
    problem = AEProblem(
        StatePreparation(Gate("X", (0,), 1), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    with pytest.raises(ValueError):
        Experiment(problem, FakeBackendAdapter((fault,))).sample(0, 5)
