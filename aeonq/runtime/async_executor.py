"""Common async execution with timeout, cancellation and guarded retries."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import threading
from time import perf_counter
from typing import Any

from aeonq.adapters.base import BackendAdapter
from aeonq.runtime.retry import RetryPolicy


def _run_in_daemon_thread(
    function: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> "asyncio.Future[Any]":
    """Run blocking backend work on its own daemon thread.

    ``asyncio.to_thread`` uses the loop's shared default executor: a timed-out
    call keeps occupying a worker, later calls queue behind it, and
    ``asyncio.run`` waits for it at shutdown. A dedicated thread avoids both.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def settle(result: Any, error: BaseException | None) -> None:
        if future.done():  # the caller timed out or cancelled
            return
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)

    def worker() -> None:
        result: Any = None
        error: BaseException | None = None
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:  # re-raised in the awaiting coroutine
            error = exc
        try:
            loop.call_soon_threadsafe(settle, result, error)
        except RuntimeError:
            # The event loop has closed, so no caller remains to receive it.
            pass

    threading.Thread(target=worker, name="aeonq-backend-execute", daemon=True).start()
    return future


@dataclass(frozen=True)
class AsyncExecutionResult:
    raw: Any
    attempts: int
    execution_time: float


class BackendTimeout(TimeoutError):
    """Execution exceeded the caller's timeout; submission outcome may be unknown."""


class AsyncExecutor:
    def __init__(
        self, backend: BackendAdapter, retry: RetryPolicy = RetryPolicy()
    ) -> None:
        self.backend = backend
        self.retry = retry

    async def execute(
        self,
        compiled_program: Any,
        shots: int,
        *,
        seed: int | None = None,
        timeout_seconds: float | None = None,
        allow_resubmit: bool = False,
    ) -> AsyncExecutionResult:
        if shots < 1 or (timeout_seconds is not None and timeout_seconds <= 0):
            raise ValueError("shots and timeout must be positive")
        start = perf_counter()
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                operation: Awaitable[Any]
                if self.backend.capabilities().supports_async_jobs:
                    operation = self.backend.execute_async(
                        compiled_program, shots, seed=seed
                    )
                else:
                    operation = _run_in_daemon_thread(
                        self.backend.execute, compiled_program, shots, seed=seed
                    )
                raw = await asyncio.wait_for(operation, timeout=timeout_seconds)
                return AsyncExecutionResult(raw, attempt, perf_counter() - start)
            except asyncio.TimeoutError as exc:
                # A timed-out local worker or remote job may still be running.
                # Never resubmit automatically; doing so can double-count shots.
                raise BackendTimeout(
                    "backend execution timed out; outcome unknown"
                ) from exc
            except ConnectionError:
                if not allow_resubmit or attempt == self.retry.max_attempts:
                    raise
                await asyncio.sleep(
                    self.retry.initial_delay_seconds * 2 ** (attempt - 1)
                )
        raise AssertionError("unreachable retry state")
