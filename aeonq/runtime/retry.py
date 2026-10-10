"""Explicit retry policy for submissions known to be safe to repeat."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    initial_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.initial_delay_seconds < 0:
            raise ValueError("retry attempts must be positive and delay nonnegative")
