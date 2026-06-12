"""Time source abstraction so the eval suite can freeze "now".

Everything that needs the current time takes a Clock, never calls
datetime.now() directly — that single rule is what makes the evaluation
pipeline deterministic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Current time as an aware UTC datetime."""


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock(Clock):
    """Deterministic clock for tests and evals.

    Starts at `start` and only moves when advance() is called.
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware datetime")
        self._now = start.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta
