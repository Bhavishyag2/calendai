from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from calendai.core.clock import FrozenClock
from calendai.db.store import Store
from calendai.providers.fake import FakeCalendarProvider

# Canonical frozen "now" for the whole test suite:
# Monday 2026-06-15 09:00 IST == 03:30 UTC.
FROZEN_NOW = datetime(2026, 6, 15, 3, 30, tzinfo=UTC)


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(FROZEN_NOW)


@pytest.fixture
def provider(clock: FrozenClock) -> FakeCalendarProvider:
    return FakeCalendarProvider(clock)


@pytest.fixture
def store(tmp_path, clock: FrozenClock):
    s = Store(tmp_path / "test.db", clock=clock)
    yield s
    s.close()


def at(hour: int, minute: int = 0, day_offset: int = 0) -> datetime:
    """Helper: a UTC datetime relative to the frozen day."""
    base = FROZEN_NOW.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return base + timedelta(days=day_offset)
