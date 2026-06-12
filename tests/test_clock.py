from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from calendai.core.clock import FrozenClock, SystemClock


def test_frozen_clock_is_stable():
    clock = FrozenClock(datetime(2026, 6, 15, 9, 0, tzinfo=UTC))
    assert clock.now() == clock.now()


def test_frozen_clock_advances():
    clock = FrozenClock(datetime(2026, 6, 15, 9, 0, tzinfo=UTC))
    clock.advance(timedelta(hours=2))
    assert clock.now() == datetime(2026, 6, 15, 11, 0, tzinfo=UTC)


def test_frozen_clock_rejects_naive_datetime():
    with pytest.raises(ValueError):
        FrozenClock(datetime(2026, 6, 15, 9, 0))


def test_frozen_clock_normalizes_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    clock = FrozenClock(datetime(2026, 6, 15, 9, 0, tzinfo=ist))
    assert clock.now() == datetime(2026, 6, 15, 3, 30, tzinfo=UTC)
    assert clock.now().tzinfo == UTC


def test_system_clock_returns_aware_utc():
    now = SystemClock().now()
    assert now.tzinfo == UTC
