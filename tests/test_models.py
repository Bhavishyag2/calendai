from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from calendai.core.models import EventDraft, EventPatch, TimeSlot

AWARE = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)
IST = timezone(timedelta(hours=5, minutes=30))


def test_draft_rejects_naive_datetimes():
    with pytest.raises(ValidationError, match="timezone-aware"):
        EventDraft(title="x", start=datetime(2026, 6, 15, 5), end=AWARE)
    with pytest.raises(ValidationError, match="timezone-aware"):
        EventDraft(title="x", start=AWARE, end=datetime(2026, 6, 15, 6))


def test_draft_normalizes_to_utc():
    draft = EventDraft(
        title="x",
        start=datetime(2026, 6, 15, 10, 30, tzinfo=IST),  # == 05:00 UTC
        end=datetime(2026, 6, 15, 11, 30, tzinfo=IST),
    )
    assert draft.start == AWARE
    assert draft.start.tzinfo == UTC


def test_draft_rejects_inverted_interval():
    with pytest.raises(ValidationError, match="strictly before"):
        EventDraft(title="x", start=AWARE, end=AWARE)
    with pytest.raises(ValidationError, match="strictly before"):
        EventDraft(title="x", start=AWARE, end=AWARE - timedelta(hours=1))


def test_timeslot_validation():
    with pytest.raises(ValidationError, match="strictly before"):
        TimeSlot(start=AWARE, end=AWARE)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TimeSlot(start=datetime(2026, 6, 15, 5), end=AWARE)


def test_patch_rejects_naive_and_inverted_when_both_given():
    with pytest.raises(ValidationError, match="timezone-aware"):
        EventPatch(start=datetime(2026, 6, 15, 5))
    with pytest.raises(ValidationError, match="strictly before"):
        EventPatch(start=AWARE, end=AWARE - timedelta(minutes=30))
    # one-sided patch is fine at model level (merged check happens in provider)
    assert EventPatch(start=AWARE).end is None
