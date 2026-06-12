from __future__ import annotations

import pytest

from calendai.core.models import (
    Attendee,
    AttendeeResponseStatus,
    EventDraft,
    EventPatch,
)
from calendai.core.provider import (
    MalformedResponseError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from tests.conftest import at

ALICE = "alice@example.com"
BOB = "bob@example.com"


def draft(title: str, start_h: int, end_h: int, attendees: list[str] | None = None) -> EventDraft:
    return EventDraft(
        title=title,
        start=at(start_h),
        end=at(end_h),
        attendees=[Attendee(email=a) for a in (attendees or [])],
    )


# ── CRUD ──────────────────────────────────────────────────────────────


def test_create_and_get(provider):
    event = provider.create_event(ALICE, draft("Standup", 5, 6))
    assert provider.get_event(ALICE, event.id).title == "Standup"
    assert event.created_at == provider._clock.now()


def test_list_window_filters_and_sorts(provider):
    provider.create_event(ALICE, draft("Late", 10, 11))
    provider.create_event(ALICE, draft("Early", 4, 5))
    provider.create_event(ALICE, draft("Outside", 20, 21))
    events = provider.list_events(ALICE, at(3), at(12))
    assert [e.title for e in events] == ["Early", "Late"]


def test_list_includes_partial_overlap(provider):
    provider.create_event(ALICE, draft("Spans", 4, 8))
    assert len(provider.list_events(ALICE, at(7), at(9))) == 1
    assert len(provider.list_events(ALICE, at(8), at(9))) == 0  # [start, end)


def test_update_patches_only_given_fields(provider):
    event = provider.create_event(ALICE, draft("Old title", 5, 6))
    updated = provider.update_event(ALICE, event.id, EventPatch(title="New title"))
    assert updated.title == "New title"
    assert updated.start == at(5)


def test_delete_then_get_raises(provider):
    event = provider.create_event(ALICE, draft("Doomed", 5, 6))
    provider.delete_event(ALICE, event.id)
    with pytest.raises(NotFoundError):
        provider.get_event(ALICE, event.id)


def test_get_unknown_raises_not_found(provider):
    with pytest.raises(NotFoundError):
        provider.get_event(ALICE, "evt_nope")


def test_update_unknown_raises_not_found(provider):
    with pytest.raises(NotFoundError):
        provider.update_event(ALICE, "evt_nope", EventPatch(title="x"))


# ── invite semantics (multi-user) ─────────────────────────────────────


def test_attendee_gets_mirrored_event(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    bobs_copy = provider.get_event(BOB, event.id)
    assert bobs_copy.title == "Sync"
    assert bobs_copy.calendar_id == BOB
    assert bobs_copy.organizer == ALICE  # invite copy keeps organizer identity
    assert event.organizer == ALICE


def test_update_propagates_to_attendee_calendars(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    provider.update_event(ALICE, event.id, EventPatch(title="Moved sync"))
    assert provider.get_event(BOB, event.id).title == "Moved sync"


def test_delete_propagates_to_attendee_calendars(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    provider.delete_event(ALICE, event.id)
    with pytest.raises(NotFoundError):
        provider.get_event(BOB, event.id)


def test_patch_adding_attendee_creates_mirror(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6))
    provider.update_event(ALICE, event.id, EventPatch(attendees=[Attendee(email=BOB)]))
    assert provider.get_event(BOB, event.id).organizer == ALICE


def test_patch_removing_attendee_deletes_mirror(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    provider.update_event(ALICE, event.id, EventPatch(attendees=[]))
    with pytest.raises(NotFoundError):
        provider.get_event(BOB, event.id)
    provider.get_event(ALICE, event.id)  # organizer copy untouched


def test_patch_replacing_attendees_swaps_mirrors(provider):
    carol = "carol@example.com"
    event = provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    provider.update_event(ALICE, event.id, EventPatch(attendees=[Attendee(email=carol)]))
    with pytest.raises(NotFoundError):
        provider.get_event(BOB, event.id)
    assert provider.get_event(carol, event.id).title == "Sync"


def test_update_rejects_inverted_merged_interval(provider):
    event = provider.create_event(ALICE, draft("Sync", 5, 6))
    with pytest.raises(ValueError, match="strictly before"):
        provider.update_event(ALICE, event.id, EventPatch(start=at(7)))  # start after stored end


# ── freebusy ──────────────────────────────────────────────────────────


def test_freebusy_multi_calendar(provider):
    provider.create_event(ALICE, draft("A", 5, 6))
    provider.create_event(BOB, draft("B", 7, 8))
    busy = provider.freebusy([ALICE, BOB], at(0), at(23))
    assert (busy[ALICE][0].start, busy[ALICE][0].end) == (at(5), at(6))
    assert (busy[BOB][0].start, busy[BOB][0].end) == (at(7), at(8))


def test_freebusy_merges_overlaps(provider):
    provider.create_event(ALICE, draft("A", 5, 7))
    provider.create_event(ALICE, draft("B", 6, 8))
    busy = provider.freebusy([ALICE], at(0), at(23))[ALICE]
    assert len(busy) == 1
    assert (busy[0].start, busy[0].end) == (at(5), at(8))


def test_freebusy_clamps_to_window(provider):
    provider.create_event(ALICE, draft("Long", 4, 10))
    busy = provider.freebusy([ALICE], at(6), at(8))[ALICE]
    assert (busy[0].start, busy[0].end) == (at(6), at(8))


def test_freebusy_reflects_invites(provider):
    provider.create_event(ALICE, draft("Sync", 5, 6, attendees=[BOB]))
    busy = provider.freebusy([BOB], at(0), at(23))[BOB]
    assert len(busy) == 1


def test_declined_attendee_is_not_busy(provider):
    """Google semantics: declining an invite frees that attendee's slot,
    while the organizer remains busy."""
    declined = Attendee(email=BOB, response_status=AttendeeResponseStatus.DECLINED)
    provider.create_event(
        ALICE,
        EventDraft(title="Sync", start=at(5), end=at(6), attendees=[declined]),
    )
    busy = provider.freebusy([ALICE, BOB], at(0), at(23))
    assert len(busy[ALICE]) == 1
    assert busy[BOB] == []


# ── failure injection ─────────────────────────────────────────────────


def test_injected_rate_limit_consumed_once(provider):
    provider.inject_failure("create_event", RateLimitError(retry_after=0.1))
    with pytest.raises(RateLimitError):
        provider.create_event(ALICE, draft("Retry me", 5, 6))
    event = provider.create_event(ALICE, draft("Retry me", 5, 6))  # second call succeeds
    assert event.title == "Retry me"


def test_injected_failures_are_per_action(provider):
    provider.inject_failure("delete_event", ServerError())
    provider.create_event(ALICE, draft("Safe", 5, 6))  # unaffected action


def test_inject_multiple_failures(provider):
    provider.inject_failure("list_events", ServerError(), times=2)
    for _ in range(2):
        with pytest.raises(ServerError):
            provider.list_events(ALICE, at(0), at(23))
    assert provider.list_events(ALICE, at(0), at(23)) == []


def test_injected_malformed_response(provider):
    provider.inject_failure("freebusy", MalformedResponseError("garbage payload"))
    with pytest.raises(MalformedResponseError):
        provider.freebusy([ALICE], at(0), at(23))


def test_deterministic_ids(provider):
    e1 = provider.create_event(ALICE, draft("First", 5, 6))
    e2 = provider.create_event(ALICE, draft("Second", 7, 8))
    assert (e1.id, e2.id) == ("evt_0001", "evt_0002")


def test_inject_failure_rejects_unknown_action(provider):
    with pytest.raises(ValueError, match="unknown provider action"):
        provider.inject_failure("create_evnet", ServerError())  # typo fails loudly


def test_inject_failure_factory_yields_fresh_instances(provider):
    provider.inject_failure("list_events", lambda: ServerError("fresh"), times=2)
    captured = []
    for _ in range(2):
        with pytest.raises(ServerError) as exc_info:
            provider.list_events(ALICE, at(0), at(23))
        captured.append(exc_info.value)
    assert captured[0] is not captured[1]
