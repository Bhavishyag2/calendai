"""GoogleCalendarProvider tests, fully respx-mocked (no network, no sleeping).

Covers the mapping layer (API items <-> frozen models), error-taxonomy
mapping (single attempt - retries belong to the Toolbox), the 401
refresh-once flow, and the all-or-nothing freebusy contract.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from calendai.core.models import (
    Attendee,
    AttendeeResponseStatus,
    EventDraft,
    EventPatch,
    EventStatus,
)
from calendai.core.provider import (
    AuthError,
    MalformedResponseError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from calendai.providers.google import BASE_URL, FREEBUSY_URL, GoogleCalendarProvider
from tests.conftest import at

EVENTS_URL = f"{BASE_URL}/calendars/primary/events"

EVENT_ITEM = {
    "id": "evt1",
    "summary": "Standup",
    "description": "Daily sync",
    "status": "confirmed",
    "organizer": {"email": "boss@example.com"},
    "start": {"dateTime": "2026-06-15T10:00:00Z"},
    "end": {"dateTime": "2026-06-15T10:30:00+00:00"},
    "attendees": [
        {"email": "boss@example.com", "responseStatus": "accepted"},
        {"email": "dev@example.com", "responseStatus": "tentative"},
    ],
    "created": "2026-06-01T00:00:00Z",
    "updated": "2026-06-10T00:00:00Z",
}

ALL_DAY_ITEM = {
    "id": "allday1",
    "summary": "Conference",
    "status": "confirmed",
    "start": {"date": "2026-06-15"},
    "end": {"date": "2026-06-16"},
}


class StubTokens:
    """Deterministic token source: token-1, then token-2 after a refresh."""

    def __init__(self) -> None:
        self.current = "token-1"
        self.refresh_calls = 0

    def token_provider(self) -> str:
        return self.current

    def refresh(self) -> str:
        self.refresh_calls += 1
        self.current = f"token-{self.refresh_calls + 1}"
        return self.current


def make_provider() -> tuple[GoogleCalendarProvider, StubTokens]:
    tokens = StubTokens()
    provider = GoogleCalendarProvider(
        token_provider=tokens.token_provider,
        refresh_fn=tokens.refresh,
    )
    return provider, tokens


def request_json(route: respx.Route, call_index: int = -1) -> dict:
    return json.loads(route.calls[call_index].request.content)


# -- mapping roundtrips -------------------------------------------------


@respx.mock
def test_list_events_maps_items_and_skips_all_day():
    route = respx.get(EVENTS_URL).mock(
        return_value=httpx.Response(200, json={"items": [EVENT_ITEM, ALL_DAY_ITEM]})
    )
    provider, _ = make_provider()

    events = provider.list_events("primary", at(0), at(23))

    assert [e.id for e in events] == ["evt1"]  # all-day item skipped
    event = events[0]
    assert event.title == "Standup"
    assert event.description == "Daily sync"
    assert event.calendar_id == "primary"
    assert event.organizer == "boss@example.com"
    assert event.status == EventStatus.CONFIRMED
    assert event.start == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    assert event.end == datetime(2026, 6, 15, 10, 30, tzinfo=UTC)
    assert [(a.email, a.response_status) for a in event.attendees] == [
        ("boss@example.com", AttendeeResponseStatus.ACCEPTED),
        ("dev@example.com", AttendeeResponseStatus.TENTATIVE),
    ]
    assert event.created_at == datetime(2026, 6, 1, tzinfo=UTC)

    params = route.calls.last.request.url.params
    assert params["timeMin"] == "2026-06-15T00:00:00+00:00"
    assert params["timeMax"] == "2026-06-15T23:00:00+00:00"
    assert params["singleEvents"] == "true"
    assert params["orderBy"] == "startTime"
    assert params["maxResults"] == "250"


@respx.mock
def test_get_event_roundtrip_and_auth_header():
    respx.get(f"{EVENTS_URL}/evt1").mock(return_value=httpx.Response(200, json=EVENT_ITEM))
    provider, _ = make_provider()

    event = provider.get_event("primary", "evt1")

    assert event.id == "evt1"
    assert event.organizer == "boss@example.com"
    assert respx.calls.last.request.headers["Authorization"] == "Bearer token-1"


@respx.mock
def test_create_event_sends_google_shape():
    created = dict(EVENT_ITEM, id="evt9")
    route = respx.post(EVENTS_URL).mock(return_value=httpx.Response(200, json=created))
    provider, _ = make_provider()
    draft = EventDraft(
        title="Standup",
        start=at(10),
        end=at(10, 30),
        attendees=[Attendee(email="dev@example.com")],
    )

    event = provider.create_event("primary", draft)

    assert event.id == "evt9"
    body = request_json(route)
    assert body == {
        "summary": "Standup",
        "description": "",
        "start": {"dateTime": "2026-06-15T10:00:00+00:00", "timeZone": "UTC"},
        "end": {"dateTime": "2026-06-15T10:30:00+00:00", "timeZone": "UTC"},
        "attendees": [{"email": "dev@example.com"}],
    }


@respx.mock
def test_update_event_patches_only_non_none_fields():
    updated = dict(EVENT_ITEM, summary="New title")
    route = respx.patch(f"{EVENTS_URL}/evt1").mock(return_value=httpx.Response(200, json=updated))
    provider, _ = make_provider()

    event = provider.update_event("primary", "evt1", EventPatch(title="New title"))

    assert event.title == "New title"
    assert request_json(route) == {"summary": "New title"}


@respx.mock
def test_update_event_validates_merged_interval_against_stored_event():
    respx.get(f"{EVENTS_URL}/evt1").mock(return_value=httpx.Response(200, json=EVENT_ITEM))
    patch_route = respx.patch(f"{EVENTS_URL}/evt1").mock(
        return_value=httpx.Response(200, json=EVENT_ITEM)
    )
    provider, _ = make_provider()

    # Stored event ends 10:30; moving start to 11:00 would invert the interval.
    with pytest.raises(ValueError, match="strictly before end"):
        provider.update_event("primary", "evt1", EventPatch(start=at(11)))
    assert patch_route.call_count == 0

    # Moving start to 10:15 keeps the interval valid; the PATCH goes through.
    provider.update_event("primary", "evt1", EventPatch(start=at(10, 15)))
    assert patch_route.call_count == 1
    assert request_json(patch_route) == {
        "start": {"dateTime": "2026-06-15T10:15:00+00:00", "timeZone": "UTC"}
    }


# -- error taxonomy mapping ----------------------------------------------


@pytest.mark.parametrize("status", [404, 410])
@respx.mock
def test_delete_event_maps_gone_statuses_to_not_found(status: int):
    respx.delete(f"{EVENTS_URL}/evt1").mock(return_value=httpx.Response(status))
    provider, _ = make_provider()
    with pytest.raises(NotFoundError):
        provider.delete_event("primary", "evt1")


@respx.mock
def test_delete_event_success_returns_none():
    respx.delete(f"{EVENTS_URL}/evt1").mock(return_value=httpx.Response(204))
    provider, _ = make_provider()
    assert provider.delete_event("primary", "evt1") is None


@respx.mock
def test_get_event_404_raises_not_found():
    respx.get(f"{EVENTS_URL}/missing").mock(return_value=httpx.Response(404))
    provider, _ = make_provider()
    with pytest.raises(NotFoundError):
        provider.get_event("primary", "missing")


@respx.mock
def test_malformed_json_body_raises_malformed_response():
    respx.get(EVENTS_URL).mock(
        return_value=httpx.Response(
            200, content=b"<html>oops</html>", headers={"Content-Type": "application/json"}
        )
    )
    provider, _ = make_provider()
    with pytest.raises(MalformedResponseError):
        provider.list_events("primary", at(0), at(23))


# -- error mapping (retries are the Toolbox's job: exactly ONE attempt here) --


@respx.mock
def test_429_raises_rate_limit_with_retry_after_no_provider_retry():
    route = respx.get(EVENTS_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "2.5"})
    )
    provider, _ = make_provider()

    with pytest.raises(RateLimitError) as excinfo:
        provider.list_events("primary", at(0), at(23))

    assert excinfo.value.retry_after == 2.5  # toolbox honors this
    assert excinfo.value.retryable is True
    assert route.call_count == 1  # no provider-level retry (no 3x3 multiplication)


@respx.mock
def test_5xx_raises_server_error_single_attempt():
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(503))
    provider, _ = make_provider()

    with pytest.raises(ServerError) as excinfo:
        provider.list_events("primary", at(0), at(23))

    assert excinfo.value.retryable is True
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("reason", ["rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"])
def test_403_rate_limit_reasons_map_to_rate_limit(reason):
    quota_body = {"error": {"errors": [{"reason": reason}]}}
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(403, json=quota_body))
    provider, _ = make_provider()

    with pytest.raises(RateLimitError) as excinfo:
        provider.list_events("primary", at(0), at(23))
    assert excinfo.value.retry_after == 1.0  # default when no Retry-After header
    assert route.call_count == 1


@respx.mock
def test_403_other_reason_raises_auth_error_without_retry():
    forbidden = {"error": {"errors": [{"reason": "forbidden"}]}}
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(403, json=forbidden))
    provider, _ = make_provider()

    with pytest.raises(AuthError):
        provider.list_events("primary", at(0), at(23))
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("body", [[], {"error": []}, {"error": {"errors": "nope"}}, "junk"])
def test_403_with_weird_error_body_degrades_to_auth_error(body):
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(403, json=body))
    provider, _ = make_provider()

    with pytest.raises(AuthError):
        provider.list_events("primary", at(0), at(23))


# -- shape robustness (gate-4 blockers) ---------------------------------------


@respx.mock
def test_naive_datetime_with_explicit_timezone_is_legal():
    # Google may omit the offset when timeZone is explicitly specified
    item = dict(EVENT_ITEM)
    item["start"] = {"dateTime": "2026-06-15T15:30:00", "timeZone": "Asia/Kolkata"}
    item["end"] = {"dateTime": "2026-06-15T16:00:00", "timeZone": "Asia/Kolkata"}
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": [item]}))
    provider, _ = make_provider()

    (event,) = provider.list_events("primary", at(0), at(23))
    assert event.start == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)  # 15:30 IST
    assert event.end == datetime(2026, 6, 15, 10, 30, tzinfo=UTC)


@respx.mock
def test_offset_wins_over_conflicting_timezone():
    # an explicit offset takes precedence over a conflicting timeZone field;
    # Event stores the resulting UTC instant. Locks this decision.
    item = dict(EVENT_ITEM)
    item["start"] = {"dateTime": "2026-06-15T10:00:00+00:00", "timeZone": "Asia/Kolkata"}
    item["end"] = {"dateTime": "2026-06-15T10:30:00+00:00", "timeZone": "Asia/Kolkata"}
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": [item]}))
    provider, _ = make_provider()

    (event,) = provider.list_events("primary", at(0), at(23))
    assert event.start == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)  # offset, not IST


@respx.mock
@pytest.mark.parametrize("items", [None, 42, "nope", {"k": "v"}])
def test_items_not_a_list_is_malformed(items):
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": items}))
    provider, _ = make_provider()

    with pytest.raises(MalformedResponseError):
        provider.list_events("primary", at(0), at(23))


@respx.mock
def test_403_unhashable_reason_degrades_to_auth_error():
    # {"reason": []} is unhashable; must not crash building the reason set
    body = {"error": {"errors": [{"reason": []}, {"reason": 5}]}}
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(403, json=body))
    provider, _ = make_provider()

    with pytest.raises(AuthError):
        provider.list_events("primary", at(0), at(23))


@respx.mock
def test_transport_failure_does_not_chain_original_exception():
    respx.get(EVENTS_URL).mock(side_effect=httpx.ConnectError("down"))
    provider, _ = make_provider()

    with pytest.raises(ServerError) as excinfo:
        provider.list_events("primary", at(0), at(23))
    # `from None`: the httpx exception (whose .request carries the bearer
    # header) is not chained into the traceback
    assert excinfo.value.__cause__ is None
    assert "token-1" not in str(excinfo.value)


@respx.mock
def test_naive_datetime_without_timezone_is_malformed():
    item = dict(EVENT_ITEM)
    item["start"] = {"dateTime": "2026-06-15T15:30:00"}
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": [item]}))
    provider, _ = make_provider()

    with pytest.raises(MalformedResponseError):
        provider.list_events("primary", at(0), at(23))


@respx.mock
def test_unknown_timezone_name_is_malformed():
    item = dict(EVENT_ITEM)
    item["start"] = {"dateTime": "2026-06-15T15:30:00", "timeZone": "Mars/Olympus"}
    item["end"] = {"dateTime": "2026-06-15T16:00:00", "timeZone": "Mars/Olympus"}
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": [item]}))
    provider, _ = make_provider()

    with pytest.raises(MalformedResponseError):
        provider.list_events("primary", at(0), at(23))


@respx.mock
@pytest.mark.parametrize(
    "items",
    [[None], ["junk"], [{"start": "nope", "end": {}}], [{"start": None, "end": None}]],
)
def test_shape_junk_in_items_is_malformed_not_a_crash(items):
    respx.get(EVENTS_URL).mock(return_value=httpx.Response(200, json={"items": items}))
    provider, _ = make_provider()

    with pytest.raises(MalformedResponseError):
        provider.list_events("primary", at(0), at(23))


@respx.mock
def test_get_event_cancelled_status_maps_to_not_found():
    # events.get returns deleted events as status=cancelled, often id-only
    respx.get(f"{EVENTS_URL}/evt1").mock(
        return_value=httpx.Response(200, json={"id": "evt1", "status": "cancelled"})
    )
    provider, _ = make_provider()

    with pytest.raises(NotFoundError):
        provider.get_event("primary", "evt1")


# -- 401 refresh flow -------------------------------------------------------


@respx.mock
def test_401_triggers_one_refresh_then_succeeds():
    route = respx.get(EVENTS_URL).mock(
        side_effect=[
            httpx.Response(401),
            httpx.Response(200, json={"items": []}),
        ]
    )
    provider, tokens = make_provider()

    assert provider.list_events("primary", at(0), at(23)) == []
    assert tokens.refresh_calls == 1
    assert route.calls[0].request.headers["Authorization"] == "Bearer token-1"
    assert route.calls[1].request.headers["Authorization"] == "Bearer token-2"


@respx.mock
def test_401_after_refresh_raises_auth_error():
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(401))
    provider, tokens = make_provider()

    with pytest.raises(AuthError):
        provider.list_events("primary", at(0), at(23))

    assert tokens.refresh_calls == 1  # exactly ONE refresh attempt
    assert route.call_count == 2  # AuthError is not retryable


# -- freebusy ---------------------------------------------------------------


@respx.mock
def test_freebusy_happy_path_merges_and_sorts():
    response = {
        "calendars": {
            "primary": {
                "busy": [
                    {"start": "2026-06-15T11:00:00Z", "end": "2026-06-15T12:00:00Z"},
                    {"start": "2026-06-15T10:00:00Z", "end": "2026-06-15T11:30:00Z"},
                ]
            },
            "dev@example.com": {"busy": []},
        }
    }
    route = respx.post(FREEBUSY_URL).mock(return_value=httpx.Response(200, json=response))
    provider, _ = make_provider()

    result = provider.freebusy(["primary", "dev@example.com"], at(0), at(23))

    assert set(result) == {"primary", "dev@example.com"}
    assert result["dev@example.com"] == []
    assert [(s.start, s.end) for s in result["primary"]] == [
        (at(10), at(12))  # overlapping intervals merged, sorted by start
    ]
    body = request_json(route)
    assert body["items"] == [{"id": "primary"}, {"id": "dev@example.com"}]
    assert body["timeMin"] == "2026-06-15T00:00:00+00:00"


@respx.mock
def test_freebusy_partial_error_raises_provider_error():
    response = {
        "calendars": {
            "primary": {"busy": []},
            "ghost@example.com": {
                "errors": [{"domain": "global", "reason": "notFound"}],
                "busy": [],
            },
        }
    }
    respx.post(FREEBUSY_URL).mock(return_value=httpx.Response(200, json=response))
    provider, _ = make_provider()

    with pytest.raises(ProviderError, match="all-or-nothing"):
        provider.freebusy(["primary", "ghost@example.com"], at(0), at(23))


@respx.mock
def test_freebusy_missing_calendar_in_response_raises_provider_error():
    respx.post(FREEBUSY_URL).mock(
        return_value=httpx.Response(200, json={"calendars": {"primary": {"busy": []}}})
    )
    provider, _ = make_provider()

    with pytest.raises(ProviderError, match="all-or-nothing"):
        provider.freebusy(["primary", "dev@example.com"], at(0), at(23))


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        {"calendars": []},  # list where an object is expected
        {"calendars": {"primary": "junk"}},  # entry is not an object
    ],
)
def test_freebusy_shape_junk_raises_within_taxonomy(response):
    respx.post(FREEBUSY_URL).mock(return_value=httpx.Response(200, json=response))
    provider, _ = make_provider()

    with pytest.raises(ProviderError):  # Malformed or all-or-nothing, never AttributeError
        provider.freebusy(["primary"], at(0), at(23))


@respx.mock
def test_freebusy_non_list_busy_is_malformed():
    response = {"calendars": {"primary": {"busy": {"start": "x"}}}}
    respx.post(FREEBUSY_URL).mock(return_value=httpx.Response(200, json=response))
    provider, _ = make_provider()

    with pytest.raises(MalformedResponseError):
        provider.freebusy(["primary"], at(0), at(23))
