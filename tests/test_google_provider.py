"""GoogleCalendarProvider tests, fully respx-mocked (no network, no sleeping).

Covers the mapping layer (API items <-> frozen models), the hand-rolled
retry/backoff layer (injectable sleep_fn records delays), the 401
refresh-once flow, and the all-or-nothing freebusy contract.
"""

from __future__ import annotations

import json
import random
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


def make_provider(
    sleeps: list[float] | None = None,
) -> tuple[GoogleCalendarProvider, StubTokens]:
    tokens = StubTokens()
    provider = GoogleCalendarProvider(
        token_provider=tokens.token_provider,
        refresh_fn=tokens.refresh,
        sleep_fn=sleeps.append if sleeps is not None else lambda _s: None,
        rng=random.Random(0),
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


# -- retry layer ----------------------------------------------------------


@respx.mock
def test_429_with_retry_after_is_retried_then_succeeds():
    route = respx.get(EVENTS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2.5"}),
            httpx.Response(200, json={"items": [EVENT_ITEM]}),
        ]
    )
    sleeps: list[float] = []
    provider, _ = make_provider(sleeps=sleeps)

    events = provider.list_events("primary", at(0), at(23))

    assert [e.id for e in events] == ["evt1"]
    assert route.call_count == 2
    assert sleeps == [2.5]  # honored the Retry-After header verbatim


@respx.mock
def test_5xx_retried_to_exhaustion_raises_server_error():
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(503))
    sleeps: list[float] = []
    provider, _ = make_provider(sleeps=sleeps)

    with pytest.raises(ServerError):
        provider.list_events("primary", at(0), at(23))

    assert route.call_count == 3  # max attempts
    assert len(sleeps) == 2  # no sleep after the final failure
    assert 0.5 <= sleeps[0] <= 1.0  # base + jitter within [base, 2*base]
    assert 1.0 <= sleeps[1] <= 2.0


@respx.mock
def test_403_rate_limit_reason_maps_to_rate_limit_and_retries():
    quota_body = {"error": {"errors": [{"reason": "rateLimitExceeded"}]}}
    route = respx.get(EVENTS_URL).mock(
        side_effect=[
            httpx.Response(403, json=quota_body),
            httpx.Response(200, json={"items": []}),
        ]
    )
    sleeps: list[float] = []
    provider, _ = make_provider(sleeps=sleeps)

    assert provider.list_events("primary", at(0), at(23)) == []
    assert route.call_count == 2
    assert sleeps == [1.0]  # default retry_after when no Retry-After header


@respx.mock
def test_403_other_reason_raises_auth_error_without_retry():
    forbidden = {"error": {"errors": [{"reason": "forbidden"}]}}
    route = respx.get(EVENTS_URL).mock(return_value=httpx.Response(403, json=forbidden))
    provider, _ = make_provider()

    with pytest.raises(AuthError):
        provider.list_events("primary", at(0), at(23))
    assert route.call_count == 1


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
