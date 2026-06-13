"""Google Calendar provider speaking raw REST over httpx.

Architecture decision (documented for the Batch 4 review): this module
deliberately does NOT use google-api-python-client.

- Raw REST keeps the dependency tree light; we need exactly six endpoints.
- Every request goes through plain httpx, so the full error taxonomy
  (429 backoff, 5xx exhaustion, 401 refresh, partial freebusy failures)
  is mockable with respx and provable in tests without network access.
- Retry behavior stays visible and under our control instead of being
  buried inside googleapiclient internals.

Retry ownership: the TOOLBOX owns retries (backoff + jitter, honors
retry_after) because it is provider-agnostic and is the layer evals
exercise with failure injection. This provider performs each logical call
exactly ONCE - its only second request is the single 401 token-refresh
resend, which is auth recovery, not a retry. (An earlier revision retried
here too; gate review flagged the 3x3 retry multiplication.)

Error mapping:
- 401: refresh the token once and resend; a second 401 raises AuthError
  (re-authentication required).
- 403: RateLimitError when the body reason is rateLimitExceeded,
  userRateLimitExceeded or quotaExceeded, otherwise AuthError.
- 404 and 410: NotFoundError (Google returns 410 Gone for deleted events).
  (Revisit 410 if syncToken/updatedMin sync is ever added - there it means
  "full sync required".)
- 429: RateLimitError carrying the Retry-After header (default 1.0s).
- 5xx and transport failures: ServerError (retryable by the toolbox).
- 200 with an unparseable or shape-invalid body: MalformedResponseError.

Known limitation (user-visible): all-day events carry date-only start/end,
which the aware-datetime Event model cannot represent; list_events SKIPS
them. The list_events tool description discloses this to the agent.

Retry idempotency policy (explicit): the Toolbox retries retryable errors
(429/5xx/transport). DELETE is idempotent (a re-delete -> 404/410 ->
NotFoundError, surfaced) and PATCH carries absolute field values (re-applying
is a no-op), so both are retry-safe. CREATE (POST) is NOT idempotent: a retry
after an AMBIGUOUS failure (5xx/transport where the write may have landed)
could duplicate an event. In this implementation that residual risk is accepted
and documented rather than solved; the production fix is a client-supplied
idempotency key (Google supports a caller-set event id on insert).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from calendai.core.models import (
    Attendee,
    AttendeeResponseStatus,
    Event,
    EventDraft,
    EventPatch,
    EventStatus,
    TimeSlot,
)
from calendai.core.provider import (
    AuthError,
    CalendarProvider,
    MalformedResponseError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
)

BASE_URL = "https://www.googleapis.com/calendar/v3"
FREEBUSY_URL = f"{BASE_URL}/freeBusy"

# quotaExceeded included: Google documents Calendar usage limits as
# HTTP 403 reason=quotaExceeded (gate-4 review citation).
_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded"})


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _api_time(dt: datetime) -> dict[str, str]:
    return {"dateTime": _rfc3339(dt), "timeZone": "UTC"}


def _parse_retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 1.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def _parse_optional_dt(raw: str | None) -> datetime | None:
    return datetime.fromisoformat(raw) if raw else None


def _parse_api_datetime(time_obj: dict[str, Any]) -> datetime:
    """Parse one Google start/end object. Per the API contract, dateTime may
    omit the UTC offset when an explicit timeZone field is present."""
    parsed = datetime.fromisoformat(time_obj["dateTime"])
    if parsed.tzinfo is None:
        tz_name = time_obj.get("timeZone")
        if not tz_name or not isinstance(tz_name, str):
            raise ValueError(f"dateTime {time_obj['dateTime']!r} has no offset and no timeZone")
        parsed = parsed.replace(tzinfo=ZoneInfo(tz_name))  # bad name -> KeyError -> malformed
    return parsed


def _attendee_from_item(raw: dict[str, Any]) -> Attendee:
    return Attendee(
        email=raw["email"],
        response_status=AttendeeResponseStatus(raw.get("responseStatus", "needsAction")),
    )


def _draft_to_body(draft: EventDraft) -> dict[str, Any]:
    return {
        "summary": draft.title,
        "description": draft.description,
        "start": _api_time(draft.start),
        "end": _api_time(draft.end),
        "attendees": [{"email": a.email} for a in draft.attendees],
    }


def _patch_to_body(patch: EventPatch) -> dict[str, Any]:
    """Serialize only the non-None fields of the patch (partial update)."""
    body: dict[str, Any] = {}
    if patch.title is not None:
        body["summary"] = patch.title
    if patch.description is not None:
        body["description"] = patch.description
    if patch.start is not None:
        body["start"] = _api_time(patch.start)
    if patch.end is not None:
        body["end"] = _api_time(patch.end)
    if patch.attendees is not None:
        body["attendees"] = [{"email": a.email} for a in patch.attendees]
    return body


def _busy_to_slots(busy: list[dict[str, str]]) -> list[TimeSlot]:
    """Parse one calendar's busy intervals, then sort and merge them."""
    try:
        slots = [
            TimeSlot(start=datetime.fromisoformat(b["start"]), end=datetime.fromisoformat(b["end"]))
            for b in busy
        ]
    except (KeyError, ValueError, TypeError) as exc:
        raise MalformedResponseError(f"could not parse freebusy intervals: {exc}") from exc
    slots.sort(key=lambda s: s.start)
    merged: list[TimeSlot] = []
    for slot in slots:
        if merged and slot.start <= merged[-1].end:
            if slot.end > merged[-1].end:
                merged[-1] = TimeSlot(start=merged[-1].start, end=slot.end)
        else:
            merged.append(slot)
    return merged


class GoogleCalendarProvider(CalendarProvider):
    """CalendarProvider backed by the Google Calendar v3 REST API.

    Auth is fully decoupled from storage: the constructor takes a
    token_provider callable returning the current access token and a
    refresh_fn callable that forces a refresh and returns the new token
    (GoogleTokenManager in calendai.auth.google_oauth supplies both).
    """

    def __init__(
        self,
        token_provider: Callable[[], str],
        refresh_fn: Callable[[], str],
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._refresh_fn = refresh_fn
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=10.0)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- CalendarProvider implementation ---------------------------------

    def list_events(self, calendar_id: str, start: datetime, end: datetime) -> list[Event]:
        url = f"{BASE_URL}/calendars/{quote(calendar_id, safe='')}/events"
        params: dict[str, Any] = {
            "timeMin": _rfc3339(start),
            "timeMax": _rfc3339(end),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": 250,
        }
        events: list[Event] = []
        page_token: str | None = None
        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token
            data = self._json(self._request("GET", url, params=page_params))
            items = data.get("items", [])
            if not isinstance(items, list):
                raise MalformedResponseError("events 'items' is not a list")
            for item in items:
                event = self._event_from_item(item, calendar_id)
                if event is not None:  # all-day events are skipped
                    events.append(event)
            page_token = data.get("nextPageToken")
            if not page_token:
                return events

    def get_event(self, calendar_id: str, event_id: str) -> Event:
        data = self._json(self._request("GET", self._event_url(calendar_id, event_id)))
        # events.get returns deleted events with status=cancelled (sometimes
        # with ONLY the id populated) instead of a 404 - normalize that
        if data.get("status") == "cancelled":
            raise NotFoundError(f"event {event_id!r} is cancelled/deleted")
        return self._require_timed_event(data, calendar_id, event_id)

    def create_event(self, calendar_id: str, draft: EventDraft) -> Event:
        url = f"{BASE_URL}/calendars/{quote(calendar_id, safe='')}/events"
        data = self._json(self._request("POST", url, json_body=_draft_to_body(draft)))
        return self._require_timed_event(data, calendar_id, data.get("id", "<new>"))

    def update_event(self, calendar_id: str, event_id: str, patch: EventPatch) -> Event:
        self._validate_merged_interval(calendar_id, event_id, patch)
        url = self._event_url(calendar_id, event_id)
        data = self._json(self._request("PATCH", url, json_body=_patch_to_body(patch)))
        return self._require_timed_event(data, calendar_id, event_id)

    def delete_event(self, calendar_id: str, event_id: str) -> None:
        # 404 and 410 both map to NotFoundError in _raise_for_status.
        self._request("DELETE", self._event_url(calendar_id, event_id))

    def freebusy(
        self, calendar_ids: list[str], start: datetime, end: datetime
    ) -> dict[str, list[TimeSlot]]:
        body = {
            "timeMin": _rfc3339(start),
            "timeMax": _rfc3339(end),
            "items": [{"id": cid} for cid in calendar_ids],
        }
        data = self._json(self._request("POST", FREEBUSY_URL, json_body=body))
        calendars = data.get("calendars", {})
        if not isinstance(calendars, dict):
            raise MalformedResponseError("freebusy 'calendars' is not a JSON object")
        result: dict[str, list[TimeSlot]] = {}
        for cid in calendar_ids:
            entry = calendars.get(cid)
            if not isinstance(entry, dict) or entry.get("errors"):
                raise ProviderError(
                    f"freebusy could not be resolved for calendar {cid!r}; refusing to "
                    "return partial availability (all-or-nothing contract)"
                )
            busy = entry.get("busy", [])
            if not isinstance(busy, list):
                raise MalformedResponseError(f"freebusy 'busy' for {cid!r} is not a list")
            result[cid] = _busy_to_slots(busy)
        return result

    # -- request layer: auth + 401 refresh (retries live in the Toolbox) ----

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send with the current token; on 401, refresh ONCE and resend.

        A second 401 means the refreshed token was also rejected, which is
        an AuthError (not retryable) - the user must re-authenticate.
        Retryable failures (429/5xx/transport) raise immediately; backoff
        retries are the Toolbox's job, not ours (no retry multiplication).
        """
        response = self._send_authed(method, url, self._token_provider(), params, json_body)
        if response.status_code == 401:
            response = self._send_authed(method, url, self._refresh_fn(), params, json_body)
            if response.status_code == 401:
                raise AuthError("access token rejected twice; re-authentication required")
        self._raise_for_status(response)
        return response

    def _send_authed(
        self,
        method: str,
        url: str,
        token: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None,
    ) -> httpx.Response:
        try:
            return self._client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            # Class name only AND `from None`: never echo the token, and never
            # chain the original httpx exception (its .request carries the
            # bearer header / form data into any traceback).
            raise ServerError(f"transport failure: {exc.__class__.__name__}") from None

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status == 403:
            raise self._map_403(response)
        if status in (404, 410):
            raise NotFoundError(f"resource not found (HTTP {status})")
        if status == 429:
            raise RateLimitError(retry_after=_parse_retry_after(response))
        if status >= 500:
            raise ServerError(f"google returned HTTP {status}")
        raise ProviderError(f"google returned HTTP {status}")

    @staticmethod
    def _map_403(response: httpx.Response) -> ProviderError:
        # fully defensive: error bodies are attacker/outage-shaped, so any
        # deviation from the documented shape degrades to AuthError
        try:
            data = response.json()
        except ValueError:
            data = None
        error = data.get("error") if isinstance(data, dict) else None
        errors = error.get("errors") if isinstance(error, dict) else None
        if not isinstance(errors, list):
            errors = []
        # only str reasons: a malformed {"reason": []} is unhashable and would
        # otherwise raise TypeError when added to the set
        reasons = {
            e["reason"] for e in errors if isinstance(e, dict) and isinstance(e.get("reason"), str)
        }
        if reasons & _RATE_LIMIT_REASONS:
            return RateLimitError("quota exceeded", retry_after=_parse_retry_after(response))
        return AuthError("permission denied (HTTP 403)")

    # -- response mapping --------------------------------------------------

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise MalformedResponseError("expected a JSON body but could not parse one") from exc
        if not isinstance(data, dict):
            raise MalformedResponseError("expected a JSON object body")
        return data

    def _event_from_item(self, item: Any, calendar_id: str) -> Event | None:
        """Map one API item to an Event; returns None for all-day events.

        All-day events carry date-only start/end, which our aware-datetime
        model cannot represent; list_events skips them gracefully. Any other
        shape deviation raises MalformedResponseError - never a raw Python
        exception escaping the provider taxonomy.
        """
        if not isinstance(item, dict):
            raise MalformedResponseError("event item is not a JSON object")
        start_obj = item.get("start")
        end_obj = item.get("end")
        if not isinstance(start_obj, dict) or not isinstance(end_obj, dict):
            raise MalformedResponseError("event start/end is not a JSON object")
        if "dateTime" not in start_obj or "dateTime" not in end_obj:
            return None  # date-only = all-day
        try:
            organizer = item.get("organizer") or {}
            return Event(
                id=item["id"],
                calendar_id=calendar_id,
                organizer=organizer.get("email", calendar_id)
                if isinstance(organizer, dict)
                else calendar_id,
                title=item.get("summary", ""),
                description=item.get("description", ""),
                start=_parse_api_datetime(start_obj),
                end=_parse_api_datetime(end_obj),
                status=EventStatus(item.get("status", "confirmed")),
                attendees=[_attendee_from_item(a) for a in item.get("attendees", [])],
                created_at=_parse_optional_dt(item.get("created")),
                updated_at=_parse_optional_dt(item.get("updated")),
            )
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            # KeyError: missing field OR unknown ZoneInfo; ValueError covers
            # pydantic validation; Type/AttributeError cover shape junk
            raise MalformedResponseError(f"could not map event item: {exc}") from exc

    def _require_timed_event(self, item: dict[str, Any], calendar_id: str, event_id: str) -> Event:
        event = self._event_from_item(item, calendar_id)
        if event is None:
            raise MalformedResponseError(
                f"event {event_id!r} has date-only start/end (all-day), "
                "which CalendAI does not model"
            )
        return event

    def _validate_merged_interval(self, calendar_id: str, event_id: str, patch: EventPatch) -> None:
        """Validate start < end against the stored event when only one
        endpoint is patched (model validation cannot see the other one)."""
        if (patch.start is None) == (patch.end is None):
            return  # both given (model-validated) or neither (no-op)
        current = self.get_event(calendar_id, event_id)
        new_start = patch.start or current.start
        new_end = patch.end or current.end
        if new_start >= new_end:
            raise ValueError("event start must be strictly before end after applying patch")

    @staticmethod
    def _event_url(calendar_id: str, event_id: str) -> str:
        return (
            f"{BASE_URL}/calendars/{quote(calendar_id, safe='')}/events/{quote(event_id, safe='')}"
        )
