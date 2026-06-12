"""In-memory CalendarProvider for tests and the evaluation suite.

Design goals:
- Deterministic: takes a Clock (FrozenClock in evals) and an optional
  sequential id_factory so event IDs are stable across runs.
- Failure injection: evals queue provider failures per action
  (rate limits, server errors, malformed responses) to prove the agent's
  retry and self-correction behavior. Hooks are built in here from day one
  rather than bolted on at eval time.
- Invite semantics, mirroring Google:
  * creating an event with attendees places a copy (same event id, own
    calendar_id, shared organizer) on each attendee's calendar;
  * patching attendees adds/removes those copies accordingly;
  * an attendee who DECLINED is not considered busy for that event;
  * tentative events still count as busy, cancelled ones never do.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

from calendai.core.clock import Clock
from calendai.core.models import (
    AttendeeResponseStatus,
    Event,
    EventDraft,
    EventPatch,
    EventStatus,
    TimeSlot,
)
from calendai.core.provider import CalendarProvider, NotFoundError

_ACTIONS = frozenset(
    {"list_events", "get_event", "create_event", "update_event", "delete_event", "freebusy"}
)


def sequential_ids(prefix: str = "evt") -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"{prefix}_{next(counter):04d}"


class FakeCalendarProvider(CalendarProvider):
    def __init__(self, clock: Clock, id_factory: Callable[[], str] | None = None) -> None:
        self._clock = clock
        self._new_id = id_factory or sequential_ids()
        # calendar_id -> {event_id -> Event}
        self._calendars: dict[str, dict[str, Event]] = defaultdict(dict)
        # action name -> queued exception factories, consumed one per call
        self._failures: dict[str, deque[Callable[[], Exception]]] = defaultdict(deque)
        self.call_log: list[str] = []

    # -- failure injection (eval hooks) ---------------------------------

    def inject_failure(
        self, action: str, exc: Exception | Callable[[], Exception], times: int = 1
    ) -> None:
        """Queue a failure for the next `times` calls of `action`.

        `action` must be one of the CalendarProvider methods (validated, so
        eval typos fail loudly instead of silently doing nothing). `exc` may
        be an exception instance or a zero-arg factory; factories produce a
        fresh instance per failure so tracebacks never carry stale state.
        """
        if action not in _ACTIONS:
            raise ValueError(f"unknown provider action {action!r}; valid: {sorted(_ACTIONS)}")
        factory = exc if callable(exc) else (lambda e=exc: e)
        for _ in range(times):
            self._failures[action].append(factory)

    def _gate(self, action: str) -> None:
        self.call_log.append(action)
        if self._failures[action]:
            raise self._failures[action].popleft()()

    # -- seeding (test/eval setup, bypasses failure gates) ---------------

    def seed(self, calendar_id: str, drafts: list[EventDraft]) -> list[Event]:
        return [self._insert(calendar_id, d) for d in drafts]

    # -- CalendarProvider implementation ---------------------------------

    def list_events(self, calendar_id: str, start: datetime, end: datetime) -> list[Event]:
        self._gate("list_events")
        events = [
            e
            for e in self._calendars[calendar_id].values()
            if e.status != EventStatus.CANCELLED and e.start < end and e.end > start
        ]
        return sorted(events, key=lambda e: (e.start, e.id))

    def get_event(self, calendar_id: str, event_id: str) -> Event:
        self._gate("get_event")
        event = self._calendars[calendar_id].get(event_id)
        if event is None or event.status == EventStatus.CANCELLED:
            raise NotFoundError(f"event {event_id} not found in {calendar_id}")
        return event

    def create_event(self, calendar_id: str, draft: EventDraft) -> Event:
        self._gate("create_event")
        return self._insert(calendar_id, draft)

    def _insert(self, calendar_id: str, draft: EventDraft) -> Event:
        now = self._clock.now()
        event = Event(
            id=self._new_id(),
            calendar_id=calendar_id,
            organizer=calendar_id,
            created_at=now,
            updated_at=now,
            **draft.model_dump(),
        )
        self._calendars[calendar_id][event.id] = event
        self._sync_attendee_mirrors(event)
        return event

    def update_event(self, calendar_id: str, event_id: str, patch: EventPatch) -> Event:
        self._gate("update_event")
        current = self._calendars[calendar_id].get(event_id)
        if current is None:
            raise NotFoundError(f"event {event_id} not found in {calendar_id}")

        # Collect changes keeping model objects intact (no dict round-trip).
        changes: dict[str, Any] = {}
        for field in ("title", "start", "end", "description", "attendees"):
            value = getattr(patch, field)
            if value is not None:
                changes[field] = value

        # Validate the MERGED interval - model validation cannot see the
        # stored endpoint when only one of start/end is patched.
        new_start = changes.get("start", current.start)
        new_end = changes.get("end", current.end)
        if new_start >= new_end:
            raise ValueError("event start must be strictly before end after applying patch")

        changes["updated_at"] = self._clock.now()

        old_attendee_emails = {a.email for a in current.attendees}
        updated: Event | None = None
        for cal in self._all_calendars_with(event_id):
            cal_id = cal[event_id].calendar_id
            cal[event_id] = cal[event_id].model_copy(update=changes)
            if cal_id == calendar_id:
                updated = cal[event_id]
        assert updated is not None

        if "attendees" in changes:
            self._reconcile_mirrors(updated, old_attendee_emails)
        return updated

    def delete_event(self, calendar_id: str, event_id: str) -> None:
        self._gate("delete_event")
        if event_id not in self._calendars[calendar_id]:
            raise NotFoundError(f"event {event_id} not found in {calendar_id}")
        for cal in self._all_calendars_with(event_id):
            del cal[event_id]

    def freebusy(
        self, calendar_ids: list[str], start: datetime, end: datetime
    ) -> dict[str, list[TimeSlot]]:
        """Busy periods per calendar. All-or-nothing: a failure injected here
        fails the whole query, matching the provider contract."""
        self._gate("freebusy")
        result: dict[str, list[TimeSlot]] = {}
        for cal_id in calendar_ids:
            busy = [
                TimeSlot(start=max(e.start, start), end=min(e.end, end))
                for e in self._calendars[cal_id].values()
                if e.start < end and e.end > start and self._counts_as_busy(e, cal_id)
            ]
            result[cal_id] = _merge_slots(sorted(busy, key=lambda s: s.start))
        return result

    # -- internals --------------------------------------------------------

    @staticmethod
    def _counts_as_busy(event: Event, calendar_id: str) -> bool:
        """Confirmed and tentative events are busy; cancelled never is; an
        attendee who declined is not busy for that event (Google semantics)."""
        if event.status == EventStatus.CANCELLED:
            return False
        for attendee in event.attendees:
            if (
                attendee.email == calendar_id
                and attendee.response_status == AttendeeResponseStatus.DECLINED
            ):
                return False
        return True

    def _sync_attendee_mirrors(self, event: Event) -> None:
        for attendee in event.attendees:
            if attendee.email != event.organizer:
                mirror = event.model_copy(update={"calendar_id": attendee.email})
                self._calendars[attendee.email][event.id] = mirror

    def _reconcile_mirrors(self, updated: Event, old_attendee_emails: set[str]) -> None:
        new_emails = {a.email for a in updated.attendees}
        for removed in old_attendee_emails - new_emails:
            if removed != updated.organizer:
                self._calendars[removed].pop(updated.id, None)
        self._sync_attendee_mirrors(updated)

    def _all_calendars_with(self, event_id: str) -> list[dict[str, Event]]:
        return [cal for cal in self._calendars.values() if event_id in cal]


def _merge_slots(slots: list[TimeSlot]) -> list[TimeSlot]:
    """Merge overlapping/adjacent busy periods (input must be sorted by start)."""
    merged: list[TimeSlot] = []
    for slot in slots:
        if merged and slot.start <= merged[-1].end:
            if slot.end > merged[-1].end:
                merged[-1] = TimeSlot(start=merged[-1].start, end=slot.end)
        else:
            merged.append(slot)
    return merged
