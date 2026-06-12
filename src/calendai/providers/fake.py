"""In-memory CalendarProvider for tests and the evaluation suite.

Design goals:
- Deterministic: takes a Clock (FrozenClock in evals) and an optional
  sequential id_factory so event IDs are stable across runs.
- Failure injection: evals queue provider failures per action
  (rate limits, server errors, malformed responses) to prove the agent's
  retry and self-correction behavior. Hooks are built in here from day one
  rather than bolted on at eval time.
- Invite semantics: creating an event with attendees mirrors it onto each
  attendee's calendar under the same event id (like a Google invite), so
  cross-calendar freebusy behaves realistically in multi-user scenarios.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime

from calendai.core.clock import Clock
from calendai.core.models import Event, EventDraft, EventPatch, EventStatus, TimeSlot
from calendai.core.provider import CalendarProvider, NotFoundError


def sequential_ids(prefix: str = "evt") -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"{prefix}_{next(counter):04d}"


class FakeCalendarProvider(CalendarProvider):
    def __init__(self, clock: Clock, id_factory: Callable[[], str] | None = None) -> None:
        self._clock = clock
        self._new_id = id_factory or sequential_ids()
        # calendar_id -> {event_id -> Event}
        self._calendars: dict[str, dict[str, Event]] = defaultdict(dict)
        # action name -> queued exceptions, consumed one per call
        self._failures: dict[str, deque[Exception]] = defaultdict(deque)
        self.call_log: list[str] = []

    # ── failure injection (eval hooks) ────────────────────────────────

    def inject_failure(self, action: str, exc: Exception, times: int = 1) -> None:
        """Queue `exc` to be raised on the next `times` calls of `action`.

        Actions: list_events, get_event, create_event, update_event,
        delete_event, freebusy. Each queued failure is consumed exactly once,
        so a retrying caller succeeds after the queue drains.
        """
        for _ in range(times):
            self._failures[action].append(exc)

    def _gate(self, action: str) -> None:
        self.call_log.append(action)
        if self._failures[action]:
            raise self._failures[action].popleft()

    # ── seeding (test/eval setup, bypasses failure gates) ─────────────

    def seed(self, calendar_id: str, drafts: list[EventDraft]) -> list[Event]:
        return [self._insert(calendar_id, d) for d in drafts]

    # ── CalendarProvider implementation ───────────────────────────────

    def list_events(self, calendar_id: str, start: datetime, end: datetime) -> list[Event]:
        self._gate("list_events")
        events = [
            e
            for e in self._calendars[calendar_id].values()
            if e.status == EventStatus.CONFIRMED and e.start < end and e.end > start
        ]
        return sorted(events, key=lambda e: e.start)

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
            created_at=now,
            updated_at=now,
            **draft.model_dump(),
        )
        self._calendars[calendar_id][event.id] = event
        # Mirror onto attendee calendars (invite semantics).
        for attendee in draft.attendees:
            if attendee.email != calendar_id:
                mirror = event.model_copy(update={"calendar_id": attendee.email})
                self._calendars[attendee.email][event.id] = mirror
        return event

    def update_event(self, calendar_id: str, event_id: str, patch: EventPatch) -> Event:
        self._gate("update_event")
        if event_id not in self._calendars[calendar_id]:
            raise NotFoundError(f"event {event_id} not found in {calendar_id}")
        changes = {k: v for k, v in patch.model_dump().items() if v is not None}
        changes["updated_at"] = self._clock.now()
        updated: Event | None = None
        for cal in self._all_calendars_with(event_id):
            cal_id = cal[event_id].calendar_id
            cal[event_id] = cal[event_id].model_copy(update=changes)
            if cal_id == calendar_id:
                updated = cal[event_id]
        assert updated is not None
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
        self._gate("freebusy")
        result: dict[str, list[TimeSlot]] = {}
        for cal_id in calendar_ids:
            busy = [
                TimeSlot(start=max(e.start, start), end=min(e.end, end))
                for e in self._calendars[cal_id].values()
                if e.status == EventStatus.CONFIRMED and e.start < end and e.end > start
            ]
            result[cal_id] = _merge_slots(sorted(busy, key=lambda s: s.start))
        return result

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
