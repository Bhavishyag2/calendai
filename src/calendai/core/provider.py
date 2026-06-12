"""Calendar provider contract and error taxonomy.

Frozen after the Batch 1 review gate. GoogleCalendarProvider and
FakeCalendarProvider both implement this interface; the agent and the eval
suite only ever see CalendarProvider, which is what lets evals run
deterministically against the fake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from calendai.core.models import Event, EventDraft, EventPatch, TimeSlot


class ProviderError(Exception):
    """Base class for calendar provider failures."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class RateLimitError(ProviderError):
    """HTTP 429 — caller should back off and retry."""

    def __init__(self, message: str = "rate limited", *, retry_after: float = 1.0) -> None:
        super().__init__(message, retryable=True)
        self.retry_after = retry_after


class ServerError(ProviderError):
    """HTTP 5xx — transient upstream failure, retryable."""

    def __init__(self, message: str = "server error") -> None:
        super().__init__(message, retryable=True)


class AuthError(ProviderError):
    """Token invalid/expired beyond refresh — needs re-authentication."""


class NotFoundError(ProviderError):
    """Event or calendar does not exist."""


class MalformedResponseError(ProviderError):
    """Provider returned something that could not be parsed into our models."""


class CalendarProvider(ABC):
    @abstractmethod
    def list_events(self, calendar_id: str, start: datetime, end: datetime) -> list[Event]:
        """Events overlapping [start, end), sorted by start time."""

    @abstractmethod
    def get_event(self, calendar_id: str, event_id: str) -> Event:
        """Raises NotFoundError if absent."""

    @abstractmethod
    def create_event(self, calendar_id: str, draft: EventDraft) -> Event:
        """Creates on the organizer's calendar; attendees receive the invite."""

    @abstractmethod
    def update_event(self, calendar_id: str, event_id: str, patch: EventPatch) -> Event:
        """Applies non-None fields of patch. Raises NotFoundError if absent."""

    @abstractmethod
    def delete_event(self, calendar_id: str, event_id: str) -> None:
        """Raises NotFoundError if absent."""

    @abstractmethod
    def freebusy(
        self, calendar_ids: list[str], start: datetime, end: datetime
    ) -> dict[str, list[TimeSlot]]:
        """Busy periods per calendar within [start, end), merged and sorted."""
