"""Shared data contracts.

These models are frozen after the Batch 1 review gate - the Google provider
track and the eval-harness track both fork from the shapes defined here.

Datetime discipline: every datetime in these models must be timezone-aware;
values are normalized to UTC at validation time. Naive datetimes are rejected
outright so behavior can never depend on the host machine's timezone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field, model_validator


def _require_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (naive datetimes are rejected)")
    return dt.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


class EventStatus(StrEnum):
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class AttendeeResponseStatus(StrEnum):
    NEEDS_ACTION = "needsAction"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"


class Attendee(BaseModel):
    email: str
    response_status: AttendeeResponseStatus = AttendeeResponseStatus.NEEDS_ACTION


class EventDraft(BaseModel):
    """What callers provide to create an event."""

    title: str
    start: UtcDatetime
    end: UtcDatetime
    description: str = ""
    attendees: list[Attendee] = Field(default_factory=list)

    @model_validator(mode="after")
    def _start_before_end(self) -> EventDraft:
        if self.start >= self.end:
            raise ValueError("event start must be strictly before end")
        return self


class Event(EventDraft):
    """A stored calendar event.

    `organizer` is the calendar that owns the event; invite copies on attendee
    calendars share the same `id` but carry their own `calendar_id` (mirroring
    Google semantics). The agent uses organizer to distinguish "my event" from
    "an event I was invited to".
    """

    id: str
    calendar_id: str
    organizer: str
    status: EventStatus = EventStatus.CONFIRMED
    created_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None


class EventPatch(BaseModel):
    """Partial update; None means leave the field unchanged.

    When only one of start/end is given, the merged interval is validated by
    the provider against the stored event (model validation alone cannot see
    the other endpoint).
    """

    title: str | None = None
    start: UtcDatetime | None = None
    end: UtcDatetime | None = None
    description: str | None = None
    attendees: list[Attendee] | None = None

    @model_validator(mode="after")
    def _start_before_end_if_both(self) -> EventPatch:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("event start must be strictly before end")
        return self


class TimeSlot(BaseModel):
    start: UtcDatetime
    end: UtcDatetime

    @model_validator(mode="after")
    def _start_before_end(self) -> TimeSlot:
        if self.start >= self.end:
            raise ValueError("slot start must be strictly before end")
        return self


class FactType(StrEnum):
    RULE = "rule"  # hard constraints the agent must enforce ("no meetings before 10:00")
    CONTACT = "contact"  # who "Alex" is
    PREFERENCE = "preference"  # soft defaults ("meetings default to 30 minutes")


class MemoryFact(BaseModel):
    """One durable fact in a user's profile.

    `key` is the stable identity used for upsert/supersession
    (e.g. "rule:no_meetings_before", "contact:alex", "pref:default_duration").
    `value` is the structured payload the enforcement code reads;
    `statement` is the human-readable line injected into the system prompt;
    `provenance` records what the user said that created this fact.
    """

    id: int | None = None
    user_id: str
    fact_type: FactType
    key: str
    value: dict[str, Any]
    statement: str
    provenance: str
    active: bool = True
    created_at: UtcDatetime | None = None
    updated_at: UtcDatetime | None = None


class User(BaseModel):
    id: str
    email: str
    display_name: str = ""
    timezone: str = "Asia/Kolkata"  # IANA name; all storage is UTC, this is for rendering
    created_at: UtcDatetime | None = None


class ToolOutcome(BaseModel):
    """Uniform envelope every agent tool returns to the LLM."""

    ok: bool
    data: Any = None
    error: str | None = None
    error_type: str | None = None  # "rate_limited" | "not_found" | "rule_violation" | ...


class SpanKind(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_OP = "memory_op"
    DECISION = "decision"


class TraceSpan(BaseModel):
    id: int | None = None
    request_id: str
    kind: SpanKind
    name: str
    started_at: UtcDatetime
    ended_at: UtcDatetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
