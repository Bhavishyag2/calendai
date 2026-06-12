"""Shared data contracts.

These models are frozen after the Batch 1 review gate — the Google provider
track and the eval-harness track both fork from the shapes defined here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventStatus(StrEnum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class Attendee(BaseModel):
    email: str
    response_status: str = "needsAction"


class EventDraft(BaseModel):
    """What callers provide to create an event. Times are aware datetimes (stored UTC)."""

    title: str
    start: datetime
    end: datetime
    description: str = ""
    attendees: list[Attendee] = Field(default_factory=list)


class Event(EventDraft):
    """A stored calendar event."""

    id: str
    calendar_id: str
    status: EventStatus = EventStatus.CONFIRMED
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EventPatch(BaseModel):
    """Partial update; None means leave the field unchanged."""

    title: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    description: str | None = None
    attendees: list[Attendee] | None = None


class TimeSlot(BaseModel):
    start: datetime
    end: datetime


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
    created_at: datetime | None = None
    updated_at: datetime | None = None


class User(BaseModel):
    id: str
    email: str
    display_name: str = ""
    timezone: str = "Asia/Kolkata"  # IANA name; all storage is UTC, this is for rendering
    created_at: datetime | None = None


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
    started_at: datetime
    ended_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
