"""Declarative scenario schema + YAML loader.

Datetimes are written as ISO-8601 with an offset (e.g. 2026-06-16T10:00:00+05:30);
the UtcDatetime validator normalizes them to UTC so every comparison is in UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from calendai.core.models import UtcDatetime

# Canonical frozen "now" for all evals: Monday 2026-06-15 09:00 IST == 03:30 UTC,
# identical to the unit-test suite so reasoning about times is shared.
DEFAULT_FROZEN_NOW = datetime(2026, 6, 15, 3, 30, tzinfo=UTC)

# Maps a scenario's symbolic error name to a provider action failure. Kept here
# (not in YAML) so scenario authors cannot inject arbitrary exception types.
FAILURE_KINDS = ("rate_limit", "server_error", "not_found", "malformed", "auth")


class UserSpec(BaseModel):
    email: str
    display_name: str = ""
    timezone: str = "Asia/Kolkata"


class SeedEvent(BaseModel):
    user: str  # email; the calendar this event is seeded on
    title: str
    start: UtcDatetime
    end: UtcDatetime
    description: str = ""
    attendees: list[str] = Field(default_factory=list)


class SeedFact(BaseModel):
    user: str
    fact_type: Literal["rule", "contact", "preference"]
    key: str
    value: dict[str, Any]
    statement: str


class FailureInjection(BaseModel):
    action: str  # a CalendarProvider method name (validated by the provider)
    error: Literal["rate_limit", "server_error", "not_found", "malformed", "auth"]
    times: int = 1


class Session(BaseModel):
    """One run of the agent. A list of >1 session simulates restarts: the
    calendar persists, the SQLite store (memory + traces) is reopened."""

    user: str
    turns: list[str]


class ExpectedEvent(BaseModel):
    user: str
    present: bool = True  # False asserts NO matching event exists
    title_contains: str | None = None
    start: UtcDatetime | None = None
    end: UtcDatetime | None = None
    attendee: str | None = None  # an email that must be on the matched event


class ExpectedFact(BaseModel):
    user: str
    key: str
    present: bool = True
    value_contains: dict[str, Any] | None = None


class Trajectory(BaseModel):
    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)


class JudgeRubric(BaseModel):
    criterion: str  # a yes/no question about the assistant's reply
    target_turn: int = -1  # index into the flat list of assistant replies


class Expectations(BaseModel):
    events: list[ExpectedEvent] = Field(default_factory=list)
    facts: list[ExpectedFact] = Field(default_factory=list)
    trajectory: Trajectory = Field(default_factory=Trajectory)
    judge: list[JudgeRubric] = Field(default_factory=list)
    final_reply_contains: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    description: str
    tags: list[str] = Field(default_factory=list)
    runs: int = Field(default=2, ge=1, le=5)
    frozen_now: UtcDatetime = DEFAULT_FROZEN_NOW
    users: list[UserSpec]
    seed_events: list[SeedEvent] = Field(default_factory=list)
    seed_facts: list[SeedFact] = Field(default_factory=list)
    inject_failures: list[FailureInjection] = Field(default_factory=list)
    sessions: list[Session]
    expect: Expectations = Field(default_factory=Expectations)

    def user_by_email(self, email: str) -> UserSpec:
        for u in self.users:
            if u.email == email:
                return u
        raise KeyError(f"scenario {self.id!r} references unknown user {email!r}")


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Scenario(**data)


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """All *.yaml scenarios in a directory, sorted by id for stable reports."""
    paths = sorted(Path(directory).glob("*.yaml"))
    scenarios = [load_scenario(p) for p in paths]
    ids = [s.id for s in scenarios]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {sorted(duplicates)}")
    return sorted(scenarios, key=lambda s: s.id)
