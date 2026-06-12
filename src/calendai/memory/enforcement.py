"""Code-level enforcement of stored rules - never trust the prompt to remember.

The system prompt also carries the user's rules, but prompts can be ignored,
truncated, or talked around. The RuleEngine re-reads the user's ACTIVE rule
facts from the store on every check (so a rule taught mid-conversation is
enforced on the very next tool call), parses the structured `value` payload -
never the free-text statement - and vetoes non-compliant create/update calls
at the tool layer with error_type="rule_violation".

Rule keys enforceable in code (the registry below). A rule fact with any
other key, or with a malformed value, is skipped here: it still reaches the
model through the system prompt, it just has no code-level teeth.
"""

from __future__ import annotations

from datetime import time as dt_time
from zoneinfo import ZoneInfo

from calendai.core.models import FactType, MemoryFact, User, UtcDatetime
from calendai.db.store import Store

_WEEKDAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def _parse_hhmm(raw: str) -> dt_time:
    hour, minute = raw.strip().split(":")
    return dt_time(int(hour), int(minute))


class RuleEngine:
    """Checks proposed event times against the user's stored rules.

    `check` matches the Toolbox rule_checker hook signature. It reads facts
    fresh from the store on each call - correctness over micro-latency
    (SQLite point reads are ~microseconds).
    """

    def __init__(self, store: Store, user: User) -> None:
        self.store = store
        self.user = user

    # -- the hook -----------------------------------------------------------

    def check(self, action: str, start: UtcDatetime, end: UtcDatetime) -> str | None:
        for fact in self.store.list_facts(self.user.id, FactType.RULE):
            checker = self._CHECKERS.get(fact.key)
            if checker is None:
                continue  # not enforceable in code; the prompt still carries it
            try:
                violation = checker(self, fact, start, end)
            except (KeyError, ValueError, TypeError):
                continue  # malformed value: unenforceable, but never block on it
            if violation:
                return f'Violates stored rule "{fact.statement}": {violation}'
        return None

    # -- per-rule checkers ----------------------------------------------------

    def _tz(self, fact: MemoryFact) -> ZoneInfo:
        name = fact.value.get("timezone") or self.user.timezone
        try:
            return ZoneInfo(name)
        except KeyError:
            return ZoneInfo("UTC")

    def _no_meetings_before(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        tz = self._tz(fact)
        limit = _parse_hhmm(fact.value["time"])
        local = start.astimezone(tz)
        if local.time() < limit:
            return (
                f"it would start at {local.strftime('%H:%M')} {tz.key}, "
                f"before the {fact.value['time']} cutoff."
            )
        return None

    def _no_meetings_after(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        tz = self._tz(fact)
        limit = _parse_hhmm(fact.value["time"])
        local = end.astimezone(tz)
        if local.time() > limit or local.date() > start.astimezone(tz).date():
            return (
                f"it would end at {local.strftime('%H:%M')} {tz.key}, "
                f"after the {fact.value['time']} cutoff."
            )
        return None

    def _no_meetings_on(self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime) -> str | None:
        days = {str(d).lower() for d in fact.value["days"]} & _WEEKDAYS
        if not days:
            raise ValueError("no valid weekday names")
        tz = self._tz(fact)
        weekday = start.astimezone(tz).strftime("%A").lower()
        if weekday in days:
            return f"it falls on a {weekday.capitalize()}, which is blocked."
        return None

    def _max_meeting_minutes(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        limit = int(fact.value["minutes"])
        duration = (end - start).total_seconds() / 60
        if duration > limit:
            return f"it would run {duration:.0f} minutes, over the {limit}-minute cap."
        return None

    _CHECKERS = {
        "rule:no_meetings_before": _no_meetings_before,
        "rule:no_meetings_after": _no_meetings_after,
        "rule:no_meetings_on": _no_meetings_on,
        "rule:max_meeting_minutes": _max_meeting_minutes,
    }
