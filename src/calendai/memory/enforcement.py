"""Code-level enforcement of stored rules - never trust the prompt to remember.

The system prompt also carries the user's rules, but prompts can be ignored,
truncated, or talked around. The RuleEngine re-reads the user's ACTIVE rule
facts from the store on every check (so a rule taught mid-conversation is
enforced on the very next tool call), parses the structured `value` payload -
never the free-text statement - and vetoes non-compliant create/update calls
at the tool layer with error_type="rule_violation".

Rules use INTERVAL-OVERLAP semantics: an event violates "no meetings before
10:00" if ANY part of it falls before 10:00 local - a 23:00-09:00 overnight
event cannot sneak past a start-time-only check. Likewise "no Saturdays"
blocks a Friday 23:00 - Saturday 01:00 event.

Rule keys enforceable in code (the registry below). A rule fact with any
other key, or with a malformed value, is skipped here: it still reaches the
model through the system prompt, it just has no code-level teeth. Canonical
keys are shape-validated at write time (memory/validation.py), so malformed
canonical rules should not exist in practice.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from calendai.core.models import FactType, MemoryFact, User, UtcDatetime
from calendai.db.store import Store
from calendai.memory.validation import _WEEKDAYS, parse_hhmm

_MAX_WINDOW_DAYS = 370  # iteration cap for absurdly long events; beyond it we just block


def _overlaps_daily_window(
    local_start: datetime, local_end: datetime, win_start: dt_time, win_end: dt_time | None
) -> bool:
    """Does the half-open event [start, end) touch the daily window
    [win_start, win_end) on any day it spans? win_end=None means midnight
    (end of day). Datetimes must already be in the rule's timezone.

    DST: window boundaries are computed for BOTH folds of an ambiguous local
    time, and an overlap with either counts. During a fall-back hour this is
    deliberately conservative (an event in the repeated hour may be blocked
    even if one wall-clock reading complies) - for hard constraints,
    over-blocking one hour a year beats silently under-blocking.
    """
    if win_start == win_end:
        return False  # empty window, e.g. "no meetings before 00:00"
    if (local_end - local_start).days > _MAX_WINDOW_DAYS:
        return True  # multi-year event: certainly overlaps; avoid silly loops
    tz = local_start.tzinfo
    # Compare epoch INSTANTS, never datetimes: Python compares same-tzinfo
    # aware datetimes by wall clock, ignoring fold - exactly what would hide
    # a violation inside the repeated DST hour.
    ev0, ev1 = local_start.timestamp(), local_end.timestamp()
    day = local_start.date()
    while day <= local_end.date():
        for fold in (0, 1):
            w0 = datetime.combine(day, win_start, tzinfo=tz).replace(fold=fold)
            w1 = (
                datetime.combine(day + timedelta(days=1), dt_time(0, 0), tzinfo=tz)
                if win_end is None
                else datetime.combine(day, win_end, tzinfo=tz)
            ).replace(fold=fold)
            t0, t1 = w0.timestamp(), w1.timestamp()
            if t1 > t0 and ev0 < t1 and ev1 > t0:
                return True
        day += timedelta(days=1)
    return False


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
        # fallback order: rule's own timezone -> user's timezone -> UTC.
        # TypeError: legacy/direct writes may hold non-string values; write
        # validation rejects those now, but reads must stay robust.
        for name in (fact.value.get("timezone"), self.user.timezone):
            if name:
                try:
                    return ZoneInfo(name)
                except (KeyError, ValueError, TypeError):
                    continue
        return ZoneInfo("UTC")

    def _no_meetings_before(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        tz = self._tz(fact)
        limit = parse_hhmm(fact.value["time"])
        local_start, local_end = start.astimezone(tz), end.astimezone(tz)
        if _overlaps_daily_window(local_start, local_end, dt_time(0, 0), limit):
            return (
                f"it would occupy time before the {fact.value['time']} {tz.key} cutoff "
                f"(event runs {local_start.strftime('%a %H:%M')} - "
                f"{local_end.strftime('%a %H:%M')} local)."
            )
        return None

    def _no_meetings_after(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        tz = self._tz(fact)
        limit = parse_hhmm(fact.value["time"])
        local_start, local_end = start.astimezone(tz), end.astimezone(tz)
        if _overlaps_daily_window(local_start, local_end, limit, None):
            return (
                f"it would occupy time after the {fact.value['time']} {tz.key} cutoff "
                f"(event runs {local_start.strftime('%a %H:%M')} - "
                f"{local_end.strftime('%a %H:%M')} local)."
            )
        return None

    def _no_meetings_on(self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime) -> str | None:
        days = {str(d).strip().lower() for d in fact.value["days"]} & _WEEKDAYS
        if not days:
            raise ValueError("no valid weekday names")
        tz = self._tz(fact)
        local_start, local_end = start.astimezone(tz), end.astimezone(tz)
        # every local date the half-open event actually touches
        last_touched = (local_end - timedelta(microseconds=1)).date()
        day = local_start.date()
        while day <= last_touched:
            weekday = day.strftime("%A").lower()
            if weekday in days:
                return f"it falls on a {weekday.capitalize()}, which is blocked."
            day += timedelta(days=1)
            if (day - local_start.date()).days > _MAX_WINDOW_DAYS:
                return "it spans more than a year, which certainly hits a blocked day."
        return None

    def _max_meeting_minutes(
        self, fact: MemoryFact, start: UtcDatetime, end: UtcDatetime
    ) -> str | None:
        limit = int(fact.value["minutes"])
        if limit <= 0:
            raise ValueError("minutes must be positive")
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
