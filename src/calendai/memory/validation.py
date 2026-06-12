"""Shared write-time validation for profile facts.

Both write paths - the agent's save_profile_fact tool and the episodic
extractor - run candidates through validate_fact BEFORE anything reaches the
store. This closes the type/key-poisoning hole (a "preference" stored under a
rule:* key would be invisible to the RuleEngine yet block the extractor's
repair via dedup) and caps free-text sizes so memory cannot become a
prompt-injection amplifier.
"""

from __future__ import annotations

import json
import re
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

KEY_RE = re.compile(r"^(rule|contact|pref):[a-z0-9_]+$")
PREFIX_TO_TYPE = {"rule": "rule", "contact": "contact", "pref": "preference"}

MAX_STATEMENT_CHARS = 240
MAX_VALUE_CHARS = 500  # JSON-serialized

_WEEKDAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def parse_hhmm(raw: str) -> dt_time:
    hour, minute = str(raw).strip().split(":")
    return dt_time(int(hour), int(minute))


def _valid_time_rule(value: dict[str, Any]) -> str | None:
    try:
        parse_hhmm(value["time"])
    except (KeyError, ValueError, TypeError):
        return 'value must be {"time": "HH:MM", "timezone": "<IANA, optional>"}'
    tz = value.get("timezone")
    if tz is not None:
        # a bad timezone stored now would silently disable the rule at
        # enforcement time - reject it while the model can still self-correct
        if not isinstance(tz, str):
            return "timezone must be an IANA string like Asia/Kolkata, or omitted"
        try:
            ZoneInfo(tz)
        except (KeyError, ValueError):
            return f"unknown timezone {tz!r}; use IANA names like Asia/Kolkata"
    return None


def _valid_days_rule(value: dict[str, Any]) -> str | None:
    days = value.get("days")
    if not isinstance(days, list) or not days:
        return 'value must be {"days": ["saturday", ...]}'
    cleaned = {str(d).strip().lower() for d in days}
    if not cleaned <= _WEEKDAYS:
        return f"unknown weekday names: {sorted(cleaned - _WEEKDAYS)}"
    return None


def _valid_minutes_rule(value: dict[str, Any]) -> str | None:
    minutes = value.get("minutes")
    # bool is an int subclass; "true" must not become a 1-minute cap
    if not isinstance(minutes, int) or isinstance(minutes, bool) or minutes <= 0:
        return 'value must be {"minutes": <positive int>}'
    return None


# Canonical rule keys get their value shape checked at WRITE time, so the
# RuleEngine never meets a malformed canonical rule it must silently skip.
CANONICAL_RULE_VALIDATORS = {
    "rule:no_meetings_before": _valid_time_rule,
    "rule:no_meetings_after": _valid_time_rule,
    "rule:no_meetings_on": _valid_days_rule,
    "rule:max_meeting_minutes": _valid_minutes_rule,
}


def validate_fact(fact_type: str, key: str, value: dict[str, Any], statement: str) -> str | None:
    """Returns a human-readable problem, or None when the fact is storable."""
    match = KEY_RE.match(key)
    if not match:
        return (
            f"key {key!r} must match <rule|contact|pref>:<lower_snake_case> "
            "(e.g. rule:no_meetings_before, contact:alex)"
        )
    if PREFIX_TO_TYPE[match.group(1)] != fact_type:
        return (
            f"key prefix {match.group(1)!r} does not match fact_type {fact_type!r}; "
            "rule:* keys need fact_type='rule', contact:* 'contact', pref:* 'preference'"
        )
    if len(statement) > MAX_STATEMENT_CHARS:
        return f"statement too long ({len(statement)} > {MAX_STATEMENT_CHARS} chars)"
    if len(json.dumps(value)) > MAX_VALUE_CHARS:
        return f"value too large (> {MAX_VALUE_CHARS} chars serialized)"
    shape_check = CANONICAL_RULE_VALIDATORS.get(key)
    if shape_check is not None:
        problem = shape_check(value)
        if problem:
            return f"canonical rule {key}: {problem}"
    return None
