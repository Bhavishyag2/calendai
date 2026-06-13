"""Scoring layers: end-state (objective), trajectory (traces), LLM judge.

Each scorer is a pure function over already-collected world state - it never
touches the network except score_judge, which is handed a client. This keeps
the objective layers fully unit-testable offline.
"""

from __future__ import annotations

import json
from typing import Any

from calendai.core.models import Event, MemoryFact
from calendai.evals.results import CheckResult
from calendai.evals.schema import Expectations, ExpectedEvent, ExpectedFact, JudgeRubric

JUDGE_MAX_TOKENS = 16

JUDGE_SYSTEM = """\
You are a strict evaluator. Given an assistant reply and a single yes/no
criterion, answer with exactly one word: PASS if the reply satisfies the
criterion, or FAIL if it does not. No punctuation, no explanation.
"""


# -- end state: events -------------------------------------------------------


def _event_matches(event: Event, exp: ExpectedEvent) -> bool:
    attendee_emails = {a.email for a in event.attendees}
    return all(
        [
            not exp.title_contains or exp.title_contains.lower() in event.title.lower(),
            not exp.start or event.start == exp.start,
            not exp.end or event.end == exp.end,
            not exp.attendee or exp.attendee in attendee_emails,
        ]
    )


def score_events(
    expected: list[ExpectedEvent], events_by_user: dict[str, list[Event]]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for exp in expected:
        candidates = events_by_user.get(exp.user, [])
        match = next((e for e in candidates if _event_matches(e, exp)), None)
        criteria = exp.model_dump(exclude_none=True, exclude={"user", "present"})
        if exp.present:
            results.append(
                CheckResult(
                    layer="end_state",
                    name=f"event present for {exp.user}: {criteria}",
                    passed=match is not None,
                    detail=""
                    if match
                    else f"no event matching {criteria} among {[e.title for e in candidates]}",
                )
            )
        else:
            results.append(
                CheckResult(
                    layer="end_state",
                    name=f"event absent for {exp.user}: {criteria}",
                    passed=match is None,
                    detail="" if match is None else f"unexpected event present: {match.title!r}",
                )
            )
    return results


# -- end state: facts --------------------------------------------------------


def _value_contains(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(stored.get(k) == v for k, v in expected.items())


def score_facts(
    expected: list[ExpectedFact], facts_by_user: dict[str, list[MemoryFact]]
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for exp in expected:
        facts = {f.key: f for f in facts_by_user.get(exp.user, [])}
        fact = facts.get(exp.key)
        if exp.present:
            ok = fact is not None and (
                exp.value_contains is None or _value_contains(fact.value, exp.value_contains)
            )
            detail = ""
            if fact is None:
                detail = f"fact {exp.key!r} not stored (have {sorted(facts)})"
            elif not ok:
                detail = f"fact {exp.key!r} value {fact.value} lacks {exp.value_contains}"
            results.append(
                CheckResult(
                    layer="end_state",
                    name=f"fact present for {exp.user}: {exp.key}",
                    passed=ok,
                    detail=detail,
                )
            )
        else:
            results.append(
                CheckResult(
                    layer="end_state",
                    name=f"fact absent for {exp.user}: {exp.key}",
                    passed=fact is None,
                    detail="" if fact is None else f"unexpected fact {exp.key!r} present",
                )
            )
    return results


# -- trajectory --------------------------------------------------------------


def score_trajectory(expected: Expectations, tools_called: list[str]) -> list[CheckResult]:
    called = set(tools_called)
    results: list[CheckResult] = []
    for tool in expected.trajectory.must_call:
        results.append(
            CheckResult(
                layer="trajectory",
                name=f"must call {tool}",
                passed=tool in called,
                detail="" if tool in called else f"tools actually called: {sorted(called)}",
            )
        )
    for tool in expected.trajectory.must_not_call:
        results.append(
            CheckResult(
                layer="trajectory",
                name=f"must NOT call {tool}",
                passed=tool not in called,
                detail="" if tool not in called else f"{tool} was called",
            )
        )
    return results


# -- final reply substring checks (cheap, deterministic) ---------------------


def score_reply_substrings(expected: Expectations, final_replies: list[str]) -> list[CheckResult]:
    blob = "\n".join(final_replies).lower()
    results: list[CheckResult] = []
    for needle in expected.final_reply_contains:
        present = needle.lower() in blob
        results.append(
            CheckResult(
                layer="reply",
                name=f"reply contains {needle!r}",
                passed=present,
                detail="" if present else "not found in any assistant reply",
            )
        )
    return results


# -- LLM judge ---------------------------------------------------------------


def _select_reply(replies: list[str], target_turn: int) -> str | None:
    if not replies:
        return None
    try:
        return replies[target_turn]
    except IndexError:
        return None


def score_judge(
    rubrics: list[JudgeRubric], final_replies: list[str], client: Any, model: str
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for rubric in rubrics:
        reply = _select_reply(final_replies, rubric.target_turn)
        if reply is None:
            results.append(
                CheckResult(
                    layer="judge",
                    name=rubric.criterion,
                    passed=False,
                    detail=f"no reply at turn index {rubric.target_turn}",
                )
            )
            continue
        verdict, raw = _judge_once(client, model, rubric.criterion, reply)
        results.append(
            CheckResult(
                layer="judge",
                name=rubric.criterion,
                passed=verdict,
                detail="" if verdict else f"judge said {raw!r} for reply: {reply[:120]!r}",
            )
        )
    return results


def _judge_once(client: Any, model: str, criterion: str, reply: str) -> tuple[bool, str]:
    user = json.dumps({"criterion": criterion, "assistant_reply": reply})
    response = client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return text.strip().upper().startswith("PASS"), text.strip()


__all__ = [
    "score_events",
    "score_facts",
    "score_trajectory",
    "score_reply_substrings",
    "score_judge",
]
