from __future__ import annotations

import json

from calendai.core.models import Attendee, Event, EventStatus, FactType, MemoryFact
from calendai.evals import scorers
from calendai.evals.report import render_report
from calendai.evals.results import CheckResult, RunResult, ScenarioResult
from calendai.evals.runner import run_scenario
from calendai.evals.schema import (
    Expectations,
    ExpectedEvent,
    ExpectedFact,
    JudgeRubric,
    Scenario,
    Session,
    Trajectory,
    UserSpec,
    load_scenarios,
)
from tests.conftest import at
from tests.scripted_client import text_response, tool_call

ALICE = "alice@example.com"


def _event(title: str, start, end, attendees=()) -> Event:
    return Event(
        id="e1",
        calendar_id=ALICE,
        organizer=ALICE,
        title=title,
        start=start,
        end=end,
        status=EventStatus.CONFIRMED,
        attendees=[Attendee(email=a) for a in attendees],
    )


# -- scorers: events ----------------------------------------------------------


def test_score_events_present_and_absent():
    events = {ALICE: [_event("Standup", at(5), at(6))]}
    present = scorers.score_events(
        [ExpectedEvent(user=ALICE, title_contains="stand", start=at(5))], events
    )
    assert present[0].passed
    absent = scorers.score_events(
        [ExpectedEvent(user=ALICE, title_contains="lunch", present=False)], events
    )
    assert absent[0].passed  # no lunch event -> absent assertion holds


def test_score_events_attendee_and_time_mismatch():
    events = {ALICE: [_event("Sync", at(5), at(6), attendees=["bob@x.com"])]}
    ok = scorers.score_events(
        [ExpectedEvent(user=ALICE, title_contains="sync", attendee="bob@x.com")], events
    )
    assert ok[0].passed
    bad_time = scorers.score_events(
        [ExpectedEvent(user=ALICE, title_contains="sync", start=at(9))], events
    )
    assert not bad_time[0].passed
    missing_attendee = scorers.score_events(
        [ExpectedEvent(user=ALICE, title_contains="sync", attendee="zoe@x.com")], events
    )
    assert not missing_attendee[0].passed


# -- scorers: facts -----------------------------------------------------------


def _fact(key: str, value: dict, ftype=FactType.RULE) -> MemoryFact:
    return MemoryFact(
        user_id="u", fact_type=ftype, key=key, value=value, statement=key, provenance="t"
    )


def test_score_facts_present_value_contains():
    facts = {
        ALICE: [_fact("rule:no_meetings_before", {"time": "10:00", "timezone": "Asia/Kolkata"})]
    }
    ok = scorers.score_facts(
        [ExpectedFact(user=ALICE, key="rule:no_meetings_before", value_contains={"time": "10:00"})],
        facts,
    )
    assert ok[0].passed
    bad = scorers.score_facts(
        [ExpectedFact(user=ALICE, key="rule:no_meetings_before", value_contains={"time": "09:00"})],
        facts,
    )
    assert not bad[0].passed
    absent = scorers.score_facts([ExpectedFact(user=ALICE, key="pref:foo", present=False)], facts)
    assert absent[0].passed


# -- scorers: trajectory + reply ---------------------------------------------


def test_score_trajectory_must_and_must_not():
    expect = Expectations(
        trajectory=Trajectory(must_call=["create_event"], must_not_call=["delete_event"])
    )
    good = scorers.score_trajectory(expect, ["get_current_datetime", "create_event"])
    assert all(c.passed for c in good)
    bad = scorers.score_trajectory(expect, ["delete_event"])
    assert not any(c.passed for c in bad)  # create missing AND delete present


def test_score_reply_substrings():
    expect = Expectations(final_reply_contains=["booked", "10:00"])
    res = scorers.score_reply_substrings(expect, ["Your meeting is BOOKED for 10:00 AM."])
    assert all(c.passed for c in res)
    miss = scorers.score_reply_substrings(
        Expectations(final_reply_contains=["cancelled"]), ["all set"]
    )
    assert not miss[0].passed


def test_score_reply_substrings_checks_final_reply_only():
    # a needle present only in an EARLIER turn must NOT satisfy the check
    expect = Expectations(final_reply_contains=["booked"])
    res = scorers.score_reply_substrings(expect, ["Booked your standup.", "Anything else?"])
    assert not res[0].passed  # "booked" was in turn 1, not the final reply


# -- scorers: judge (scripted utility client) --------------------------------


class ScriptedJudge:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.messages = self
        self.seen = []

    def create(self, *, system, messages, **kwargs):
        self.seen.append(messages[0]["content"])
        return text_response(self._verdicts.pop(0))


def test_score_judge_pass_and_fail():
    client = ScriptedJudge(["PASS", "FAIL"])
    rubrics = [JudgeRubric(criterion="confirms the time"), JudgeRubric(criterion="is polite")]
    res = scorers.score_judge(rubrics, ["Booked for 10am."], client, "scripted-judge")
    assert res[0].passed and not res[1].passed


def test_score_judge_no_reply_at_index():
    client = ScriptedJudge(["PASS"])
    res = scorers.score_judge(
        [JudgeRubric(criterion="x", target_turn=5)], ["only one"], client, "m"
    )
    assert not res[0].passed and "no reply" in res[0].detail


class ExplodingJudge:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        raise RuntimeError("judge API down")


def test_score_judge_api_error_is_a_failed_check_not_a_crash():
    res = scorers.score_judge(
        [JudgeRubric(criterion="confirms time")], ["Booked."], ExplodingJudge(), "m"
    )
    assert len(res) == 1 and not res[0].passed
    assert "judge call failed" in res[0].detail  # recorded, not propagated


# -- report rendering ---------------------------------------------------------


def _scenario_result(sid, passed, tags=("crud",)) -> ScenarioResult:
    check = CheckResult(
        layer="end_state", name="event present", passed=passed, detail="" if passed else "missing"
    )
    return ScenarioResult(
        scenario_id=sid,
        description=f"{sid} desc",
        tags=list(tags),
        runs=[RunResult(run_index=0, checks=[check])],
    )


def test_render_report_summarizes_and_lists_failures():
    results = [_scenario_result("a_pass", True), _scenario_result("b_fail", False)]
    md = render_report(
        results, agent_model="sonnet", utility_model="haiku", generated_at="2026-06-13"
    )
    assert "Scenarios passed:** 1/2 (50%)" in md
    assert "`a_pass`" in md and "`b_fail`" in md
    assert "## Failure analysis" in md
    assert "event present" in md  # the failing check surfaces
    assert "No failures" not in md


def test_render_report_all_pass():
    md = render_report(
        [_scenario_result("a", True)], agent_model="s", utility_model="h", generated_at="t"
    )
    assert "No failures. Every scenario passed on every run." in md


# -- end-to-end runner: full stack with restart persistence ------------------


class StatefulAgentClient:
    """Books at 9am when asked; otherwise just acknowledges. After any
    tool_result it wraps up with text."""

    def __init__(self):
        self.messages = self

    def create(self, *, messages, **kwargs):
        last = messages[-1]
        content = last["content"]
        if isinstance(content, list) and content and content[0].get("type") == "tool_result":
            return text_response("All done.")
        text = (content if isinstance(content, str) else "").lower()
        if "9am" in text:
            return tool_call(
                "create_event",
                {
                    "title": "Early sync",
                    "start": "2026-06-15T09:00:00+05:30",
                    "end": "2026-06-15T09:30:00+05:30",
                    "rationale": "user asked to book at 9am",
                },
            )
        return text_response("Noted, I'll remember that.")


class StatefulUtilityClient:
    """Extraction: emits the no-meetings-before rule when taught; else []. Judge: PASS."""

    RULE = json.dumps(
        [
            {
                "fact_type": "rule",
                "key": "rule:no_meetings_before",
                "value": {"time": "10:00", "timezone": "Asia/Kolkata"},
                "statement": "Never schedule meetings before 10:00 IST.",
            }
        ]
    )

    def __init__(self):
        self.messages = self

    def create(self, *, system, messages, **kwargs):
        if "extract durable profile facts" in system.lower():
            user = messages[0]["content"].lower()
            if "never" in user and "10" in user:
                return text_response(self.RULE)
            return text_response("[]")
        return text_response("PASS")  # judge


def _memory_scenario() -> Scenario:
    return Scenario(
        id="memory_persists_across_restart",
        description="A rule taught in session 1 is enforced in code in session 2 after a restart",
        tags=["memory", "rule_adherence"],
        runs=1,
        users=[UserSpec(email=ALICE, display_name="Alice")],
        sessions=[
            Session(user=ALICE, turns=["Please remember: never book me before 10am."]),
            Session(user=ALICE, turns=["Book me a sync at 9am."]),
        ],
        expect=Expectations(
            events=[ExpectedEvent(user=ALICE, title_contains="sync", present=False)],
            facts=[
                ExpectedFact(
                    user=ALICE, key="rule:no_meetings_before", value_contains={"time": "10:00"}
                )
            ],
            trajectory=Trajectory(must_call=["create_event"]),
            final_reply_contains=["done"],
        ),
    )


def test_end_to_end_memory_persistence_and_rule_enforcement():
    result = run_scenario(
        _memory_scenario(),
        agent_client=StatefulAgentClient(),
        agent_model="scripted-agent",
        utility_client=StatefulUtilityClient(),
        utility_model="scripted-utility",
        run_judge=False,
    )
    assert result.passed, result.failing_checks()
    # the fact was extracted in session 1 and survived the restart...
    fact_checks = [c for c in result.runs[0].checks if c.layer == "end_state" and "fact" in c.name]
    assert fact_checks and all(c.passed for c in fact_checks)
    # ...and the 9am booking was attempted (trajectory) but blocked (no event).
    assert "create_event" in [
        c.name.split()[-1] for c in result.runs[0].checks if "must call" in c.name
    ]


def test_runner_records_judge_checks_when_enabled():
    result = run_scenario(
        _memory_scenario().model_copy(
            update={"expect": Expectations(judge=[JudgeRubric(criterion="acknowledges the rule")])}
        ),
        agent_client=StatefulAgentClient(),
        agent_model="scripted-agent",
        utility_client=StatefulUtilityClient(),
        utility_model="scripted-utility",
        run_judge=True,
    )
    judge_checks = [c for r in result.runs for c in r.checks if c.layer == "judge"]
    assert judge_checks and all(c.passed for c in judge_checks)


# -- scenario loader ----------------------------------------------------------


def test_packaged_scenarios_load_and_are_unique():
    from pathlib import Path

    scenario_dir = Path(__file__).resolve().parents[1] / "evals" / "scenarios"
    scenarios = load_scenarios(scenario_dir)
    assert len(scenarios) >= 18  # a substantial scenario suite is expected
    for s in scenarios:
        assert s.users and s.sessions  # every scenario is runnable


# -- gate-5 soundness fixes ---------------------------------------------------


def test_zero_check_run_does_not_pass_vacuously():
    # a run that scored nothing (e.g. a judge-only scenario under --no-judge)
    # must NOT report a green
    assert RunResult(run_index=0, checks=[]).passed is False
    sr = ScenarioResult(scenario_id="x", description="d", runs=[RunResult(run_index=0, checks=[])])
    assert sr.passed is False


def test_judge_only_scenario_fails_when_judge_disabled():
    judge_only = _memory_scenario().model_copy(
        update={
            "expect": Expectations(judge=[JudgeRubric(criterion="acknowledges the rule")]),
        }
    )
    result = run_scenario(
        judge_only,
        agent_client=StatefulAgentClient(),
        agent_model="scripted-agent",
        utility_client=StatefulUtilityClient(),
        utility_model="scripted-utility",
        run_judge=False,
    )
    assert not result.passed  # nothing was scored -> not a vacuous pass


def test_scenario_rejects_undeclared_user_reference():
    import pytest

    with pytest.raises(ValueError, match="undeclared users"):
        Scenario(
            id="typo",
            description="references a user that is not declared",
            users=[UserSpec(email="alice@example.com")],
            sessions=[Session(user="alcie@example.com", turns=["hi"])],  # typo
        )


def test_judge_parsing_is_first_token_only():
    # "PASSABLE" / "PASS but..." must not count as PASS
    tricky = ScriptedJudge(["PASSABLE thing", "PASS — but it actually fails"])
    res = scorers.score_judge(
        [JudgeRubric(criterion="a"), JudgeRubric(criterion="b")],
        ["reply"],
        tricky,
        "m",
    )
    assert not res[0].passed  # "PASSABLE" rejected
    assert res[1].passed  # "PASS —" is a clean first-token PASS
