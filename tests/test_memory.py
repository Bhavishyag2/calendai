from __future__ import annotations

import json

import pytest

from calendai.agent.loop import AgentLoop
from calendai.agent.tools import CreateEventArgs, Toolbox
from calendai.core.models import FactType, MemoryFact, User
from calendai.db.store import Store
from calendai.memory.enforcement import RuleEngine
from calendai.memory.episodic import EpisodicExtractor
from calendai.memory.profile import profile_facts
from calendai.traces.emitter import SQLiteTraceEmitter
from tests.conftest import at
from tests.scripted_client import ScriptedClient, text_response

ALICE = User(id="u_alice", email="alice@example.com", display_name="Alice")
# Frozen day is Monday 2026-06-15; 03:30 UTC == 09:00 IST.


def rule(key: str, value: dict, statement: str) -> MemoryFact:
    return MemoryFact(
        user_id=ALICE.id,
        fact_type=FactType.RULE,
        key=key,
        value=value,
        statement=statement,
        provenance="test",
    )


@pytest.fixture
def engine(store):
    store.upsert_user(ALICE)
    return RuleEngine(store, ALICE)


# -- RuleEngine: per-rule checks ---------------------------------------------


def test_no_meetings_before_blocks_and_allows(engine, store):
    store.upsert_fact(rule("rule:no_meetings_before", {"time": "10:00"}, "Never before 10:00 IST"))
    # at(3) UTC == 08:30 IST -> blocked; at(5) == 10:30 IST -> fine
    violation = engine.check("create_event", at(3), at(3, 30))
    assert violation and "Never before 10:00 IST" in violation and "08:30" in violation
    assert engine.check("create_event", at(5), at(6)) is None


def test_no_meetings_after_blocks_late_end(engine, store):
    store.upsert_fact(rule("rule:no_meetings_after", {"time": "18:00"}, "Nothing after 18:00"))
    # at(13) UTC == 18:30 IST end -> blocked
    violation = engine.check("create_event", at(12), at(13))
    assert violation and "18:30" in violation
    assert engine.check("create_event", at(10), at(11)) is None  # ends 16:30 IST


def test_no_meetings_on_blocks_weekend(engine, store):
    store.upsert_fact(rule("rule:no_meetings_on", {"days": ["saturday", "sunday"]}, "No weekends"))
    saturday = at(5, day_offset=5)  # Mon + 5 = Saturday Jun 20
    violation = engine.check("create_event", saturday, at(6, day_offset=5))
    assert violation and "Saturday" in violation
    assert engine.check("create_event", at(5), at(6)) is None  # Monday is fine


def test_max_meeting_minutes(engine, store):
    store.upsert_fact(rule("rule:max_meeting_minutes", {"minutes": 60}, "Max 1 hour"))
    violation = engine.check("create_event", at(5), at(6, 30))  # 90 min
    assert violation and "90" in violation and "60" in violation
    assert engine.check("create_event", at(5), at(6)) is None


def test_rule_timezone_override(engine, store):
    # 10:00 cutoff in UTC: at(5) is 05:00 UTC -> blocked under UTC rule,
    # though it is 10:30 IST
    store.upsert_fact(
        rule(
            "rule:no_meetings_before",
            {"time": "10:00", "timezone": "UTC"},
            "Never before 10:00 UTC",
        )
    )
    assert engine.check("create_event", at(5), at(6)) is not None


def test_unknown_rule_key_is_skipped(engine, store):
    store.upsert_fact(rule("rule:quiet_fridays_vibe", {"vibe": "calm"}, "Fridays are calm"))
    assert engine.check("create_event", at(3), at(4)) is None


def test_malformed_rule_value_never_blocks(engine, store):
    store.upsert_fact(rule("rule:no_meetings_before", {"time": "ten-ish"}, "Vague rule"))
    store.upsert_fact(rule("rule:no_meetings_on", {"days": ["caturday"]}, "No caturdays"))
    assert engine.check("create_event", at(3), at(4)) is None


def test_rule_taught_mid_session_enforced_immediately(engine, store):
    assert engine.check("create_event", at(3), at(4)) is None
    store.upsert_fact(rule("rule:no_meetings_before", {"time": "10:00"}, "Never before 10:00"))
    assert engine.check("create_event", at(3), at(4)) is not None  # fresh read, no rebuild


def test_engine_wired_into_toolbox_blocks_create(engine, store, provider, clock):
    store.upsert_fact(rule("rule:no_meetings_before", {"time": "10:00"}, "Never before 10:00"))
    toolbox = Toolbox(
        provider=provider,
        store=store,
        user=ALICE,
        clock=clock,
        rule_checker=engine.check,
        sleep_fn=lambda _s: None,
    )
    outcome = toolbox.create_event(CreateEventArgs(title="Early", start=at(3), end=at(4)))
    assert not outcome.ok and outcome.error_type == "rule_violation"
    assert "Never before 10:00" in outcome.error


def test_overnight_event_cannot_bypass_before_rule(engine, store):
    store.upsert_fact(rule("rule:no_meetings_before", {"time": "10:00"}, "Never before 10:00"))
    # Sun 23:00 IST -> Mon 09:00 IST: starts late but occupies the morning window
    overnight = engine.check("create_event", at(17, 30, day_offset=-1), at(3, 30))
    assert overnight is not None
    # an afternoon event the same days is fine
    assert engine.check("create_event", at(5, 30), at(6, 30)) is None


def test_weekend_rule_blocks_overnight_friday_event(engine, store):
    store.upsert_fact(rule("rule:no_meetings_on", {"days": ["saturday"]}, "No Saturdays"))
    # Fri 23:00 IST -> Sat 01:00 IST touches Saturday
    crossing = engine.check("create_event", at(17, 30, day_offset=4), at(19, 30, day_offset=4))
    assert crossing is not None and "Saturday" in crossing
    # Fri 22:00 -> Sat 00:00 IST: half-open end at midnight never touches Saturday
    upto_midnight = engine.check("create_event", at(16, 30, day_offset=4), at(18, 30, day_offset=4))
    assert upto_midnight is None


def test_invalid_value_timezone_falls_back_to_user_tz(engine, store):
    # "IST" is not an IANA name; fallback must be the user's tz (Asia/Kolkata),
    # NOT UTC: at(5) is 10:30 IST (allowed) but 05:00 UTC (would be blocked)
    store.upsert_fact(
        rule(
            "rule:no_meetings_before",
            {"time": "10:00", "timezone": "IST"},
            "Never before 10:00",
        )
    )
    assert engine.check("create_event", at(5), at(6)) is None


def test_dst_fallback_hour_cannot_hide_violation(engine, store):
    from datetime import UTC, datetime

    # America/New_York 2026-11-01: clocks fall back at 02:00 EDT (06:00Z).
    # 05:45Z-06:15Z reads 01:45 EDT -> 01:15 EST: the event occupies the
    # repeated before-01:30 wall-clock window and must be blocked.
    store.upsert_fact(
        rule(
            "rule:no_meetings_before",
            {"time": "01:30", "timezone": "America/New_York"},
            "Never before 01:30 ET",
        )
    )
    start = datetime(2026, 11, 1, 5, 45, tzinfo=UTC)
    end = datetime(2026, 11, 1, 6, 15, tzinfo=UTC)
    assert engine.check("create_event", start, end) is not None


def test_before_midnight_rule_blocks_nothing(engine, store):
    store.upsert_fact(
        rule("rule:no_meetings_before", {"time": "00:00"}, "Nothing before midnight (no-op)")
    )
    # even an overnight event crossing midnight is fine: the window is empty
    assert engine.check("create_event", at(17, 30, day_offset=-1), at(3, 30)) is None


def test_non_string_timezone_never_disables_rule(engine, store):
    # legacy/direct write with a junk timezone: enforcement falls back to the
    # user's tz instead of silently skipping the rule
    store.upsert_fact(
        rule(
            "rule:no_meetings_before",
            {"time": "10:00", "timezone": 5},
            "Never before 10:00",
        )
    )
    assert engine.check("create_event", at(3), at(4)) is not None  # 08:30 IST blocked


def test_update_event_rule_enforced_after_confirmation(engine, store, provider, clock):
    from calendai.agent.tools import UpdateEventArgs
    from calendai.core.models import EventDraft

    store.upsert_fact(rule("rule:no_meetings_before", {"time": "10:00"}, "Never before 10:00"))
    toolbox = Toolbox(
        provider=provider,
        store=store,
        user=ALICE,
        clock=clock,
        rule_checker=engine.check,
        sleep_fn=lambda _s: None,
    )
    event = provider.create_event(ALICE.email, EventDraft(title="Movable", start=at(5), end=at(6)))
    args = UpdateEventArgs(event_id=event.id, start=at(3), end=at(3, 30))

    toolbox.new_turn("move it earlier")
    first = toolbox.update_event(args)
    assert first.error_type == "confirmation_required"
    token = next(iter(toolbox.gate.pending()))
    toolbox.new_turn("yes")
    confirmed = toolbox.update_event(
        UpdateEventArgs(event_id=event.id, start=at(3), end=at(3, 30), confirmation_token=token)
    )
    # user consent does not override the stored rule - code wins
    assert not confirmed.ok and confirmed.error_type == "rule_violation"


def test_save_tool_rejects_mistyped_key(store, provider, clock):
    store.upsert_user(ALICE)
    toolbox = Toolbox(
        provider=provider, store=store, user=ALICE, clock=clock, sleep_fn=lambda _s: None
    )
    from calendai.agent.tools import SaveProfileFactArgs

    mistyped = toolbox.save_profile_fact(
        SaveProfileFactArgs(
            fact_type="preference",  # poisoned: rule:* key under preference type
            key="rule:no_meetings_before",
            value={"time": "10:00"},
            statement="Never before 10:00",
        )
    )
    assert not mistyped.ok and mistyped.error_type == "invalid_arguments"
    assert store.list_facts(ALICE.id) == []

    oversized = toolbox.save_profile_fact(
        SaveProfileFactArgs(
            fact_type="rule",
            key="rule:no_meetings_before",
            value={"time": "10:00"},
            statement="x" * 300,
        )
    )
    assert not oversized.ok and oversized.error_type == "invalid_arguments"

    bad_tz = toolbox.save_profile_fact(
        SaveProfileFactArgs(
            fact_type="rule",
            key="rule:no_meetings_before",
            value={"time": "10:00", "timezone": "IST"},  # not an IANA name
            statement="Never before 10:00",
        )
    )
    assert not bad_tz.ok and "timezone" in bad_tz.error

    bool_minutes = toolbox.save_profile_fact(
        SaveProfileFactArgs(
            fact_type="rule",
            key="rule:max_meeting_minutes",
            value={"minutes": True},  # bool is not a positive int
            statement="Cap meetings",
        )
    )
    assert not bool_minutes.ok and bool_minutes.error_type == "invalid_arguments"
    assert store.list_facts(ALICE.id) == []


# -- profile ordering -------------------------------------------------------------


def test_profile_facts_orders_rules_first(store):
    store.upsert_user(ALICE)
    for fact_type, key in [
        (FactType.CONTACT, "contact:zara"),
        (FactType.PREFERENCE, "pref:default_duration"),
        (FactType.RULE, "rule:no_meetings_before"),
    ]:
        store.upsert_fact(
            MemoryFact(
                user_id=ALICE.id,
                fact_type=fact_type,
                key=key,
                value={"x": 1},
                statement=key,
                provenance="test",
            )
        )
    ordered = [f.fact_type for f in profile_facts(store, ALICE.id)]
    assert ordered == [FactType.RULE, FactType.PREFERENCE, FactType.CONTACT]


# -- episodic extraction (recorded fixtures, no live API) ---------------------------


def utility(reply: str) -> ScriptedClient:
    return ScriptedClient([text_response(reply)])


@pytest.fixture
def extractor_for(store, clock):
    store.upsert_user(ALICE)

    def make(reply: str) -> EpisodicExtractor:
        return EpisodicExtractor(utility(reply), "scripted-utility", store, clock)

    return make


RULE_REPLY = json.dumps(
    [
        {
            "fact_type": "rule",
            "key": "rule:no_meetings_before",
            "value": {"time": "10:00", "timezone": "Asia/Kolkata"},
            "statement": "Never schedule meetings before 10:00 IST.",
        }
    ]
)


def test_extracts_rule_fact(extractor_for, store):
    saved = extractor_for(RULE_REPLY).extract(ALICE, "never book me before 10am", "Noted!")
    assert [f.key for f in saved] == ["rule:no_meetings_before"]
    assert store.list_facts(ALICE.id, FactType.RULE)[0].value["time"] == "10:00"
    assert "auto-extracted" in saved[0].provenance


def test_extracts_contact_and_tolerates_code_fences(extractor_for, store):
    contact = json.dumps(
        [
            {
                "fact_type": "contact",
                "key": "contact:alex",
                "value": {"email": "alex@corp.com"},
                "statement": "Alex is alex@corp.com.",
            }
        ]
    )
    fenced = f"```json\n{contact}\n```"
    saved = extractor_for(fenced).extract(ALICE, "Alex is alex@corp.com", "Saved.")
    assert [f.key for f in saved] == ["contact:alex"]


def test_empty_extraction(extractor_for, store):
    saved = extractor_for("[]").extract(ALICE, "what's on my calendar?", "Three meetings.")
    assert saved == [] and store.list_facts(ALICE.id) == []


def test_malformed_json_never_breaks_turn(extractor_for, store):
    saved = extractor_for("I think the user wants...").extract(ALICE, "hi", "hello")
    assert saved == [] and store.list_facts(ALICE.id) == []


def test_api_error_never_breaks_turn(store, clock):
    store.upsert_user(ALICE)
    broken = ScriptedClient([RuntimeError("utility model down")])
    extractor = EpisodicExtractor(broken, "scripted-utility", store, clock)
    assert extractor.extract(ALICE, "never before 10", "ok") == []


def test_mismatched_key_prefix_is_rejected(extractor_for, store):
    sneaky = json.dumps(
        [
            {
                "fact_type": "preference",
                "key": "rule:no_meetings_before",  # type says pref, key says rule
                "value": {"time": "10:00"},
                "statement": "sneaky",
            }
        ]
    )
    assert extractor_for(sneaky).extract(ALICE, "x", "y") == []


def test_duplicate_fact_does_not_churn_supersession(extractor_for, store):
    extractor_for(RULE_REPLY).extract(ALICE, "never before 10am", "Noted!")
    again = extractor_for(RULE_REPLY).extract(ALICE, "remember: nothing before 10!", "Yes.")
    assert again == []  # identical key+value: no new version
    versions = store.list_facts(ALICE.id, FactType.RULE, active_only=False)
    assert len(versions) == 1


def test_changed_value_supersedes(extractor_for, store):
    extractor_for(RULE_REPLY).extract(ALICE, "never before 10am", "Noted!")
    eleven = RULE_REPLY.replace("10:00", "11:00")
    saved = extractor_for(eleven).extract(ALICE, "make that 11am", "Updated.")
    assert len(saved) == 1
    active = store.list_facts(ALICE.id, FactType.RULE)
    assert len(active) == 1 and active[0].value["time"] == "11:00"


def test_duplicate_candidates_in_one_reply_saved_once(extractor_for, store):
    item = json.loads(RULE_REPLY)[0]
    doubled = json.dumps([item, item])
    saved = extractor_for(doubled).extract(ALICE, "never before 10", "ok")
    assert len(saved) == 1
    assert len(store.list_facts(ALICE.id, FactType.RULE, active_only=False)) == 1


def test_same_turn_tool_save_then_extractor_dedups(store, provider, clock):
    store.upsert_user(ALICE)
    toolbox = Toolbox(
        provider=provider, store=store, user=ALICE, clock=clock, sleep_fn=lambda _s: None
    )
    from calendai.agent.tools import SaveProfileFactArgs

    # the agent saved it via the tool mid-turn...
    toolbox.save_profile_fact(
        SaveProfileFactArgs(
            fact_type="rule",
            key="rule:no_meetings_before",
            value={"time": "10:00", "timezone": "Asia/Kolkata"},
            statement="Never before 10:00 IST.",
        )
    )
    # ...then post-turn extraction emits the same fact: no second version
    saved = EpisodicExtractor(utility(RULE_REPLY), "scripted-utility", store, clock).extract(
        ALICE, "never book me before 10am", "Saved!"
    )
    assert saved == []
    assert len(store.list_facts(ALICE.id, FactType.RULE, active_only=False)) == 1


def test_extractor_repairs_mistyped_legacy_fact(extractor_for, store):
    # a poisoned legacy row (inserted directly, bypassing write validation)
    store.upsert_fact(
        MemoryFact(
            user_id=ALICE.id,
            fact_type=FactType.PREFERENCE,
            key="rule:no_meetings_before",
            value={"time": "10:00", "timezone": "Asia/Kolkata"},
            statement="mistyped",
            provenance="legacy",
        )
    )
    saved = extractor_for(RULE_REPLY).extract(ALICE, "never before 10am", "ok")
    assert len(saved) == 1  # same key+value but WRONG type: supersedes, not dedup
    active = store.list_facts(ALICE.id, FactType.RULE)
    assert len(active) == 1 and active[0].key == "rule:no_meetings_before"


def test_store_failure_never_breaks_turn(store, clock):
    store.upsert_user(ALICE)
    extractor = EpisodicExtractor(utility(RULE_REPLY), "scripted-utility", store, clock)
    store.close()  # simulate a dead database under the extractor
    saved = extractor.extract(ALICE, "never before 10", "ok")
    assert saved == []
    assert extractor.last_error is not None


# -- loop integration ------------------------------------------------------------


def test_loop_runs_extraction_post_turn_with_trace(provider, store, clock):
    store.upsert_user(ALICE)
    agent_client = ScriptedClient([text_response("Got it - nothing before 10am.")])
    toolbox = Toolbox(
        provider=provider, store=store, user=ALICE, clock=clock, sleep_fn=lambda _s: None
    )
    tracer = SQLiteTraceEmitter(store, clock=clock)
    extractor = EpisodicExtractor(utility(RULE_REPLY), "scripted-utility", store, clock)
    loop = AgentLoop(
        client=agent_client,
        model="scripted-model",
        toolbox=toolbox,
        store=store,
        tracer=tracer,
        clock=clock,
        user=ALICE,
        extractor=extractor,
    )
    loop.run_turn("never book me before 10am")
    assert store.list_facts(ALICE.id, FactType.RULE)[0].key == "rule:no_meetings_before"
    spans = tracer.spans_for(loop.last_request_id)
    mem = next(s for s in spans if s["kind"] == "memory_op")
    assert mem["payload"]["extracted_keys"] == ["rule:no_meetings_before"]
    assert mem["payload"]["error"] is None


def test_extractor_error_lands_in_trace_span(provider, store, clock):
    store.upsert_user(ALICE)
    agent_client = ScriptedClient([text_response("Hi!")])
    toolbox = Toolbox(
        provider=provider, store=store, user=ALICE, clock=clock, sleep_fn=lambda _s: None
    )
    tracer = SQLiteTraceEmitter(store, clock=clock)
    broken_utility = ScriptedClient([RuntimeError("utility model down")])
    loop = AgentLoop(
        client=agent_client,
        model="scripted-model",
        toolbox=toolbox,
        store=store,
        tracer=tracer,
        clock=clock,
        user=ALICE,
        extractor=EpisodicExtractor(broken_utility, "scripted-utility", store, clock),
    )
    assert loop.run_turn("hello") == "Hi!"  # the turn survives
    mem = next(s for s in tracer.spans_for(loop.last_request_id) if s["kind"] == "memory_op")
    assert "RuntimeError" in mem["payload"]["error"]


# -- the headline: memory survives a restart ------------------------------------------


def test_rule_taught_in_session_one_enforced_in_session_two(tmp_path, provider, clock):
    db = tmp_path / "persist.db"

    # Session 1: the rule is learned (via extraction) and the process "exits".
    store1 = Store(db, clock=clock)
    store1.upsert_user(ALICE)
    EpisodicExtractor(utility(RULE_REPLY), "scripted-utility", store1, clock).extract(
        ALICE, "never book me before 10am", "Noted!"
    )
    store1.close()

    # Session 2: fresh store, fresh engine, fresh toolbox - same database file.
    store2 = Store(db, clock=clock)
    engine = RuleEngine(store2, ALICE)
    toolbox = Toolbox(
        provider=provider,
        store=store2,
        user=ALICE,
        clock=clock,
        rule_checker=engine.check,
        sleep_fn=lambda _s: None,
    )
    outcome = toolbox.create_event(CreateEventArgs(title="Early", start=at(3), end=at(4)))
    assert not outcome.ok and outcome.error_type == "rule_violation"
    allowed = toolbox.create_event(CreateEventArgs(title="Later", start=at(5), end=at(6)))
    assert allowed.ok
    store2.close()
