from __future__ import annotations

import pytest

from calendai.agent.loop import BAIL_MESSAGE, MAX_ITERATIONS, NO_REPLY, AgentLoop
from calendai.agent.tools import Toolbox
from calendai.core.models import EventDraft, User
from calendai.core.provider import RateLimitError, ServerError
from calendai.traces.emitter import SQLiteTraceEmitter
from tests.conftest import at
from tests.scripted_client import ScriptedClient, text_response, tool_call

ALICE = User(id="u_alice", email="alice@example.com", display_name="Alice")


@pytest.fixture
def harness(provider, store, clock):
    """Builds an AgentLoop wired to the fake provider and a scripted LLM."""

    def make(responses):
        store.upsert_user(ALICE)
        client = ScriptedClient(responses)
        toolbox = Toolbox(
            provider=provider, store=store, user=ALICE, clock=clock, sleep_fn=lambda _s: None
        )
        tracer = SQLiteTraceEmitter(store, clock=clock)
        loop = AgentLoop(
            client=client,
            model="scripted-model",
            toolbox=toolbox,
            store=store,
            tracer=tracer,
            clock=clock,
            user=ALICE,
        )
        return loop, client

    return make


def iso(hour: int) -> str:
    return at(hour).isoformat()


def spans(loop: AgentLoop):
    return loop.tracer.spans_for(loop.last_request_id)


# -- basics ---------------------------------------------------------------


def test_plain_text_turn(harness):
    loop, client = harness([text_response("Hello Alice!")])
    assert loop.run_turn("hi") == "Hello Alice!"
    kinds = [s["kind"] for s in spans(loop)]
    assert kinds == ["llm_call"]
    assert spans(loop)[0]["payload"]["stop_reason"] == "end_turn"


def test_create_event_via_tool(harness, provider):
    loop, client = harness(
        [
            tool_call(
                "create_event",
                {
                    "title": "Standup",
                    "start": iso(5),
                    "end": iso(6),
                    "rationale": "User asked for a standup.",
                },
            ),
            text_response("Booked your standup."),
        ]
    )
    reply = loop.run_turn("book a standup 10:30-11:30")
    assert reply == "Booked your standup."
    events = provider.list_events(ALICE.email, at(0), at(23))
    assert [e.title for e in events] == ["Standup"]
    tool_span = next(s for s in spans(loop) if s["kind"] == "tool_call")
    assert tool_span["payload"]["ok"] is True
    assert tool_span["rationale"] == "User asked for a standup."
    assert "rationale" not in tool_span["payload"]["args"]


def test_silent_turn_after_mutation_becomes_a_confirmation(harness):
    # A cheaper model can end the turn with no text right after a successful
    # tool call. The user must still get a confirmation, never "(no response)".
    loop, client = harness(
        [
            tool_call(
                "save_profile_fact",
                {
                    "fact_type": "rule",
                    "key": "rule:no_meetings_before",
                    "value": {"time": "10:00", "timezone": "Asia/Kolkata"},
                    "statement": "Never schedule before 10:00",
                },
            ),
            text_response(""),  # model says nothing after the save
        ]
    )
    reply = loop.run_turn("never schedule me before 10am")
    assert reply != NO_REPLY
    assert reply == "Done - I've saved that to your profile."


def test_silent_turn_with_no_mutation_stays_no_reply(harness):
    # No side effect + no text is a genuinely empty turn; don't fabricate one.
    loop, client = harness([text_response("")])
    assert loop.run_turn("...") == NO_REPLY


def test_history_persists_across_turns(harness):
    loop, client = harness([text_response("first"), text_response("second")])
    loop.run_turn("message one")
    loop.run_turn("message two")
    sent = client.calls[1]["messages"]
    contents = [m["content"] for m in sent if isinstance(m["content"], str)]
    assert "message one" in contents and "first" in contents


# -- self-correction --------------------------------------------------------


def test_invalid_args_fed_back_for_self_correction(harness, provider):
    loop, client = harness(
        [
            tool_call(
                "create_event",
                {
                    "title": "Bad",
                    "start": "2026-06-15T05:00:00",  # naive!
                    "end": iso(6),
                },
            ),
            tool_call("create_event", {"title": "Fixed", "start": iso(5), "end": iso(6)}),
            text_response("Done after fixing my arguments."),
        ]
    )
    reply = loop.run_turn("schedule something")
    assert "Done" in reply
    # first tool_result was an error and said why
    first_result = client.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] is True
    assert "timezone-aware" in first_result["content"]
    # event was created on the second attempt only
    assert len(provider.list_events(ALICE.email, at(0), at(23))) == 1


def test_unknown_tool_is_survivable(harness):
    loop, client = harness(
        [
            tool_call("teleport_user", {"to": "goa"}),
            text_response("Sorry, I can't do that."),
        ]
    )
    assert loop.run_turn("teleport me") == "Sorry, I can't do that."
    tool_span = next(s for s in spans(loop) if s["kind"] == "tool_call")
    assert tool_span["payload"]["error_type"] == "unknown_tool"


# -- provider failures -------------------------------------------------------


def test_rate_limit_retried_transparently(harness, provider):
    provider.inject_failure("create_event", RateLimitError(retry_after=0.1))
    loop, client = harness(
        [
            tool_call("create_event", {"title": "Retry me", "start": iso(5), "end": iso(6)}),
            text_response("Booked."),
        ]
    )
    assert loop.run_turn("book it") == "Booked."
    tool_span = next(s for s in spans(loop) if s["kind"] == "tool_call")
    assert tool_span["payload"]["ok"] is True
    assert tool_span["payload"]["retries"] == 1  # observable in the trace


def test_retry_exhaustion_surfaces_gracefully(harness, provider):
    provider.inject_failure("create_event", lambda: ServerError("still down"), times=5)
    loop, client = harness(
        [
            tool_call("create_event", {"title": "Doomed", "start": iso(5), "end": iso(6)}),
            text_response("The calendar service is down; I couldn't book it."),
        ]
    )
    reply = loop.run_turn("book it")
    assert "down" in reply
    tool_span = next(s for s in spans(loop) if s["kind"] == "tool_call")
    assert tool_span["payload"]["error_type"] == "provider_unavailable"


# -- loop guard ---------------------------------------------------------------


def test_loop_guard_bails(harness):
    loop, client = harness([])
    client.repeat_forever(tool_call("get_current_datetime", {}))
    reply = loop.run_turn("loop forever")
    assert reply == BAIL_MESSAGE
    llm_calls = [s for s in spans(loop) if s["kind"] == "llm_call"]
    assert len(llm_calls) == MAX_ITERATIONS
    assert any(s["name"] == "loop_guard" for s in spans(loop))


# -- confirmation gate ---------------------------------------------------------


def test_delete_requires_cross_turn_confirmation(harness, provider):
    event = provider.create_event(ALICE.email, EventDraft(title="Victim", start=at(5), end=at(6)))

    loop, client = harness(
        [
            # Turn 1: model tries to delete; gets confirmation_required; tells user.
            tool_call("delete_event", {"event_id": event.id}),
            text_response("This will delete 'Victim' at 10:30. Confirm?"),
        ]
    )

    reply1 = loop.run_turn("delete the victim meeting")
    assert "Confirm" in reply1
    assert provider.list_events(ALICE.email, at(0), at(23))  # still there

    # The tool exchange is not persisted in history, so the model can only
    # learn the token from turn 2's system prompt. The test extracts it from
    # the gate the same way the loop does - nothing hardcoded.
    pending = loop.toolbox.gate.pending()
    assert len(pending) == 1
    token = next(iter(pending))

    client.queue(
        tool_call("delete_event", {"event_id": event.id, "confirmation_token": token}),
        text_response("Deleted."),
    )
    reply2 = loop.run_turn("yes, go ahead")
    assert reply2 == "Deleted."
    assert provider.list_events(ALICE.email, at(0), at(23)) == []

    # turn 2's system prompt carried everything the model needed to recover:
    turn2_system = client.calls[2]["system"]
    assert token in turn2_system
    assert event.id in turn2_system


def test_model_cannot_self_confirm_within_one_turn(harness, provider):
    event = provider.create_event(ALICE.email, EventDraft(title="Safe", start=at(5), end=at(6)))
    loop, client = harness(
        [
            tool_call("delete_event", {"event_id": event.id}),
            # model immediately "confirms" itself in the SAME turn
            tool_call("delete_event", {"event_id": event.id, "confirmation_token": "confirm-001"}),
            text_response("I tried."),
        ]
    )
    loop.run_turn("delete it")
    # the second call was rejected: token from the same turn is invalid
    assert provider.list_events(ALICE.email, at(0), at(23)), "event must survive self-confirm"


def test_decline_revokes_confirmation_even_with_correct_token(harness, provider):
    event = provider.create_event(ALICE.email, EventDraft(title="Keeper", start=at(5), end=at(6)))
    loop, client = harness(
        [
            tool_call("delete_event", {"event_id": event.id}),
            text_response("Delete 'Keeper' at 10:30. Confirm?"),
        ]
    )
    loop.run_turn("delete keeper")
    token = next(iter(loop.toolbox.gate.pending()))

    # Adversarial model: the user says NO, but the model replays the real token.
    client.queue(
        tool_call("delete_event", {"event_id": event.id, "confirmation_token": token}),
        text_response("Understood, I won't delete it."),
    )
    loop.run_turn("no, keep it")
    assert provider.list_events(ALICE.email, at(0), at(23)), "decline must keep the event"
    assert "cancelled" in client.calls[2]["system"]


def test_unrelated_reply_does_not_authorize_pending_delete(harness, provider):
    event = provider.create_event(
        ALICE.email, EventDraft(title="Bystander", start=at(5), end=at(6))
    )
    loop, client = harness(
        [
            tool_call("delete_event", {"event_id": event.id}),
            text_response("Delete 'Bystander'? Confirm?"),
        ]
    )
    loop.run_turn("delete bystander")
    token = next(iter(loop.toolbox.gate.pending()))

    client.queue(
        tool_call("delete_event", {"event_id": event.id, "confirmation_token": token}),
        text_response("Here's your schedule."),
    )
    loop.run_turn("what's on my calendar today?")
    assert provider.list_events(ALICE.email, at(0), at(23)), "unrelated reply must not consent"


# -- crash hygiene ---------------------------------------------------------------


def test_llm_exception_leaves_history_clean(harness, store):
    loop, client = harness([RuntimeError("api down")])
    with pytest.raises(RuntimeError, match="api down"):
        loop.run_turn("hello?")
    roles = [m["role"] for m in store.recent_messages(ALICE.id, limit=10)]
    assert roles == ["user"], "no phantom assistant message after a crash"

    # and the loop recovers cleanly on the next turn
    client.queue(text_response("recovered"))
    assert loop.run_turn("you there?") == "recovered"
    roles = [m["role"] for m in store.recent_messages(ALICE.id, limit=10)]
    assert roles == ["user", "user", "assistant"]


def test_loop_guard_bail_discloses_saved_facts(harness):
    loop, client = harness(
        [
            tool_call(
                "save_profile_fact",
                {
                    "fact_type": "rule",
                    "key": "rule:no_meetings_before",
                    "value": {"time": "10:00"},
                    "statement": "Never schedule before 10:00",
                },
            )
        ]
    )
    client.repeat_forever(tool_call("get_current_datetime", {}))
    reply = loop.run_turn("remember: no meetings before 10, and then loop forever")
    assert reply != BAIL_MESSAGE  # memory writes are mutations too
    assert "rule:no_meetings_before" in reply


def test_loop_guard_bail_discloses_partial_changes(harness, provider):
    loop, client = harness(
        [tool_call("create_event", {"title": "Halfway", "start": iso(5), "end": iso(6)})]
    )
    client.repeat_forever(tool_call("get_current_datetime", {}))
    reply = loop.run_turn("create it and then loop forever")
    assert reply != BAIL_MESSAGE  # plain bail would falsely claim "nothing changed"
    assert "Halfway" in reply
    assert len(provider.list_events(ALICE.email, at(0), at(23))) == 1
