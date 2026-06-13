"""Scenario runner: builds the deterministic world and drives the real agent.

One FakeCalendarProvider per run persists across sessions (the calendar is
external and survives restarts). The SQLite store is reopened each session;
conversation history is cleared at every session boundary so that any
cross-session behaviour MUST come from persisted profile memory, not from
lingering chat - that is what makes the memory differentiator a real test.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

from calendai.agent.loop import AgentLoop
from calendai.agent.tools import Toolbox
from calendai.core.clock import FrozenClock
from calendai.core.models import Attendee, EventDraft, FactType, MemoryFact, User
from calendai.core.provider import (
    AuthError,
    MalformedResponseError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from calendai.db.store import Store
from calendai.evals import scorers
from calendai.evals.results import RunResult, ScenarioResult
from calendai.evals.schema import Scenario, SeedEvent, SeedFact
from calendai.memory.enforcement import RuleEngine
from calendai.memory.episodic import EpisodicExtractor
from calendai.providers.fake import FakeCalendarProvider, sequential_ids
from calendai.traces.emitter import SQLiteTraceEmitter

_READ_WINDOW_DAYS = 366  # how far around frozen_now to scan for end-state events


def _failure_factory(kind: str):  # noqa: ANN202 - returns a zero-arg exception factory
    return {
        "rate_limit": lambda: RateLimitError(retry_after=0.0),
        "server_error": lambda: ServerError("injected server error"),
        "not_found": lambda: NotFoundError("injected not found"),
        "malformed": lambda: MalformedResponseError("injected malformed body"),
        "auth": lambda: AuthError("injected auth failure"),
    }[kind]


def _user_id(email: str) -> str:
    return "u_" + re.sub(r"[^a-z0-9]+", "_", email.lower())


def _user(spec: Any) -> User:
    return User(
        id=_user_id(spec.email),
        email=spec.email,
        display_name=spec.display_name or spec.email.split("@")[0],
        timezone=spec.timezone,
    )


def _seed_event_draft(ev: SeedEvent) -> EventDraft:
    return EventDraft(
        title=ev.title,
        start=ev.start,
        end=ev.end,
        description=ev.description,
        attendees=[Attendee(email=e) for e in ev.attendees],
    )


def _seed_fact(store: Store, fact: SeedFact) -> None:
    store.upsert_fact(
        MemoryFact(
            user_id=_user_id(fact.user),
            fact_type=FactType(fact.fact_type),
            key=fact.key,
            value=fact.value,
            statement=fact.statement,
            provenance="eval seed",
        )
    )


def run_scenario(
    scenario: Scenario,
    *,
    agent_client: Any,
    agent_model: str,
    utility_client: Any,
    utility_model: str,
    run_judge: bool = True,
) -> ScenarioResult:
    runs = [
        _run_once(
            scenario,
            i,
            agent_client=agent_client,
            agent_model=agent_model,
            utility_client=utility_client,
            utility_model=utility_model,
            run_judge=run_judge,
        )
        for i in range(scenario.runs)
    ]
    return ScenarioResult(
        scenario_id=scenario.id,
        description=scenario.description,
        tags=scenario.tags,
        runs=runs,
    )


def _run_once(
    scenario: Scenario,
    run_index: int,
    *,
    agent_client: Any,
    agent_model: str,
    utility_client: Any,
    utility_model: str,
    run_judge: bool,
) -> RunResult:
    workdir = Path(tempfile.mkdtemp(prefix=f"calendai-eval-{scenario.id}-"))
    db_path = workdir / "eval.db"
    clock = FrozenClock(scenario.frozen_now)
    provider = FakeCalendarProvider(clock, sequential_ids())
    final_replies: list[str] = []
    try:
        _setup_world(scenario, db_path, clock, provider)
        for session_index, session in enumerate(scenario.sessions):
            final_replies.extend(
                _run_session(
                    scenario,
                    session,
                    session_index,
                    db_path=db_path,
                    clock=clock,
                    provider=provider,
                    agent_client=agent_client,
                    agent_model=agent_model,
                    utility_client=utility_client,
                    utility_model=utility_model,
                )
            )
        checks = _score(
            scenario,
            db_path=db_path,
            clock=clock,
            provider=provider,
            final_replies=final_replies,
            utility_client=utility_client,
            utility_model=utility_model,
            run_judge=run_judge,
        )
        return RunResult(run_index=run_index, checks=checks, final_replies=final_replies)
    except Exception as exc:  # a harness/agent crash fails the run, not the suite
        return RunResult(
            run_index=run_index,
            final_replies=final_replies,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _setup_world(
    scenario: Scenario, db_path: Path, clock: FrozenClock, provider: FakeCalendarProvider
) -> None:
    with Store(db_path, clock=clock) as store:
        for spec in scenario.users:
            store.upsert_user(_user(spec))
        for fact in scenario.seed_facts:
            _seed_fact(store, fact)
    for ev in scenario.seed_events:
        provider.seed(ev.user, [_seed_event_draft(ev)])
    for failure in scenario.inject_failures:
        provider.inject_failure(failure.action, _failure_factory(failure.error), failure.times)


def _run_session(
    scenario: Scenario,
    session: Any,
    session_index: int,
    *,
    db_path: Path,
    clock: FrozenClock,
    provider: FakeCalendarProvider,
    agent_client: Any,
    agent_model: str,
    utility_client: Any,
    utility_model: str,
) -> list[str]:
    replies: list[str] = []
    with Store(db_path, clock=clock) as store:
        # a restart starts a fresh conversation; profile memory + traces persist
        if session_index > 0:
            store.conn.execute("DELETE FROM messages")
            store.conn.commit()
        user = store.get_user_by_email(session.user)
        if user is None:
            raise KeyError(f"session references unknown user {session.user!r}")
        rule_engine = RuleEngine(store, user)
        toolbox = Toolbox(
            provider=provider,
            store=store,
            user=user,
            clock=clock,
            rule_checker=rule_engine.check,
            sleep_fn=lambda _s: None,  # injected failures retry instantly
        )
        tracer = SQLiteTraceEmitter(store, clock=clock)
        extractor = EpisodicExtractor(utility_client, utility_model, store, clock)
        loop = AgentLoop(
            client=agent_client,
            model=agent_model,
            toolbox=toolbox,
            store=store,
            tracer=tracer,
            clock=clock,
            user=user,
            extractor=extractor,
        )
        for turn in session.turns:
            replies.append(loop.run_turn(turn))
    return replies


def _score(
    scenario: Scenario,
    *,
    db_path: Path,
    clock: FrozenClock,
    provider: FakeCalendarProvider,
    final_replies: list[str],
    utility_client: Any,
    utility_model: str,
    run_judge: bool,
):  # noqa: ANN202
    expect = scenario.expect
    window_start = clock.now() - timedelta(days=_READ_WINDOW_DAYS)
    window_end = clock.now() + timedelta(days=_READ_WINDOW_DAYS)
    events_by_user = {
        spec.email: _safe_list_events(provider, spec.email, window_start, window_end)
        for spec in scenario.users
    }
    with Store(db_path, clock=clock) as store:
        facts_by_user = {
            spec.email: store.list_facts(_user_id(spec.email)) for spec in scenario.users
        }
        tools_called = _tools_called(store, clock)

    checks = []
    checks += scorers.score_events(expect.events, events_by_user)
    checks += scorers.score_facts(expect.facts, facts_by_user)
    checks += scorers.score_trajectory(expect, tools_called)
    checks += scorers.score_reply_substrings(expect, final_replies)
    if run_judge and expect.judge:
        checks += scorers.score_judge(expect.judge, final_replies, utility_client, utility_model)
    return checks


def _safe_list_events(provider: FakeCalendarProvider, email, start, end):  # noqa: ANN202
    # drain any injected failures still queued for list_events so scoring the
    # final calendar state never crashes on a leftover failure
    for _ in range(5):
        try:
            return provider.list_events(email, start, end)
        except ProviderError:
            continue
    return provider.list_events(email, start, end)


def _tools_called(store: Store, clock: FrozenClock) -> list[str]:
    tracer = SQLiteTraceEmitter(store, clock=clock)
    names: list[str] = []
    for req in tracer.recent_requests(limit=10000):
        for span in tracer.spans_for(req["request_id"]):
            if span["kind"] == "tool_call":
                names.append(span["name"])
    return names
