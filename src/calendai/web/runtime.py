"""Per-request wiring: build the agent loop and the calendar provider for a user.

The provider is chosen by CALENDAI_PROVIDER:
- "google" (default): a GoogleCalendarProvider driven by the user's encrypted
  OAuth token via GoogleTokenManager;
- "fake": the in-memory provider, so the UI is demoable without Google
  credentials. Dev login (which only works with the fake provider) is gated
  behind CALENDAI_DEV_LOGIN to keep it out of any real deployment.
"""

from __future__ import annotations

import os
from typing import Any

from calendai.agent.loop import AgentLoop
from calendai.agent.tools import Toolbox
from calendai.auth.google_oauth import GoogleTokenManager
from calendai.auth.tokens import TokenCipher
from calendai.core.clock import Clock
from calendai.core.config import Settings
from calendai.core.models import User
from calendai.core.provider import CalendarProvider
from calendai.db.store import Store
from calendai.memory.enforcement import RuleEngine
from calendai.memory.episodic import EpisodicExtractor
from calendai.providers.fake import FakeCalendarProvider
from calendai.providers.google import GoogleCalendarProvider
from calendai.traces.emitter import SQLiteTraceEmitter


def provider_mode() -> str:
    return os.environ.get("CALENDAI_PROVIDER", "google").lower()


def dev_login_enabled() -> bool:
    return os.environ.get("CALENDAI_DEV_LOGIN") == "1" and provider_mode() == "fake"


def build_provider(
    store: Store,
    user: User,
    settings: Settings,
    cipher: TokenCipher,
    clock: Clock,
    *,
    shared_fake: FakeCalendarProvider | None = None,
) -> CalendarProvider:
    if provider_mode() == "fake":
        # one shared fake provider per process so demo bookings persist
        if shared_fake is None:
            raise RuntimeError("fake provider mode requires a shared FakeCalendarProvider")
        return shared_fake
    manager = GoogleTokenManager(store, cipher, settings, user.id, clock=clock)
    return GoogleCalendarProvider(
        token_provider=manager.get_access_token,
        refresh_fn=manager.force_refresh,
    )


def build_loop(
    store: Store,
    user: User,
    provider: CalendarProvider,
    settings: Settings,
    clock: Clock,
    agent_client: Any,
) -> AgentLoop:
    rule_engine = RuleEngine(store, user)
    toolbox = Toolbox(
        provider=provider,
        store=store,
        user=user,
        clock=clock,
        rule_checker=rule_engine.check,
    )
    tracer = SQLiteTraceEmitter(store, clock=clock)
    extractor = EpisodicExtractor(agent_client, settings.calendai_utility_model, store, clock)
    return AgentLoop(
        client=agent_client,
        model=settings.calendai_agent_model,
        toolbox=toolbox,
        store=store,
        tracer=tracer,
        clock=clock,
        user=user,
        extractor=extractor,
    )
