from __future__ import annotations

import json
import re
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from calendai.auth.google_oauth import TOKEN_ENDPOINT, USERINFO_ENDPOINT
from calendai.core.config import Settings
from calendai.core.models import EventDraft
from calendai.db.store import Store
from calendai.providers.fake import FakeCalendarProvider, sequential_ids
from calendai.web.app import create_app
from tests.conftest import FROZEN_NOW
from tests.scripted_client import text_response, tool_call


class WebAgentClient:
    """Books at 10am tomorrow when asked; extracts a contact when taught;
    otherwise acknowledges. Used for both the loop and episodic extraction."""

    CONTACT = json.dumps(
        [
            {
                "fact_type": "contact",
                "key": "contact:alex",
                "value": {"email": "alex@corp.com"},
                "statement": "Alex is alex@corp.com.",
            }
        ]
    )

    def __init__(self):
        self.messages = self

    def create(self, *, messages, system=None, **kwargs):
        if system and "extract durable profile facts" in system.lower():
            user = messages[0]["content"].lower()
            return text_response(self.CONTACT if "alex" in user else "[]")
        last = messages[-1]["content"]
        if isinstance(last, list) and last and last[0].get("type") == "tool_result":
            return text_response("Done - booked your standup.")
        text = (last if isinstance(last, str) else "").lower()
        if "book" in text:
            return tool_call(
                "create_event",
                {
                    "title": "Standup",
                    "start": "2026-06-16T10:00:00+05:30",
                    "end": "2026-06-16T10:30:00+05:30",
                    "rationale": "user asked",
                },
            )
        return text_response("Got it.")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CALENDAI_PROVIDER", "fake")
    monkeypatch.setenv("CALENDAI_DEV_LOGIN", "1")


@pytest.fixture
def app_client(tmp_path, clock, env):
    settings = Settings(
        anthropic_api_key="test",
        calendai_fernet_key=Fernet.generate_key().decode(),
        google_client_id="cid",
        google_client_secret="secret",
        google_redirect_uri="http://localhost:8000/auth/callback",
    )
    store = Store(tmp_path / "web.db", clock=clock, check_same_thread=False)
    fake = FakeCalendarProvider(clock, sequential_ids())
    app = create_app(
        settings=settings,
        clock=clock,
        agent_client=WebAgentClient(),
        store=store,
        shared_fake=fake,
    )
    with TestClient(app) as tc:
        yield tc, store, fake


def _dev_login(tc, email="alice@example.com"):
    r = tc.post("/auth/dev-login", json={"email": email})
    assert r.status_code == 200
    return r


# -- auth gating --------------------------------------------------------------


def test_unauthenticated_endpoints_401(app_client):
    tc, _, _ = app_client
    for path in ("/api/me", "/api/facts", "/api/traces"):
        assert tc.get(path).status_code == 401
    assert tc.post("/api/chat", json={"message": "hi"}).status_code == 401


def test_index_served(app_client):
    tc, _, _ = app_client
    r = tc.get("/")
    assert r.status_code == 200 and "CalendAI" in r.text


# -- dev login ----------------------------------------------------------------


def test_dev_login_then_me(app_client):
    tc, _, _ = app_client
    _dev_login(tc)
    me = tc.get("/api/me")
    assert me.status_code == 200 and me.json()["email"] == "alice@example.com"


def test_dev_login_disabled_without_env(tmp_path, clock, monkeypatch):
    monkeypatch.setenv("CALENDAI_PROVIDER", "fake")
    monkeypatch.delenv("CALENDAI_DEV_LOGIN", raising=False)
    settings = Settings(anthropic_api_key="t", calendai_fernet_key=Fernet.generate_key().decode())
    store = Store(tmp_path / "w.db", clock=clock, check_same_thread=False)
    fake = FakeCalendarProvider(clock)
    with TestClient(
        create_app(settings=settings, clock=clock, store=store, shared_fake=fake)
    ) as tc:
        assert tc.post("/auth/dev-login", json={"email": "x@y.com"}).status_code == 404


# -- chat end-to-end through the real stack -----------------------------------


def test_chat_books_event_and_populates_traces(app_client):
    tc, store, fake = app_client
    _dev_login(tc)
    r = tc.post("/api/chat", json={"message": "book a standup tomorrow at 10am"})
    assert r.status_code == 200
    assert "booked" in r.json()["reply"].lower()
    # the event really landed on the fake calendar
    events = fake.list_events("alice@example.com", FROZEN_NOW, FROZEN_NOW.replace(day=20))
    assert [e.title for e in events] == ["Standup"]
    # and a trace exists for the turn
    traces = tc.get("/api/traces").json()["requests"]
    assert len(traces) == 1 and "standup" in traces[0]["user_message"].lower()


def test_chat_extracts_and_shows_memory(app_client):
    tc, _, _ = app_client
    _dev_login(tc)
    tc.post("/api/chat", json={"message": "fyi, Alex is alex@corp.com"})
    facts = tc.get("/api/facts").json()["facts"]
    assert any(f["key"] == "contact:alex" for f in facts)


def test_empty_message_rejected(app_client):
    tc, _, _ = app_client
    _dev_login(tc)
    assert tc.post("/api/chat", json={"message": "   "}).status_code == 400


def test_oversized_message_rejected(app_client):
    tc, _, _ = app_client
    _dev_login(tc)
    assert tc.post("/api/chat", json={"message": "x" * 5000}).status_code == 413


# -- session + CSRF + headers (gate-6 security fixes) -------------------------


def test_logout_invalidates_server_side_session(app_client):
    tc, store, _ = app_client
    _dev_login(tc)
    token = tc.cookies.get("calendai_session")
    assert store.get_session_user(token) is not None
    tc.post("/auth/logout")
    assert store.get_session_user(token) is None  # row gone, not just the cookie
    # even replaying the old token is dead (server-side invalidation)
    tc.cookies.set("calendai_session", token)
    assert tc.get("/api/me").status_code == 401


def test_sessions_carry_expiry(app_client):
    tc, store, _ = app_client
    _dev_login(tc)
    token = tc.cookies.get("calendai_session")
    row = store.conn.execute("SELECT expires_at FROM sessions WHERE token = ?", (token,)).fetchone()
    assert row["expires_at"] is not None  # not a never-expiring session


def test_cross_origin_chat_rejected(app_client):
    tc, _, _ = app_client
    _dev_login(tc)
    r = tc.post(
        "/api/chat",
        json={"message": "book a standup tomorrow at 10am"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_security_headers_present(app_client):
    tc, _, _ = app_client
    r = tc.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" in r.headers


def test_facts_endpoint_returns_raw_statement_not_html(app_client):
    # the API returns the raw statement; the SPA renders it with textContent.
    # Here we assert the API does not itself HTML-encode (no double-encoding),
    # so the XSS defense lives at the single, correct layer (the DOM).
    tc, _, _ = app_client
    _dev_login(tc)
    tc.post("/api/chat", json={"message": "fyi, Alex is alex@corp.com"})
    facts = tc.get("/api/facts").json()["facts"]
    assert all("<" not in f["key"] for f in facts)


# -- trace isolation between users --------------------------------------------


def test_user_cannot_read_another_users_trace(app_client):
    tc, _, _ = app_client
    _dev_login(tc, "alice@example.com")
    tc.post("/api/chat", json={"message": "book a standup tomorrow at 10am"})
    rid = tc.get("/api/traces").json()["requests"][0]["request_id"]
    # switch to Bob
    tc.post("/auth/logout")
    _dev_login(tc, "bob@example.com")
    assert tc.get(f"/api/traces/{rid}").status_code == 404
    assert tc.get("/api/traces").json()["requests"] == []  # Bob sees none


# -- OAuth flow ---------------------------------------------------------------


def test_oauth_callback_rejects_bad_state(app_client):
    tc, _, _ = app_client
    # no state cookie set -> CSRF rejection
    r = tc.get("/auth/callback", params={"code": "x", "state": "forged"}, follow_redirects=False)
    assert r.status_code == 400


def test_oauth_callback_handles_user_denied_consent(app_client):
    tc, _, _ = app_client
    login = tc.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    # Google redirects with ?error=access_denied (and the valid state) on denial
    r = tc.get(
        "/auth/callback",
        params={"error": "access_denied", "state": state},
        follow_redirects=False,
    )
    assert r.status_code == 400 and "not completed" in r.json()["detail"]


# -- destructive-action confirmation across requests (cross-request gate) -----


class ConfirmDeleteClient:
    """Drives the two-turn delete flow the way a real model would, but
    deterministically: it issues delete_event without a token, relays the
    confirmation_required result, and on the turn where the system prompt
    carries an armed token (placed there by the persisted gate) it re-issues
    delete_event with that exact token. Used to prove the confirmation gate
    survives the web app's per-request loop rebuild."""

    def __init__(self, event_id: str):
        self.event_id = event_id
        self.messages = self

    def create(self, *, messages, system=None, **kwargs):
        if system and "extract durable profile facts" in system.lower():
            return text_response("[]")
        last = messages[-1]["content"]
        if isinstance(last, list) and last and last[0].get("type") == "tool_result":
            result = json.loads(last[0]["content"])
            if result.get("error_type") == "confirmation_required":
                return text_response("This will permanently delete your standup. Confirm?")
            return text_response("Done - your standup has been deleted.")
        token_match = re.search(r"confirmation_token='(confirm-[0-9a-f]+)'", system or "")
        args = {"event_id": self.event_id, "rationale": "user asked to delete the standup"}
        if token_match:  # the gate armed a token after the user confirmed
            args["confirmation_token"] = token_match.group(1)
        return tool_call("delete_event", args)


def test_confirmation_gate_survives_per_request_rebuild(tmp_path, clock, env):
    # Each /api/chat request rebuilds the loop + ConfirmationGate from scratch,
    # so the pending confirmation MUST round-trip through the store, or the
    # "yes" turn can never recover the token. This test fails (infinite
    # re-confirmation) if the gate keeps its pending state only in memory.
    settings = Settings(
        anthropic_api_key="test",
        calendai_fernet_key=Fernet.generate_key().decode(),
    )
    store = Store(tmp_path / "confirm.db", clock=clock, check_same_thread=False)
    fake = FakeCalendarProvider(clock, sequential_ids())
    # seed an event to delete
    event = fake.create_event(
        "alice@example.com",
        EventDraft(title="Standup", start=FROZEN_NOW, end=FROZEN_NOW + timedelta(minutes=30)),
    )
    app = create_app(
        settings=settings,
        clock=clock,
        agent_client=ConfirmDeleteClient(event.id),
        store=store,
        shared_fake=fake,
    )
    with TestClient(app) as tc:
        _dev_login(tc)
        # turn 1: ask to delete -> gate issues a token, persisted, and asks to confirm
        r1 = tc.post("/api/chat", json={"message": "delete my standup"})
        assert r1.status_code == 200 and "confirm" in r1.json()["reply"].lower()
        n_pending = store.conn.execute("SELECT COUNT(*) c FROM pending_confirmations").fetchone()
        assert n_pending["c"] == 1
        # turn 2: confirm -> a FRESH gate recovers the token from the store and deletes
        r2 = tc.post("/api/chat", json={"message": "yes, delete it"})
        assert r2.status_code == 200 and "deleted" in r2.json()["reply"].lower()

    # the event is really gone, and the one-shot confirmation was consumed
    remaining = fake.list_events("alice@example.com", FROZEN_NOW, FROZEN_NOW.replace(day=20))
    assert remaining == []
    assert store.conn.execute("SELECT COUNT(*) c FROM pending_confirmations").fetchone()["c"] == 0


@respx.mock
def test_oauth_callback_happy_path(app_client):
    tc, store, _ = app_client
    respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}
        )
    )
    respx.get(USERINFO_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"email": "carol@example.com", "email_verified": True}
        )
    )
    # drive the real state handshake: login sets the state cookie
    login = tc.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlsplit(login.headers["location"]).query)["state"][0]
    cb = tc.get("/auth/callback", params={"code": "abc", "state": state}, follow_redirects=False)
    assert cb.status_code == 303
    assert tc.get("/api/me").json()["email"] == "carol@example.com"
    # the token was persisted ENCRYPTED, never as plaintext
    user = store.get_user_by_email("carol@example.com")
    row = store.conn.execute(
        "SELECT token_blob FROM oauth_tokens WHERE user_id = ?", (user.id,)
    ).fetchone()
    assert b"at-1" not in row["token_blob"] and b"rt-1" not in row["token_blob"]
