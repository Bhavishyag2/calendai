from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from calendai.auth.google_oauth import TOKEN_ENDPOINT, USERINFO_ENDPOINT
from calendai.core.config import Settings
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
            return text_response("Done — booked your standup.")
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


@respx.mock
def test_oauth_callback_happy_path(app_client):
    tc, store, _ = app_client
    respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}
        )
    )
    respx.get(USERINFO_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"email": "carol@example.com"})
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
