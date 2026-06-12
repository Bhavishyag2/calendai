"""OAuth flow and GoogleTokenManager tests, respx-mocked and FrozenClock-driven.

The token-manager tests exercise the full storage path (Store + TokenCipher
on a real temp SQLite db) to prove no plaintext token ever reaches disk.
"""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from calendai.auth.google_oauth import (
    AUTH_ENDPOINT,
    SCOPE,
    TOKEN_ENDPOINT,
    USERINFO_ENDPOINT,
    GoogleTokenManager,
    build_auth_url,
    exchange_code,
    fetch_user_email,
    refresh_access_token,
)
from calendai.auth.tokens import TokenCipher
from calendai.core.config import Settings
from calendai.core.models import User
from calendai.core.provider import AuthError, MalformedResponseError
from tests.conftest import FROZEN_NOW

USER_ID = "u1"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        google_client_id="client-id",
        google_client_secret="client-secret",
        google_redirect_uri="http://localhost:8000/auth/callback",
        _env_file=None,
    )


@pytest.fixture
def cipher() -> TokenCipher:
    return TokenCipher(Fernet.generate_key().decode())


@pytest.fixture
def manager(store, cipher, settings, clock) -> GoogleTokenManager:
    store.upsert_user(User(id=USER_ID, email="u1@example.com"))
    return GoogleTokenManager(store, cipher, settings, USER_ID, clock=clock)


def form_data(route: respx.Route) -> dict[str, list[str]]:
    return parse_qs(route.calls.last.request.content.decode())


def token_payload(expires_at: str) -> dict:
    return {"access_token": "at-1", "refresh_token": "rt-1", "expires_at": expires_at}


# -- auth URL ----------------------------------------------------------------


def test_build_auth_url_contains_all_oauth_params(settings):
    url = build_auth_url(settings, state="xyz123")

    base, _, query = url.partition("?")
    assert base == AUTH_ENDPOINT
    params = parse_qs(query)
    assert params == {
        "client_id": ["client-id"],
        "redirect_uri": ["http://localhost:8000/auth/callback"],
        "response_type": ["code"],
        "scope": [SCOPE],
        "access_type": ["offline"],
        "prompt": ["consent"],
        "state": ["xyz123"],
    }
    assert urlsplit(url).scheme == "https"


# -- code exchange and refresh ----------------------------------------------


@respx.mock
def test_exchange_code_computes_expires_at_from_clock(settings, clock):
    route = respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600},
        )
    )

    result = exchange_code(settings, "auth-code", clock=clock)

    assert result == {
        "access_token": "at-1",
        "refresh_token": "rt-1",
        "expires_at": (FROZEN_NOW + timedelta(seconds=3600)).isoformat(),
    }
    form = form_data(route)
    assert form["grant_type"] == ["authorization_code"]
    assert form["code"] == ["auth-code"]
    assert form["client_id"] == ["client-id"]
    assert form["client_secret"] == ["client-secret"]
    assert form["redirect_uri"] == ["http://localhost:8000/auth/callback"]


@respx.mock
def test_exchange_code_failure_raises_auth_error(settings, clock):
    respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(400, json={"error": "invalid_grant"})
    )
    with pytest.raises(AuthError):
        exchange_code(settings, "bad-code", clock=clock)


@respx.mock
def test_exchange_code_non_json_body_raises_malformed(settings, clock):
    respx.post(TOKEN_ENDPOINT).mock(return_value=httpx.Response(200, content=b"not json"))
    with pytest.raises(MalformedResponseError):
        exchange_code(settings, "auth-code", clock=clock)


@respx.mock
def test_refresh_keeps_old_refresh_token_when_google_omits_it(settings, clock):
    route = respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
    )

    result = refresh_access_token(settings, "rt-old", clock=clock)

    assert result["access_token"] == "at-2"
    assert result["refresh_token"] == "rt-old"  # preserved
    assert result["expires_at"] == (FROZEN_NOW + timedelta(seconds=3600)).isoformat()
    assert form_data(route)["grant_type"] == ["refresh_token"]


@respx.mock
def test_refresh_uses_new_refresh_token_when_present(settings, clock):
    respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "at-2", "refresh_token": "rt-new", "expires_in": 60},
        )
    )
    assert refresh_access_token(settings, "rt-old", clock=clock)["refresh_token"] == "rt-new"


# -- userinfo ---------------------------------------------------------------


@respx.mock
def test_fetch_user_email():
    route = respx.get(USERINFO_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"email": "person@example.com", "sub": "123"})
    )
    assert fetch_user_email("at-1") == "person@example.com"
    assert route.calls.last.request.headers["Authorization"] == "Bearer at-1"


@respx.mock
def test_fetch_user_email_missing_field_raises_malformed():
    respx.get(USERINFO_ENDPOINT).mock(return_value=httpx.Response(200, json={"sub": "123"}))
    with pytest.raises(MalformedResponseError):
        fetch_user_email("at-1")


# -- GoogleTokenManager -------------------------------------------------------


@respx.mock
def test_manager_returns_stored_token_when_not_near_expiry(manager, store, cipher):
    route = respx.post(TOKEN_ENDPOINT).mock(return_value=httpx.Response(500))
    expires_at = (FROZEN_NOW + timedelta(hours=1)).isoformat()
    store.save_oauth_token(USER_ID, token_payload(expires_at), cipher)

    assert manager.get_access_token() == "at-1"
    assert route.call_count == 0  # no refresh round trip


@respx.mock
def test_manager_auto_refreshes_when_clock_reaches_expiry_skew(manager, store, cipher, clock):
    route = respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
    )
    expires_at = (FROZEN_NOW + timedelta(hours=1)).isoformat()
    store.save_oauth_token(USER_ID, token_payload(expires_at), cipher)

    assert manager.get_access_token() == "at-1"  # fresh: no refresh yet
    clock.advance(timedelta(minutes=59, seconds=30))  # within the 60s skew window

    assert manager.get_access_token() == "at-2"
    assert route.call_count == 1

    stored = store.load_oauth_token(USER_ID, cipher)
    assert stored["access_token"] == "at-2"
    assert stored["refresh_token"] == "rt-1"  # carried over
    assert stored["expires_at"] == (clock.now() + timedelta(seconds=3600)).isoformat()


@respx.mock
def test_manager_force_refresh_ignores_expiry(manager, store, cipher):
    route = respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
    )
    expires_at = (FROZEN_NOW + timedelta(hours=1)).isoformat()
    store.save_oauth_token(USER_ID, token_payload(expires_at), cipher)

    assert manager.force_refresh() == "at-2"
    assert route.call_count == 1
    assert form_data(route)["refresh_token"] == ["rt-1"]


def test_manager_without_stored_token_raises_auth_error(manager):
    with pytest.raises(AuthError):
        manager.get_access_token()


def test_manager_without_refresh_token_raises_auth_error(manager, store, cipher):
    expired = (FROZEN_NOW - timedelta(hours=1)).isoformat()
    payload = {"access_token": "at-1", "refresh_token": None, "expires_at": expired}
    store.save_oauth_token(USER_ID, payload, cipher)

    with pytest.raises(AuthError, match="re-authenticate"):
        manager.get_access_token()


@respx.mock
def test_manager_never_persists_plaintext_tokens(manager, store, cipher):
    respx.post(TOKEN_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600},
        )
    )
    expired = (FROZEN_NOW - timedelta(hours=1)).isoformat()
    store.save_oauth_token(USER_ID, token_payload(expired), cipher)

    assert manager.get_access_token() == "at-2"  # refresh + re-save through the Store

    row = store.conn.execute(
        "SELECT token_blob FROM oauth_tokens WHERE user_id = ?", (USER_ID,)
    ).fetchone()
    blob = row["token_blob"]
    assert isinstance(blob, bytes)
    for secret in (b"at-1", b"at-2", b"rt-1", b"rt-2", b"access_token"):
        assert secret not in blob  # Fernet blob leaks nothing

    decrypted = cipher.decrypt(blob)  # but the cipher path still round-trips
    assert decrypted["access_token"] == "at-2"
    assert decrypted["refresh_token"] == "rt-2"
