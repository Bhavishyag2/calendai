"""Google OAuth 2.0 authorization-code flow over raw httpx.

Same architecture decision as the provider module: no google-auth or
oauthlib dependency. The flow is three small HTTP calls (code exchange,
token refresh, userinfo) and one URL builder - raw httpx keeps every byte
on the wire visible and respx-mockable.

Token discipline:
- expires_at is computed from expires_in via an injected Clock so expiry
  logic is deterministic under FrozenClock in tests.
- GoogleTokenManager persists tokens ONLY through the cipher-enforced
  Store API (save_oauth_token/load_oauth_token), so no plaintext token can
  reach disk, and nothing in this module logs or prints token values.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from calendai.auth.tokens import TokenCipher
from calendai.core.clock import Clock, SystemClock
from calendai.core.config import Settings
from calendai.core.provider import AuthError, MalformedResponseError, ServerError
from calendai.db.store import Store

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPE = "openid email https://www.googleapis.com/auth/calendar"

_EXPIRY_SKEW = timedelta(seconds=60)


def build_auth_url(settings: Settings, state: str) -> str:
    """Authorization URL the user is redirected to for consent.

    access_type=offline plus prompt=consent guarantees Google returns a
    refresh_token on the subsequent code exchange.
    """
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


def _post_token(form: dict[str, str], clock: Clock) -> dict[str, Any]:
    """POST to the token endpoint and normalize the response payload.

    Returns {access_token, refresh_token, expires_at} where expires_at is
    an aware ISO-8601 string computed from expires_in via the clock.
    """
    try:
        response = httpx.post(TOKEN_ENDPOINT, data=form)
    except httpx.HTTPError as exc:
        # stays inside the provider error taxonomy (retryable ServerError);
        # class name only AND `from None` - the form carries client_secret/
        # refresh_token, and exc.request would echo it into any traceback
        raise ServerError(f"token endpoint transport failure: {exc.__class__.__name__}") from None
    if response.status_code != 200:
        raise AuthError(f"token endpoint returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise MalformedResponseError("token endpoint returned a non-JSON body") from exc
    try:
        access_token = data["access_token"]
        expires_in = int(data["expires_in"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MalformedResponseError("token response is missing access_token/expires_in") from exc
    expires_at = (clock.now() + timedelta(seconds=expires_in)).isoformat()
    return {
        "access_token": access_token,
        "refresh_token": data.get("refresh_token"),
        "expires_at": expires_at,
    }


def exchange_code(settings: Settings, code: str, clock: Clock | None = None) -> dict[str, Any]:
    """Exchange an authorization code for tokens."""
    return _post_token(
        {
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code",
        },
        clock or SystemClock(),
    )


def refresh_access_token(
    settings: Settings, refresh_token: str, clock: Clock | None = None
) -> dict[str, Any]:
    """Refresh the access token; keeps the old refresh_token if Google
    omits it from the response (its usual behavior on refresh)."""
    payload = _post_token(
        {
            "refresh_token": refresh_token,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "grant_type": "refresh_token",
        },
        clock or SystemClock(),
    )
    if not payload["refresh_token"]:
        payload["refresh_token"] = refresh_token
    return payload


def fetch_user_email(access_token: str) -> str:
    """Resolve the authenticated user's email via the OIDC userinfo endpoint."""
    try:
        response = httpx.get(USERINFO_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"})
    except httpx.HTTPError as exc:
        # `from None`: exc.request carries the bearer header
        raise ServerError(f"userinfo transport failure: {exc.__class__.__name__}") from None
    if response.status_code != 200:
        raise AuthError(f"userinfo endpoint returned HTTP {response.status_code}")
    try:
        # TypeError covers a JSON list/null/scalar body (not subscriptable)
        email = response.json()["email"]
    except (ValueError, KeyError, TypeError) as exc:
        raise MalformedResponseError("userinfo response is missing email") from exc
    if not isinstance(email, str) or not email:
        raise MalformedResponseError("userinfo email is not a non-empty string")
    return email


class GoogleTokenManager:
    """Bridges encrypted token storage and GoogleCalendarProvider.

    Wire-up:
        manager = GoogleTokenManager(store, cipher, settings, user_id, clock)
        provider = GoogleCalendarProvider(
            token_provider=manager.get_access_token,
            refresh_fn=manager.force_refresh,
        )

    get_access_token auto-refreshes when the stored token is within 60s of
    expiry (clock skew guard); force_refresh refreshes unconditionally,
    which is what the provider calls after an API-side 401.
    """

    def __init__(
        self,
        store: Store,
        cipher: TokenCipher,
        settings: Settings,
        user_id: str,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._cipher = cipher
        self._settings = settings
        self._user_id = user_id
        self._clock = clock or SystemClock()

    def get_access_token(self) -> str:
        """Current access token, refreshed first if expired or near expiry."""
        payload = self._load()
        if self._needs_refresh(payload):
            return self._refresh_and_save(payload)
        return payload["access_token"]

    def force_refresh(self) -> str:
        """Refresh unconditionally (e.g. the API just rejected the token)."""
        return self._refresh_and_save(self._load())

    def _load(self) -> dict[str, Any]:
        payload = self._store.load_oauth_token(self._user_id, self._cipher)
        if payload is None:
            raise AuthError(f"no stored OAuth token for user {self._user_id!r}")
        return payload

    def _needs_refresh(self, payload: dict[str, Any]) -> bool:
        raw = payload.get("expires_at")
        if not raw:
            return True
        try:
            expires_at = datetime.fromisoformat(raw)
        except ValueError:
            return True
        if expires_at.tzinfo is None:
            return True  # naive timestamps are untrusted; refresh defensively
        return self._clock.now() >= expires_at - _EXPIRY_SKEW

    def _refresh_and_save(self, payload: dict[str, Any]) -> str:
        refresh_token = payload.get("refresh_token")
        if not refresh_token:
            raise AuthError("stored token has no refresh_token; user must re-authenticate")
        fresh = refresh_access_token(self._settings, refresh_token, self._clock)
        self._store.save_oauth_token(self._user_id, fresh, self._cipher)
        return fresh["access_token"]
