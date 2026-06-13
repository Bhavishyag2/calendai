"""FastAPI application: OAuth login, chat (SSE), memory sidebar, trace viewer.

Security posture (Batch 6 gate focus):
- session cookies are opaque server-side tokens (secrets.token_urlsafe), set
  HttpOnly + SameSite=Lax (+ Secure when CALENDAI_HTTPS=1);
- the OAuth flow carries a random `state` in a short-lived cookie and rejects
  a callback whose state does not match (CSRF defense);
- OAuth tokens are only ever persisted encrypted (cipher-enforced store API);
- no endpoint returns a token, and the trace viewer only exposes the current
  user's own requests.
"""

from __future__ import annotations

import secrets
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from calendai.auth.google_oauth import build_auth_url, exchange_code, fetch_user_email
from calendai.auth.tokens import TokenCipher
from calendai.core.clock import Clock, SystemClock
from calendai.core.config import Settings, get_settings
from calendai.core.models import User
from calendai.db.store import Store
from calendai.memory.profile import profile_facts
from calendai.providers.fake import FakeCalendarProvider
from calendai.traces.emitter import SQLiteTraceEmitter
from calendai.web import runtime

SESSION_COOKIE = "calendai_session"
STATE_COOKIE = "calendai_oauth_state"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_TTL = timedelta(days=7)
MAX_MESSAGE_CHARS = 4000

# Defense in depth for the inline SPA. 'unsafe-inline' is a deliberate, noted
# weakening because the single-file SPA carries inline <script>/<style>; a build
# step that externalizes them would let this tighten to nonces.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
    ),
}


class AppState:
    def __init__(
        self,
        *,
        store: Store,
        settings: Settings,
        cipher: TokenCipher,
        clock: Clock,
        agent_client: Any,
        shared_fake: FakeCalendarProvider | None,
        secure_cookies: bool,
    ) -> None:
        self.store = store
        self.settings = settings
        self.cipher = cipher
        self.clock = clock
        self.agent_client = agent_client
        self.shared_fake = shared_fake
        self.secure_cookies = secure_cookies
        self.lock = threading.Lock()  # serialize agent turns / DB writes


def _state(request: Request) -> AppState:
    return request.app.state.ctx


def _current_user(request: Request) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    user = _state(request).store.get_session_user(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


def _require_same_origin(request: Request) -> None:
    """CSRF defense for state-changing requests: when an Origin (or Referer)
    header is present it MUST match this app's own origin. Browsers always send
    Origin on cross-site fetch/XHR, so a forged request from another site is
    rejected here regardless of SameSite behaviour."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin is None:
        return  # same-origin form posts may omit it; cookies+SameSite still apply
    parts = urlsplit(origin)
    if (parts.scheme, parts.hostname, parts.port) != (
        request.url.scheme,
        request.url.hostname,
        request.url.port,
    ):
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def _set_session_cookie(response: Response, ctx: AppState, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=ctx.secure_cookies,
        path="/",
        max_age=int(SESSION_TTL.total_seconds()),
    )


def create_app(
    *,
    settings: Settings | None = None,
    clock: Clock | None = None,
    agent_client: Any = None,
    store: Store | None = None,
    shared_fake: FakeCalendarProvider | None = None,
    secure_cookies: bool = False,
) -> FastAPI:
    settings = settings or get_settings()
    clock = clock or SystemClock()
    cipher = TokenCipher(settings.calendai_fernet_key)
    store = store or Store(settings.calendai_db_path, clock=clock, check_same_thread=False)
    if runtime.provider_mode() == "fake" and shared_fake is None:
        shared_fake = FakeCalendarProvider(clock)

    app = FastAPI(title="CalendAI")
    app.state.ctx = AppState(
        store=store,
        settings=settings,
        cipher=cipher,
        clock=clock,
        agent_client=agent_client,
        shared_fake=shared_fake,
        secure_cookies=secure_cookies,
    )

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # noqa: ANN202
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - cohesive route table
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/me")
    def me(user: User = Depends(_current_user)) -> dict[str, str]:
        return {"email": user.email, "display_name": user.display_name, "timezone": user.timezone}

    # -- auth ---------------------------------------------------------------

    @app.get("/auth/login")
    def login(request: Request) -> RedirectResponse:
        ctx = _state(request)
        state = secrets.token_urlsafe(24)
        url = build_auth_url(ctx.settings, state)
        resp = RedirectResponse(url)
        resp.set_cookie(
            STATE_COOKIE,
            state,
            httponly=True,
            samesite="lax",
            secure=ctx.secure_cookies,
            max_age=600,
        )
        return resp

    @app.get("/auth/callback")
    def callback(request: Request, code: str = "", state: str = "") -> RedirectResponse:
        ctx = _state(request)
        expected = request.cookies.get(STATE_COOKIE)
        if not expected or not state or not secrets.compare_digest(state, expected):
            raise HTTPException(status_code=400, detail="invalid OAuth state")
        if not code:
            raise HTTPException(status_code=400, detail="missing authorization code")
        token_payload = exchange_code(ctx.settings, code, clock=ctx.clock)
        email = fetch_user_email(token_payload["access_token"])
        user = _upsert_user(ctx, email)
        ctx.store.save_oauth_token(user.id, token_payload, ctx.cipher)
        session_token = _new_session(ctx, user)
        resp = RedirectResponse("/", status_code=303)
        _set_session_cookie(resp, ctx, session_token)
        resp.delete_cookie(STATE_COOKIE)
        return resp

    @app.post("/auth/dev-login")
    def dev_login(request: Request, payload: dict[str, str]) -> JSONResponse:
        if not runtime.dev_login_enabled():
            raise HTTPException(status_code=404, detail="dev login disabled")
        _require_same_origin(request)
        ctx = _state(request)
        email = (payload.get("email") or "").strip()
        if "@" not in email or len(email) > 254:
            raise HTTPException(status_code=400, detail="a valid email is required")
        user = _upsert_user(ctx, email)
        session_token = _new_session(ctx, user)
        resp = JSONResponse({"email": user.email})
        _set_session_cookie(resp, ctx, session_token)
        return resp

    @app.post("/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            _state(request).store.delete_session(token)  # server-side invalidation
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    # -- chat ---------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(request: Request, payload: dict[str, str]) -> JSONResponse:
        user = _current_user(request)
        _require_same_origin(request)
        message = (payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        if len(message) > MAX_MESSAGE_CHARS:
            raise HTTPException(status_code=413, detail="message too long")
        reply, request_id = await run_in_threadpool(_run_turn, _state(request), user, message)
        return JSONResponse({"reply": reply, "request_id": request_id})

    # -- memory + traces ----------------------------------------------------

    @app.get("/api/facts")
    def facts(request: Request, user: User = Depends(_current_user)) -> dict[str, list]:
        rows = profile_facts(_state(request).store, user.id)
        return {
            "facts": [
                {"type": f.fact_type.value, "key": f.key, "statement": f.statement} for f in rows
            ]
        }

    @app.get("/api/traces")
    def traces(request: Request, user: User = Depends(_current_user)) -> dict[str, list]:
        tracer = SQLiteTraceEmitter(_state(request).store, clock=_state(request).clock)
        reqs = [r for r in tracer.recent_requests(limit=50) if r["user_id"] == user.id]
        return {"requests": reqs}

    @app.get("/api/traces/{request_id}")
    def trace_detail(
        request: Request, request_id: str, user: User = Depends(_current_user)
    ) -> dict:
        ctx = _state(request)
        tracer = SQLiteTraceEmitter(ctx.store, clock=ctx.clock)
        owner = next(
            (r for r in tracer.recent_requests(limit=500) if r["request_id"] == request_id), None
        )
        if owner is None or owner["user_id"] != user.id:
            raise HTTPException(status_code=404, detail="trace not found")
        return {"request": owner, "spans": tracer.spans_for(request_id)}


# -- helpers (module-level so they stay small + testable) --------------------


def _upsert_user(ctx: AppState, email: str) -> User:
    existing = ctx.store.get_user_by_email(email)
    if existing is not None:
        return existing
    user = User(
        id="u_" + secrets.token_hex(6),
        email=email,
        display_name=email.split("@")[0],
    )
    return ctx.store.upsert_user(user)


def _new_session(ctx: AppState, user: User) -> str:
    token = secrets.token_urlsafe(32)
    ctx.store.create_session(token, user.id, expires_at=ctx.clock.now() + SESSION_TTL)
    return token


def _run_turn(ctx: AppState, user: User, message: str) -> tuple[str, str]:
    with ctx.lock:  # one agent turn at a time: serializes the shared SQLite conn
        provider = runtime.build_provider(
            ctx.store, user, ctx.settings, ctx.cipher, ctx.clock, shared_fake=ctx.shared_fake
        )
        loop = runtime.build_loop(
            ctx.store, user, provider, ctx.settings, ctx.clock, ctx.agent_client
        )
        reply = loop.run_turn(message)
        return reply, loop.last_request_id or ""
