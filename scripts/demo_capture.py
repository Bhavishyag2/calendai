"""Capture the CalendAI demo against the REAL Google Calendar.

Runs the actual web app (google provider, Haiku agent), drives a Chromium
browser through Playwright, and records BOTH a continuous video (.webm) and a
screenshot at every step. Mid-demo it restarts the server process to prove that
long-term memory (and a signed-in session) survive a real restart on disk.

Prereqs (already done in this session):
  - the user has signed in once via the browser (an oauth_tokens row exists);
  - playwright + chromium installed project-local (PLAYWRIGHT_BROWSERS_PATH).

Run:  ./.venv/Scripts/python scripts/demo_capture.py
Artifacts land in docs/demo/ (screenshots/ and video/).
"""

from __future__ import annotations

import contextlib
import os
import secrets
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".playwright-browsers"))

# import after sys.path so the package resolves when run as a script
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from calendai.auth.google_oauth import GoogleTokenManager  # noqa: E402
from calendai.auth.tokens import TokenCipher  # noqa: E402
from calendai.core.clock import SystemClock  # noqa: E402
from calendai.core.config import Settings  # noqa: E402
from calendai.db.store import Store  # noqa: E402
from calendai.providers.google import GoogleCalendarProvider  # noqa: E402

PORT = 8000
BASE = f"http://localhost:{PORT}"
SHOTS = ROOT / "docs" / "demo" / "screenshots"
VIDEO = ROOT / "docs" / "demo" / "video"


def signed_in_user_id(store: Store) -> str:
    row = store.conn.execute(
        "SELECT u.id FROM users u JOIN oauth_tokens t ON t.user_id = u.id "
        "ORDER BY t.updated_at DESC"
    ).fetchone()
    if row is None:
        raise SystemExit("No signed-in Google user found. Sign in at /auth/login first.")
    return row["id"]


def mint_session(store: Store, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    store.create_session(token, user_id, expires_at=store._clock.now() + timedelta(hours=2))
    return token


def start_server() -> subprocess.Popen:
    env = dict(os.environ, CALENDAI_PROVIDER="google")
    return subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "calendai.web"],
        cwd=str(ROOT),
        env=env,
    )


def wait_ready(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            # any HTTP answer (200 for /, 401 for /api/me) means the app is up
            httpx.get(BASE + "/", timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(0.4)
    raise SystemExit("server did not become ready")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def send(
    page,
    text: str,
    shot: str | None = None,
    timeout: int = 90_000,
    wait_fact: bool = False,
) -> None:
    """Type a message, send it, wait for the assistant's reply to render.

    wait_fact=True additionally waits for the Memory panel's async refresh to
    land a fact, so a screenshot taken right after teaching a rule actually
    shows it (the panel refresh races the reply otherwise)."""
    before = page.locator(".bot").count()
    page.fill("#box", text)
    page.click("#chat button")
    page.wait_for_function(
        "n => document.querySelectorAll('.bot').length > n", arg=before, timeout=timeout
    )
    if wait_fact:
        # best-effort: don't abort the demo if the panel is slow/empty
        with contextlib.suppress(Exception):
            page.wait_for_selector("#facts .fact", timeout=8_000)
    page.wait_for_timeout(1500)  # let the memory + traces panels finish refreshing
    if shot:
        page.screenshot(path=str(SHOTS / shot), full_page=True)


def reset_demo_state(store: Store, settings: Settings, user_id: str) -> None:
    """Start from a clean slate so the demo visibly LEARNS the rule from zero:
    clear this user's memory, chat history, and any pending confirmation, and
    delete leftover 'Standup' events from a previous run off the REAL calendar."""
    store.conn.execute("DELETE FROM memory_facts WHERE user_id = ?", (user_id,))
    store.conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    store.conn.execute("DELETE FROM pending_confirmations WHERE user_id = ?", (user_id,))
    # also clear traces so the sidebar starts empty (spans cascade on delete)
    store.conn.execute("DELETE FROM trace_requests WHERE user_id = ?", (user_id,))
    store.conn.commit()
    user = store.get_user(user_id)
    cipher = TokenCipher(settings.calendai_fernet_key)
    mgr = GoogleTokenManager(store, cipher, settings, user_id, clock=store._clock)
    prov = GoogleCalendarProvider(
        token_provider=mgr.get_access_token, refresh_fn=mgr.force_refresh
    )
    try:
        now = store._clock.now()
        for e in prov.list_events(user.email, now - timedelta(days=1), now + timedelta(days=4)):
            if e.title.strip().lower() == "standup":
                prov.delete_event(user.email, e.id)
    finally:
        prov.close()


def _drive(page, proc_ref: list, settings: Settings) -> None:
    """Run the scripted walkthrough. proc_ref[0] holds the live server process
    so the mid-demo restart can replace it and teardown can always reach it."""
    page.goto(BASE)
    page.wait_for_selector("#appui", state="visible", timeout=15_000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(SHOTS / "00-signed-in.png"), full_page=True)

    if os.environ.get("DEMO_SMOKE") == "1":
        page.wait_for_timeout(500)  # harness check only; no Anthropic API calls
        print("SMOKE OK")
        return

    # 1. book via natural language -> real event on Google Calendar
    send(page, "Book a standup tomorrow at 10am.", "01-book-standup.png")
    # 2. teach a hard rule -> persisted to long-term memory (wait for the panel)
    send(
        page,
        "From now on, never schedule me for any meeting before 10am.",
        "02-teach-rule.png",
        wait_fact=True,
    )

    # 3. RESTART the server process: SQLite (memory + session) persists on disk
    print("restarting server to prove persistence...")
    stop_server(proc_ref[0])
    proc_ref[0] = start_server()
    wait_ready()
    page.reload()
    page.wait_for_selector("#appui", state="visible", timeout=15_000)
    page.wait_for_timeout(1000)
    page.screenshot(path=str(SHOTS / "03-after-restart-memory-intact.png"), full_page=True)

    # 4. try to violate the rule -> refused by CODE (RuleEngine), not just the prompt
    send(page, "Book a 9am meeting tomorrow.", "04-rule-enforced-refusal.png")
    # 5. graceful handling of an impossible request -> no hallucinated result
    send(page, "Cancel my dentist appointment next Tuesday.", "05-graceful-no-such-event.png")
    # 6. destructive action: confirmation required (two-step, code-enforced)
    send(page, "Delete my standup tomorrow.", "06-delete-confirmation-required.png")
    # 7. confirm -> the gate (now persisted across requests) validates and deletes
    send(page, "Yes, delete it.", "07-delete-confirmed.png")

    # 8. open the trace viewer for a turn -> full tool-call audit
    traces = page.locator(".trace")
    if traces.count():
        traces.last.click()  # expand the earliest visible turn (the booking)
        page.wait_for_timeout(600)
        page.screenshot(path=str(SHOTS / "08-trace-viewer.png"), full_page=True)
    page.wait_for_timeout(800)


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    VIDEO.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    store = Store(settings.calendai_db_path, clock=SystemClock(), check_same_thread=False)
    user_id = signed_in_user_id(store)
    reset_demo_state(store, settings, user_id)
    token = mint_session(store, user_id)
    print(f"driving as user_id={user_id}")

    proc_ref = [start_server()]
    wait_ready()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1280, "height": 820},
                record_video_dir=str(VIDEO),
                record_video_size={"width": 1280, "height": 820},
            )
            context.add_cookies([{"name": "calendai_session", "value": token, "url": BASE}])
            page = context.new_page()
            try:
                _drive(page, proc_ref, settings)
            finally:
                video_path = None
                with contextlib.suppress(Exception):  # best-effort video finalization
                    video_path = page.video.path()
                context.close()  # flushes the .webm to disk
                browser.close()
                if video_path:
                    print(f"video: {video_path}")
    finally:
        stop_server(proc_ref[0])
        print(f"screenshots: {SHOTS}")


if __name__ == "__main__":
    main()
