"""Developer REPL: chat with the agent from a terminal.

This exists for fast iteration during development (and the Batch 2 gate
demo) - the web UI arrives later. Uses the real Anthropic API with the fake
calendar provider by default; --provider google switches to a real calendar
once the Google track is merged.

Usage:
    python -m calendai.cli --user alice@example.com
    /facts   show stored profile facts
    /trace   show spans of the last request
    /quit    exit
"""

from __future__ import annotations

import argparse
import sys
import uuid

import anthropic

from calendai.agent.loop import AgentLoop
from calendai.agent.tools import Toolbox
from calendai.core.clock import SystemClock
from calendai.core.config import get_settings
from calendai.core.models import User
from calendai.db.store import Store
from calendai.providers.fake import FakeCalendarProvider
from calendai.traces.emitter import SQLiteTraceEmitter


def build_loop(user_email: str, provider_name: str, db_path: str | None = None) -> AgentLoop:
    settings = get_settings()
    clock = SystemClock()
    store = Store(db_path or settings.calendai_db_path, clock=clock)

    user = store.get_user_by_email(user_email)
    if user is None:
        user = store.upsert_user(
            User(
                id=f"u_{uuid.uuid4().hex[:8]}",
                email=user_email,
                display_name=user_email.split("@")[0],
            )
        )

    if provider_name == "fake":
        provider = FakeCalendarProvider(clock)
    else:  # pragma: no cover - google path lands with Batch 4
        raise SystemExit("provider 'google' arrives with the Google track merge (Batch 4)")

    toolbox = Toolbox(provider=provider, store=store, user=user, clock=clock)
    tracer = SQLiteTraceEmitter(store, clock=clock)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return AgentLoop(
        client=client,
        model=settings.calendai_agent_model,
        toolbox=toolbox,
        store=store,
        tracer=tracer,
        clock=clock,
        user=user,
    )


def _print_facts(loop: AgentLoop) -> None:
    facts = loop.store.list_facts(loop.user.id)
    if not facts:
        print("  (no facts stored)")
    for fact in facts:
        print(f"  [{fact.fact_type.value}] {fact.key}: {fact.statement}")


def _print_trace(loop: AgentLoop) -> None:
    if not loop.last_request_id:
        print("  (no request yet)")
        return
    for span in loop.tracer.spans_for(loop.last_request_id):
        line = f"  {span['kind']:<10} {span['name']:<22}"
        if span["kind"] == "llm_call":
            p = span["payload"]
            line += f" in={p.get('input_tokens')} out={p.get('output_tokens')}"
            line += f" {p.get('latency_ms')}ms stop={p.get('stop_reason')}"
        elif span["kind"] == "tool_call":
            p = span["payload"]
            line += f" ok={p.get('ok')} retries={p.get('retries')}"
        print(line)
        if span["rationale"]:
            print(f"             why: {span['rationale']}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252, which cannot print emoji the model
    # may emit; never let an encoding error kill the REPL.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="CalendAI developer REPL")
    parser.add_argument("--user", required=True, help="user email")
    parser.add_argument("--provider", default="fake", choices=["fake", "google"])
    parser.add_argument("--db", default=None, help="override SQLite path")
    args = parser.parse_args(argv)

    loop = build_loop(args.user, args.provider, args.db)
    print(
        f"CalendAI REPL - user={args.user} provider={args.provider} "
        f"model={loop.model}\nCommands: /facts /trace /quit\n"
    )

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/facts":
            _print_facts(loop)
            continue
        if text == "/trace":
            _print_trace(loop)
            continue
        try:
            reply = loop.run_turn(text)
        except Exception as exc:  # surface, don't crash the REPL
            print(f"[error] {type(exc).__name__}: {exc}")
            continue
        print(f"calendai> {reply}\n")

    loop.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
