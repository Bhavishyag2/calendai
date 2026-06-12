# CalendAI

An intelligent, memory-driven calendar agent. It doesn't just book slots — it remembers
that you never take meetings before 10 AM, knows who "Alex" is, and applies your rules
to every scheduling decision without being reminded.

> **Status:** under active development. Full quickstart, architecture docs, and the
> evaluation report land with the final submission.

## What's here so far

- `src/calendai/core/` — frozen contracts: data models, `CalendarProvider` interface, clock abstraction
- `src/calendai/providers/fake.py` — deterministic in-memory calendar (frozen clock + failure injection) that powers the evaluation suite
- `src/calendai/db/` — SQLite schema + store (users, sessions, memory facts, traces)
- `src/calendai/traces/` — request tracing (every LLM call, tool call, and decision rationale)
- `tests/` — unit suite (no network, no API keys needed)

## Dev setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
copy .env.example .env           # then fill in ANTHROPIC_API_KEY
pytest
```
