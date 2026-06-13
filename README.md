# CalendAI

A stateful, multi-user AI agent that manages Google Calendar through natural
language. It doesn't just book slots - it remembers that you never take meetings
before 10 AM, knows who "Alex" is, applies your rules to every scheduling decision
without being reminded, and refuses to delete anything until you actually confirm.

Built around three differentiators: **long-term memory**, **robust error
handling**, and a **rigorous automated evaluation pipeline**.

## Highlights

- **Memory that persists across sessions.** Rules, contacts, and preferences are
  extracted after every turn and stored; a rule taught today is enforced *in code*
  tomorrow, even in a brand-new conversation.
- **Safety enforced in code, not the prompt.** A cross-turn confirmation gate makes
  destructive actions impossible to trigger without explicit consent; stored rules
  are enforced at the tool layer regardless of what the model decides.
- **Deterministic evaluation over the real model.** 20 frozen YAML scenarios run the
  real agent against a deterministic fake calendar, scored on objective end-state,
  tool trajectory, and an LLM judge - producing `EVALUATION.md` with success rates
  and failure analysis.
- **Full tracing.** Every request records its LLM calls (tokens, latency), tool
  calls (args, retries), and the model's own rationale - visible in the web UI.

See [`docs/architecture.md`](docs/architecture.md), [`docs/tradeoffs.md`](docs/tradeoffs.md),
and [`docs/evaluation.md`](docs/evaluation.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate on POSIX)
pip install -e ".[dev]"
copy .env.example .env                  # then fill in ANTHROPIC_API_KEY + CALENDAI_FERNET_KEY
pytest                                  # 250+ tests, no network or API key needed
```

Generate a Fernet key for token encryption:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Try it

**Developer REPL** (real model, fake calendar - no Google needed):

```bash
python -m calendai.cli --user you@example.com
# then: "book a standup tomorrow at 10am", "/facts", "/trace", "/quit"
```

**Web app** (chat UI + memory sidebar + trace viewer). Demo mode runs without
Google credentials:

```bash
CALENDAI_PROVIDER=fake CALENDAI_DEV_LOGIN=1 python -m calendai.web
# open http://127.0.0.1:8000  → dev-login with any email
```

For the real Google Calendar, create an OAuth client (Web) with redirect URI
`http://localhost:8000/auth/callback`, put `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`
in `.env`, leave `CALENDAI_PROVIDER` unset (defaults to `google`), and sign in with
"Continue with Google".

## Evaluation

```bash
python -m calendai.evals.cli                  # all scenarios -> EVALUATION.md
python -m calendai.evals.cli --filter memory  # one capability
python -m calendai.evals.cli --no-judge       # skip the LLM judge (cheaper)
```

## Layout

```
src/calendai/
  core/        frozen contracts: models, CalendarProvider interface, clock
  providers/   GoogleCalendarProvider (httpx REST) + FakeCalendarProvider
  agent/       hand-rolled tool-use loop, 8 tools, confirmation gate, retry
  memory/      profile facts, code-level rule enforcement, episodic extraction
  db/          SQLite store (users, sessions, encrypted tokens, facts, traces)
  traces/      per-request span emitter
  evals/       scenario schema, runner, scorers, report
  web/         FastAPI app: OAuth, chat, memory sidebar, trace viewer
evals/scenarios/   20 frozen YAML scenarios
docs/              architecture, trade-offs, evaluation
```

## Models

Two env-swappable tiers: a reasoning model for the agent
(`CALENDAI_AGENT_MODEL`, default `claude-sonnet-4-6`) and a cheaper utility model
for memory extraction and the eval judge (`CALENDAI_UTILITY_MODEL`, default
`claude-haiku-4-5`).
