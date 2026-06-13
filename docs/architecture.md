# CalendAI — Architecture

CalendAI is a stateful, multi-user AI agent that manages Google Calendar through
natural language. This document describes how it is put together and why.

## 1. System at a glance

```
            ┌─────────────────────────────────────────────────────────┐
   Browser  │  FastAPI web app (calendai.web)                          │
  ────────► │  OAuth login · chat (SSE) · memory sidebar · trace view  │
            └───────────────┬─────────────────────────────────────────┘
                            │  one agent turn
                            ▼
            ┌─────────────────────────────────────────────────────────┐
            │  AgentLoop (calendai.agent.loop)                         │
            │  hand-rolled Anthropic tool-use loop, fully traced       │
            └───┬───────────────┬───────────────────┬─────────────────┘
                │ tools         │ memory            │ traces
                ▼               ▼                   ▼
        ┌──────────────┐ ┌──────────────┐  ┌──────────────────┐
        │ Toolbox      │ │ memory/*     │  │ SQLiteTrace      │
        │ 8 tools,     │ │ RuleEngine,  │  │ Emitter          │
        │ confirmation │ │ profile,     │  │ llm/tool/memory  │
        │ gate, retry  │ │ episodic     │  │ spans            │
        └──────┬───────┘ └──────┬───────┘  └────────┬─────────┘
               │ CalendarProvider ABC                │
               ▼                                     ▼
   ┌───────────────────────┐                ┌──────────────────┐
   │ GoogleCalendarProvider│  ◄── seam ──►  │ SQLite Store     │
   │ FakeCalendarProvider  │                │ users, facts,    │
   └───────────────────────┘                │ sessions, tokens,│
                                            │ messages, traces │
                                            └──────────────────┘
```

The same `AgentLoop` is driven by three front-ends — the web app, a developer
REPL (`calendai.cli`), and the evaluation runner (`calendai.evals`) — so every
surface exercises identical agent behaviour.

## 2. Layers

**Core contracts (`calendai.core`).** Frozen Pydantic models and the provider
interface. `UtcDatetime` is an `Annotated[datetime, AfterValidator]` that rejects
naive datetimes and normalizes everything to UTC, so time is unambiguous
end-to-end. `CalendarProvider` is an ABC with a documented error taxonomy
(`ProviderError` → `RateLimitError`/`ServerError`/`AuthError`/`NotFoundError`/
`MalformedResponseError`). `Clock` is injectable (`SystemClock` in production,
`FrozenClock` in tests and evals).

**Provider implementations (`calendai.providers`).** `GoogleCalendarProvider`
speaks raw REST over httpx; `FakeCalendarProvider` is a deterministic in-memory
calendar with Google-faithful invite mirroring and failure injection. They are
interchangeable behind the ABC — the agent never knows which it is talking to.
This seam is what makes deterministic, offline evaluation possible.

**Agent core (`calendai.agent`).** A hand-rolled tool-use loop on the Anthropic
SDK (see [tradeoffs.md](tradeoffs.md) for why not a framework). Eight tools with
Pydantic-validated arguments; invalid arguments come back as error results the
model self-corrects from. Three safety mechanisms live in *code*, not the prompt:
the confirmation gate, the rule checker, and provider retry.

**Memory (`calendai.memory`).** Profile facts (rules, contacts, preferences)
persisted in SQLite. `RuleEngine` enforces stored rules at the tool layer;
`EpisodicExtractor` runs after each turn (on the cheap utility model) to capture
durable facts the agent forgot to save explicitly. `validation.py` guards both
write paths.

**Persistence (`calendai.db`).** One SQLite `Store`. Transactional fact
supersession (at most one active fact per `(user, key)`), cipher-enforced OAuth
token storage, sessions, messages, and traces.

**Tracing (`calendai.traces`).** Every request emits an ordered span stream —
`llm_call` (model, stop reason, token usage, latency), `tool_call` (args, ok,
error type, retries), `memory_op`, and `decision` — each carrying the model's own
one-line rationale. The web trace viewer and the eval trajectory scorer both read
these.

**Web (`calendai.web`) and Evals (`calendai.evals`).** Front-ends; see §5–6.

## 3. One agent turn

1. The loop records the user message, rebuilds recent history from the store, and
   assembles the system prompt: stable persona → the user's profile facts
   (rendered as escaped, explicitly-untrusted JSON) → any pending-confirmation
   state → current datetime last (cache-friendly ordering).
2. It calls the model with the tool schemas. While the model returns `tool_use`,
   each tool is validated and executed; results (or validation/provider errors)
   are fed back for self-correction. A 12-iteration guard prevents runaways.
3. Destructive tools (`update_event`/`delete_event`) return `confirmation_required`
   with a token on first call; the token only arms after the *next* user message
   passes a deterministic consent check (see §4).
4. On a normal stop, the reply is returned and persisted; then the episodic
   extractor runs. An exception never leaves a phantom assistant message in
   history, and a loop-guard bail honestly reports any mutations already applied.

## 4. Defense in depth (rules enforced in code, never trusted to the prompt)

- **Confirmation gate.** A destructive action issues a single-use token bound to
  the exact action + canonical args. It arms only if the user's *next* reply is an
  explicit, leading affirmative drawn from a closed consent vocabulary, with the
  echoed action matching the pending one ("yes, update it" cannot authorize a
  pending delete). Questions, conditions, declines, and unrelated replies all
  revoke it. The model cannot self-confirm within a turn.
- **Rule engine.** Stored rules (`no_meetings_before/after`, `no_meetings_on`,
  `max_meeting_minutes`) are re-read fresh on every create/update and enforced
  with interval-overlap semantics (an overnight event cannot slip past a
  start-time check) across the rule's timezone, DST-fold-safe. A violation is
  vetoed at the tool layer regardless of what the prompt or the user said.
- **Untrusted memory.** Stored facts are user-controlled, so they enter the system
  prompt as JSON-escaped data explicitly labelled untrusted, and enforcement reads
  the structured `value`, never the free-text statement.

## 5. Web app

FastAPI serving a vanilla-JS SPA. Google OAuth authorization-code flow with a
random `state` (HttpOnly cookie, constant-time compared on callback); opaque
server-side session tokens; OAuth tokens persisted only through the
Fernet-cipher-enforced store API. The trace viewer is scoped to the current
user. A `fake` provider mode plus an opt-in dev-login make the UI demoable
without Google credentials.

## 6. Evaluation pipeline

Declarative YAML scenarios define a deterministic world (frozen clock, seeded
calendar + facts, injectable failures) and user turns across one or more
sessions. A multi-session scenario simulates a restart — the calendar persists
while the store is reopened and conversation history cleared — so cross-session
behaviour must come from persisted memory. Three scoring layers: objective
end-state (calendar + facts), trajectory (tool calls from traces), and an LLM
judge for response-quality rubrics. A scenario passes only if every repeated run
passes. See [evaluation.md](evaluation.md).

## 7. Models

Two tiers, env-swappable: a reasoning model for the agent
(`CALENDAI_AGENT_MODEL`, default `claude-sonnet-4-6`) and a cheaper utility model
for memory extraction and the eval judge (`CALENDAI_UTILITY_MODEL`, default
`claude-haiku-4-5`). The split is the latency/cost-vs-intelligence lever discussed
in [tradeoffs.md](tradeoffs.md).
