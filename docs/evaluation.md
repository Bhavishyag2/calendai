# CalendAI — Evaluation Strategy

How CalendAI is evaluated, and why this design produces a trustworthy number.
The *results* — success rates and failure analysis — live in the generated
[`EVALUATION.md`](../EVALUATION.md); this document is the methodology behind them.

## Goal

Measure the **real agent** — the actual model plus the harness around it — not a
mock of it. A test that stubs the model's decisions only proves the plumbing; the
differentiators (memory, error handling, scheduling judgment) live in how the real
model behaves inside the real loop. So evals call the live Anthropic API.

The tension that creates — real models are non-deterministic and cost tokens — is
resolved by making *everything except the model* deterministic, and by separating
the harness's own correctness (unit-tested offline, for free) from the agent's
measured performance (the paid suite).

## The deterministic world

Each scenario runs against:

- a **frozen clock** at Monday 2026-06-15 09:00 IST, so "tomorrow at 10am" always
  resolves to the same instant and end-state assertions can be exact;
- a **`FakeCalendarProvider`** — an in-memory calendar with deterministic event
  ids, Google-faithful invite mirroring, and **failure injection** (queue a
  rate-limit / 5xx / malformed response on a specific provider call);
- a fresh temporary SQLite store per run.

Because the world is fixed, any variation across runs comes from the model alone —
which is exactly what repeated runs are meant to surface.

## Scenarios (declarative, frozen)

20 YAML scenarios under `evals/scenarios/`, spanning every capability the system
offers: CRUD, natural-language time, cross-session rule adherence, memory
persistence, multi-user invites and availability, error injection, edge cases
(timezone, conflict, ambiguity), and safety-confirmation. A scenario declares its
users, seeded calendar + facts, injected failures, the user turns (across one or
more sessions), and its expectations. Scenarios were **frozen** at the evaluation
gate — the agent and prompts may change in response to failures, but the test
cases may not, so the suite cannot be gamed by editing the targets.

### Restart simulation

A scenario with more than one session simulates a process restart between them:
the calendar (external) persists, but the store is reopened and the conversation
history cleared. Any behaviour that survives the boundary — a rule taught in
session 1 blocking a booking in session 2 — must therefore come from **persisted
profile memory**, not from lingering chat context. This is what makes the memory
differentiator a genuine test rather than a within-conversation recall.

## Three scoring layers

In decreasing order of trust:

1. **End-state assertions (objective, primary).** The calendar and the stored
   facts after the run — does the expected event exist with the right
   start/end/attendee; is the rule fact persisted with the right value; is a
   forbidden event *absent*. Exact, deterministic, model-agnostic.
2. **Trajectory assertions (from traces).** Which tools were and were not called —
   e.g. a destructive change must call `update_event`; an ambiguous request must
   *not*. Read from the same trace spans the web viewer shows.
3. **LLM judge (utility model).** A yes/no rubric for what objective checks cannot
   capture — "did the assistant flag the conflict", "did it ask which meeting".
   Reserved for response *quality*; parsed strictly (first token `PASS`/`FAIL`).

A scenario is scored on whatever layers it declares; a run with no scored checks
fails rather than passing vacuously.

## Pass criteria

Each scenario runs **twice** (configurable). A scenario passes only if **every**
run passes — a flaky scenario is a failing scenario, because an assistant that
books the right meeting only half the time is not trustworthy. The headline number
is the fraction of scenarios that pass on every run; the report also breaks success
down by capability and lists the distinct failing checks for each failure.

## Harness correctness vs. agent performance

The scoring functions, scenario schema, restart logic, and report generator are
pure and **unit-tested offline** with scripted model clients (no network) —
including a full end-to-end run that exercises restart persistence and code-level
rule enforcement. Iterating on the harness is therefore free; only the final
measurement spends tokens. This separation is what lets the eval suite itself be
trusted.

## Running it

```bash
python -m calendai.evals.cli                  # full suite -> EVALUATION.md
python -m calendai.evals.cli --filter memory  # one capability
python -m calendai.evals.cli --no-judge        # objective layers only (cheaper)
```

## The improvement loop

The suite is also the development loop for agent quality: run it, read the failure
analysis, change **only** the agent or its prompts (never the frozen scenarios),
and rerun — stopping when the pass rate clears the bar or stops improving. See
[`EVALUATION.md`](../EVALUATION.md) for the outcome of that loop and the remaining
failure modes.
