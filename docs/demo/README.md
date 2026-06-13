# CalendAI — Demo

A scripted walkthrough driven through the **real web UI** against a **live Google
Calendar**, captured by [`scripts/demo_capture.py`](../../scripts/demo_capture.py)
(Playwright). It produces a continuous screen recording plus a screenshot per step.

- **Video:** [`video/calendai-demo.mp4`](video/) (and the source `.webm`)
- **Screenshots:** [`screenshots/`](screenshots/) — `00`–`08` below

## What it shows

| # | Step | What it demonstrates |
|---|------|----------------------|
| 00 | Signed in | Real Google OAuth session (`merrick0816@gmail.com`, IST) |
| 01 | "Book a standup tomorrow at 10am" | NL time-parsing → a real event on Google Calendar; conflict check |
| 02 | "Never schedule me before 10am" | A rule is learned and persisted to long-term memory (`rule:no_meetings_before`, shown in the Memory panel) |
| 03 | **Server process restarted** | Memory + traces survive on disk — the agent is genuinely stateful, not session-scoped |
| 04 | "Book a 9am meeting tomorrow" | Refused by the **code-enforced** rule engine (not just the prompt) + a compliant alternative proposed |
| 05 | "Cancel my dentist appointment" (none exists) | Graceful handling — asks to clarify, never invents an event |
| 06 | "Delete my standup" | Destructive action requires explicit confirmation (two-step gate) |
| 07 | "Yes, delete it" | Cross-turn consent validated in code → the event is deleted from the real calendar |
| 08 | Trace viewer | Full per-turn tool-call audit with the model's own rationale for each call |

## Why the demo runs on Sonnet

The project defaults to Haiku for cost (`CALENDAI_AGENT_MODEL` in `.env`), but the
demo is captured with `claude-sonnet-4-6`. Completing this multi-step happy path
end-to-end depends on the agent making the right tool call at each step, and
Haiku's measured limitations (see [`../../EVALUATION.md`](../../EVALUATION.md) →
"Demo capture") surface exactly here — e.g. it sometimes asks for delete
confirmation in prose without first making the call that arms the gate
(`delete_confirmed` 0/2 on Haiku). None of these are safety failures — the
code-enforced guards hold on both models — but they make an unattended Haiku
capture flaky. Sonnet completes it reliably in one run. The system is one env var
away from either model; the eval report measures both honestly.

## Reproducing

```bash
# one-time: user signs in via the browser at http://localhost:8000
CALENDAI_AGENT_MODEL=claude-sonnet-4-6 ./.venv/Scripts/python scripts/demo_capture.py
```

The script resets the demo user's memory/chat/traces and removes leftover
`Standup` events first, so each run starts from a clean slate.
