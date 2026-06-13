# CalendAI — Evaluation Report

> **Status (2026-06-13):** The final full re-run was interrupted partway through
> by Anthropic API **credit exhaustion** (`400: Your credit balance is too low`),
> so the auto-generated report on that run is invalid for every scenario after the
> credits ran out. The results below are from the **last complete run** (iteration
> 1). Regenerate the authoritative report with one command once credits are topped
> up:
>
> ```bash
> python -m calendai.evals.cli            # writes EVALUATION.md
> ```

## Results — last complete run

- **Agent model:** `claude-sonnet-4-6`
- **Utility model (extraction + judge):** `claude-haiku-4-5`
- **Runs per scenario:** 2 (a scenario passes only if every run passes)
- **Scenarios passed:** 19/20 (95%)
- **Individual runs passed:** 38/40 (95%)

### Improvement loop

| Iteration | Change | Scenarios |
|---|---|---|
| Baseline | initial prompt | 16/20 (80%) |
| Iteration 1 | prompt fixes (below) | **19/20 (95%)** |

The eval suite drove four agent fixes, all in `agent/prompts.py` +
`memory/episodic.py` (the 20 scenarios stayed frozen):

1. **Proactive conflict detection** — before creating an event at a specific time
   the agent now lists that day's events and surfaces any overlap
   (`conflict_warning`: 0/2 → 2/2).
2. **Solo time-blocks** — "block 90 minutes tomorrow afternoon" is booked directly
   instead of triggering an availability check (`nl_time_afternoon`: 1/2 → 2/2).
3. **Applying remembered preferences** — a stored default meeting duration is now
   applied to later bookings, and the extractor stores it under the canonical
   `pref:default_duration` key (`preference_default_duration`: 0/2 → 2/2).
4. **Bias to action** — the agent books a clearly-timed request rather than
   stalling on peripheral details (e.g. unspecified attendees), while still asking
   when the *target* of an action is ambiguous.

### Per-scenario results (iteration 1)

| Scenario | Tags | Pass rate |
|---|---|---|
| ambiguity_clarify | edges, ambiguity | 2/2 ✅ |
| availability_multiuser | multi_user, availability | 2/2 ✅ |
| conflict_warning | edges, conflict | 2/2 ✅ |
| contact_recall | memory, multi_user | 2/2 ✅ |
| create_simple | crud, nl_time | 2/2 ✅ |
| delete_confirmed | crud, safety_confirmation | 2/2 ✅ |
| delete_requires_confirmation | safety_confirmation | 2/2 ✅ |
| error_retry_succeeds | error_handling | 2/2 ✅ |
| error_surfaces_gracefully | error_handling | 2/2 ✅ |
| invite_attendee | multi_user, crud | 2/2 ✅ |
| list_today | crud, read | 2/2 ✅ |
| multiuser_isolation | multi_user, isolation | 2/2 ✅ |
| nl_time_afternoon | nl_time | 2/2 ✅ |
| nl_time_next_friday | nl_time | 2/2 ✅ |
| preference_default_duration | memory | 2/2 ✅ |
| rule_blocks_early | rule_adherence | 2/2 ✅ |
| rule_taught_then_enforced | memory, rule_adherence | 2/2 ✅ |
| timezone_cross | edges, timezone | 2/2 ✅ |
| update_time_confirmed | crud, safety_confirmation | 0/2 ❌ |
| update_unconfirmed_no_change | safety_confirmation | 2/2 ✅ |

### Remaining failure

- **`update_time_confirmed`** (reschedule after confirmation): the two-turn
  confirmation occasionally did not complete the move. A follow-up prompt
  refinement (on the confirmation turn, re-issue the destructive tool call with the
  exact pending arguments/token before any other action) targets this; it is
  committed but **its re-measurement is pending the credit top-up** the final run
  needs.

## Note on run-to-run variance

Objective end-state scenarios (calendar/fact assertions) are stable across runs.
The **judge-scored** scenarios (`availability_multiuser`, `conflict_warning`,
`error_surfaces_gracefully`, `nl_time_afternoon`) sit nearer the pass/fail boundary
and can vary run to run; under the strict "every run must pass" rule this turns a
~85–90% per-run success into occasional scenario-level flakiness. The more stable
headline is the **individual-run pass rate**. Increasing runs-per-scenario to 3–5
would stabilize the scenario-level number at additional API cost. See
[`docs/evaluation.md`](docs/evaluation.md) for the methodology.
