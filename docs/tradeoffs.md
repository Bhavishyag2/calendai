# CalendAI — Design Trade-offs

The decisions that shaped CalendAI, the alternatives considered, and why each
went the way it did. Where a choice has a real cost, this says so.

## 1. Hand-rolled tool-use loop, not a framework

**Decision.** The agent is a ~150-line loop directly on the Anthropic SDK, not
LangGraph / the Agent SDK / LlamaIndex.

**Why.** Every step the agent takes is auditable line by line, and every safety
property (the confirmation gate, code-level rule enforcement, the loop guard,
retry semantics) lives in code I can test deterministically rather than in a
framework's internals. For a system whose differentiators are *memory*,
*error handling*, and *evaluation rigour*, the ability to assert exactly what
happens — and to trace it — is worth more than the convenience a framework buys.

**Cost.** I re-implemented things frameworks give free: history management,
tool dispatch, retry/backoff. That code is small and fully tested, but it is code
I own.

## 2. Raw httpx for Google, not google-api-python-client

**Decision.** `GoogleCalendarProvider` calls the six REST endpoints it needs
directly over httpx.

**Why.** It keeps the dependency tree light, and — critically — every request is
mockable with `respx`, so the entire error taxonomy (429 backoff, 5xx, 401
refresh, partial freebusy failures, malformed bodies) is proven in tests without
a network. The client library would have buried that behaviour.

**Cost.** I hand-wrote response mapping and had to defend it against malformed
bodies (a multi-round review found several shape-junk paths that could have
escaped as raw `TypeError`). The official client would have handled some of that.

## 3. Two-tier model strategy

**Decision.** A reasoning model for the agent (default `claude-sonnet-4-6`) and a
cheaper utility model for memory extraction and the eval judge (default
`claude-haiku-4-5`), both swappable by env var.

**Why.** Extraction and judging are narrow, structured tasks where a smaller,
faster, cheaper model is sufficient; the agent's planning is where intelligence
pays off. Splitting them is the latency/cost-vs-intelligence lever, and making it
an env var means the trade-off is tunable without code changes.

**Cost.** Two models to reason about and validate. Mitigated by the eval suite,
which exercises the whole stack with whichever models are configured.

## 4. Safety enforced in code, not in the prompt

**Decision.** Confirmation-before-destruction and the user's scheduling rules are
enforced at the tool layer in Python; the prompt also describes them, but the code
is the source of truth.

**Why.** Prompts can be ignored, truncated, or talked around — including by a
user's own stored "fact" attempting prompt injection. A delete that requires a
code-validated, cross-turn, action-matched consent token *cannot* be talked into
firing early. A rule stored as structured data is enforced even if the model never
sees it.

**Cost.** The deterministic consent check is conservative: it can mis-read an
unusual affirmative and ask again. That false-negative is the right failure mode
for a destructive action, but it is a real UX cost. (An LLM consent classifier was
rejected precisely because consent must be auditable and reproducible in evals.)

## 5. Toolbox owns retries; the provider attempts once

**Decision.** Retry/backoff lives in the provider-agnostic `Toolbox`. The Google
provider performs each logical call exactly once (its only second request is the
single 401 token-refresh resend).

**Why.** An earlier revision retried in *both* layers, which a review caught as a
3×3 = 9-attempt multiplication. Centralizing retry in one place the evals already
exercise (via `FakeCalendarProvider` failure injection) makes the behaviour
testable and the blast radius bounded.

**Cost / known gap.** `create_event` (POST) is not idempotent: a retry after an
*ambiguous* failure (5xx/transport where the write may have landed) could
duplicate an event. In this implementation that residual risk is documented rather
than solved; the production fix is a caller-supplied idempotency key (Google
supports a client-set event id on insert).

## 6. Episodic extraction is synchronous and best-effort

**Decision.** After every completed turn, the utility model extracts durable facts
and upserts them; any failure degrades to "no facts extracted" and is recorded in
the trace, never breaking the turn.

**Why.** The agent is told to call `save_profile_fact`, but models forget; the
extractor is the safety net that makes memory reliable. Running it synchronously
keeps the architecture simple and the next turn's prompt immediately consistent.

**Cost.** It adds one utility-model call of latency per turn. A background queue
would hide that latency at the cost of consistency and a lot more machinery —
not worth it at this scale.

## 7. Deterministic evaluation over a real model

**Decision.** Scenarios run the *real* agent against a deterministic fake calendar
and frozen clock, scoring objective end-state first, then trajectory, then an LLM
judge. A scenario passes only if every repeated run passes.

**Why.** Measuring the real agent (not the plumbing) is the point of an eval
suite; determinism comes from the world (fake provider + frozen clock + fixed
scenarios), and repeated runs surface flakiness. Objective end-state checks are
trusted over the judge, which is reserved for what objective checks cannot capture.

**Cost.** Real runs cost tokens and are non-deterministic in the model's output;
the harness logic is therefore unit-tested separately with scripted clients so
iteration is free, and the paid suite is run deliberately.

## 8. Chat is a single POST, not SSE

**Decision.** Chat is one `POST /api/chat` that returns the complete reply. An
earlier revision added a `GET /api/chat/stream` SSE endpoint; the security review
flagged it as **CSRF-able** — `SameSite=Lax` cookies ride along on a top-level
cross-site GET navigation, so an attacker link `…/api/chat/stream?message=delete…`
could trigger calendar actions. It was removed.

**Why.** A state-changing action that mutates the calendar must not be a GET. The
hand-rolled loop returns a complete reply anyway, so true token streaming would
also require refactoring the loop to yield intermediate events — real risk for
marginal demo value. A CSRF-protected POST is the honest, safe choice.

**Cost.** No live typing effect; the user waits for the turn, then sees the reply.

## 9. Web security posture and the dev-login affordance

**Decision.** Opaque server-side session tokens (HttpOnly, SameSite=Lax, Secure
under TLS, Path=/), with a 7-day server-side expiry and logout that deletes the
session row. State-changing POSTs additionally enforce a same-origin check
(Origin/Referer) as CSRF defense in depth. Security headers (CSP,
`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`) are set on every
response. A `dev-login` endpoint mints a session for any email, gated behind
`CALENDAI_DEV_LOGIN=1` *and* the fake provider.

**Why — shaped by the security gate.** The first cut had three exploitable holes
the review caught: the CSRF-able SSE GET (§8); logout that only cleared the cookie
while the session lived forever server-side (a stolen token survived logout
indefinitely); and a stored-XSS path where an LLM/user-derived memory `statement`
was injected into the sidebar via `innerHTML`. All three are fixed — server-side
invalidation + expiry, same-origin checks, and `textContent`-only rendering. The
`resolve_contact` tool also lost its global registered-users fallback, which in a
multi-tenant deployment would have leaked other users' email addresses through the
agent; contacts now resolve only from the user's own memory.

**Cost / known limits.** The CSP allows `'unsafe-inline'` because the SPA is a
single inline-script/style file; a build step that externalizes them would let it
tighten to nonces. Trace spans contain calendar metadata (titles, attendees) and
are owner-scoped but not redacted — acceptable as per-user audit data at this
scope, noted for production.

## 10. Known limitations (deliberately deferred)

A final multi-agent review pass surfaced these; each is a real observation whose
*fix* is production-scale work disproportionate to this project's current scope, so
they are documented rather than built:

- **Single process-wide lock.** One `threading.Lock` serializes all agent turns
  so the shared SQLite connection stays safe. In a multi-tenant deployment one
  user's slow turn blocks others. Production fix: a per-user lock plus SQLite WAL
  (or Postgres) and a connection pool. Acceptable for a single-process demo.
- **Shared merge-interval logic** appears in three places (fake provider, Google
  provider, `find_free_slots`). It should live in one `core` helper; the
  duplication is small and currently consistent.
- **Loop assembly** (RuleEngine + Toolbox + tracer + extractor + AgentLoop) is
  built in three call sites (`cli`, `web/runtime`, `evals/runner`) with minor
  per-context differences; `web/runtime.build_loop` is the natural single home.

Several genuine bugs the same review found *were* fixed: an unverified-email
account-takeover path in the OAuth userinfo step, an `httpx.Client` leak per web
request, a trace lookup that scanned a recency window instead of querying by
owner, an eval substring scorer that checked any turn instead of the final reply,
unbounded attendee lists, and a judge API error that crashed a run instead of
failing one check.

## 11. Process: an adversarial second reviewer on every batch

**Decision.** Each batch passed a blocking review gate by a second model (Codex)
before the next batch began; a gate could fail repeatedly until clean.

**Why — with evidence it works.** The gates caught real, exploitable issues that
the implementing model missed. The confirmation gate alone took five rounds, and
each round's blocker was a strictly smaller class than the last: architectural
flaws → affirmative-word-anywhere matching → conditional consent ("yes but move
it") → action-echo mismatch ("yes, update it" authorizing a delete) → clean. The
memory and Google-provider gates similarly surfaced interval-overlap rule
bypasses, a DST fold that hid violations, a 3×3 retry multiplication, and a
traceback that chained the bearer token. None of these were visible without an
adversarial second pass. The cost — several extra review cycles — bought
correctness on exactly the safety-critical surfaces that matter.
