"""Episodic memory: post-turn fact extraction with the cheap utility model.

The agent is prompted to call save_profile_fact explicitly, but models
forget. This extractor is the safety net: after every completed turn, the
utility model (Haiku tier) reads the exchange and emits durable facts as
strict JSON. Facts are validated (key pattern must agree with fact type),
deduplicated against what is already stored (identical key+value is a
no-op, so re-stating a rule doesn't churn the supersession chain), and
upserted with provenance.

Extraction must NEVER break a turn: any failure - API error, malformed
JSON, schema mismatch - degrades to "no facts extracted".
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from calendai.core.clock import Clock
from calendai.core.models import FactType, MemoryFact, User
from calendai.db.store import Store

MAX_TOKENS = 512

EXTRACTION_SYSTEM = """\
You extract durable profile facts from one exchange between a user and their
calendar assistant. Output ONLY a JSON array, no prose, no code fences.

Extract a fact only when the USER states something durable about themselves:
- a standing rule ("never book me before 10am")
- who a person is ("Alex is alex@corp.com")
- a lasting preference ("default my meetings to 25 minutes")

Do NOT extract: one-off requests, things the assistant said, calendar events
themselves, or anything speculative.

Extract the fact even if the assistant claims it already saved or noted it -
you are the system of record, the assistant's claim is not.

Each array item: {"fact_type": "rule"|"contact"|"preference",
"key": "<see patterns>", "value": {...}, "statement": "<one sentence>"}

Key patterns and required value shapes:
- rule:no_meetings_before  value {"time": "HH:MM", "timezone": "<IANA, optional>"}
- rule:no_meetings_after   value {"time": "HH:MM", "timezone": "<IANA, optional>"}
- rule:no_meetings_on      value {"days": ["saturday", ...]}
- rule:max_meeting_minutes value {"minutes": <int>}
- contact:<lowercase first name>  value {"email": "..."}
- pref:<short_snake_case_slug>    value (any JSON object)

If the exchange contains no durable fact, output exactly: []
"""

_KEY_RE = re.compile(r"^(rule|contact|pref):[a-z0-9_]+$")
_PREFIX_TO_TYPE = {"rule": "rule", "contact": "contact", "pref": "preference"}


class ExtractedFact(BaseModel):
    fact_type: Literal["rule", "contact", "preference"]
    key: str
    value: dict[str, Any]
    statement: str


class EpisodicExtractor:
    def __init__(self, client: Any, model: str, store: Store, clock: Clock) -> None:
        self.client = client
        self.model = model
        self.store = store
        self.clock = clock
        self.last_error: str | None = None  # surfaced in the memory_op trace span

    def extract(self, user: User, user_text: str, assistant_text: str) -> list[MemoryFact]:
        """Extract and persist facts from one turn. Returns what was newly saved."""
        self.last_error = None
        try:
            raw = self._call_model(user_text, assistant_text)
            candidates = self._parse(raw)
        except Exception as exc:
            # best-effort by design: the turn already succeeded; the failure
            # is still visible in the trace via last_error
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

        saved: list[MemoryFact] = []
        existing = {f.key: f for f in self.store.list_facts(user.id)}
        for cand in candidates:
            prior = existing.get(cand.key)
            if prior is not None and prior.value == cand.value:
                continue  # already known; don't churn the supersession chain
            fact = MemoryFact(
                user_id=user.id,
                fact_type=FactType(cand.fact_type),
                key=cand.key,
                value=cand.value,
                statement=cand.statement,
                provenance=f"auto-extracted from conversation on {self.clock.now().date()}",
            )
            saved.append(self.store.upsert_fact(fact))
        return saved

    # -- internals ------------------------------------------------------------

    def _call_model(self, user_text: str, assistant_text: str) -> str:
        exchange = f"USER: {user_text}\nASSISTANT: {assistant_text}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": exchange}],
        )
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts)

    @staticmethod
    def _parse(raw: str) -> list[ExtractedFact]:
        text = raw.strip()
        if text.startswith("```"):  # tolerate a fenced reply despite instructions
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        out: list[ExtractedFact] = []
        for item in data:
            try:
                fact = ExtractedFact(**item)
            except (ValidationError, TypeError):
                continue
            match = _KEY_RE.match(fact.key)
            if not match or _PREFIX_TO_TYPE[match.group(1)] != fact.fact_type:
                continue  # key pattern must agree with the declared type
            out.append(fact)
        return out
