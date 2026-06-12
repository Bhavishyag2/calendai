"""Profile memory: which facts enter the system prompt, and in what order.

Rules come first (hard constraints the model must see even if the list is
ever truncated), then preferences, then contacts. Deterministic ordering
keeps prompts stable across runs, which matters for evals and for a future
prompt-caching pass.
"""

from __future__ import annotations

from calendai.core.models import FactType, MemoryFact
from calendai.db.store import Store

_TYPE_PRIORITY = {FactType.RULE: 0, FactType.PREFERENCE: 1, FactType.CONTACT: 2}

MAX_PROMPT_FACTS = 30  # plenty for this product; guards prompt bloat at the tail


def profile_facts(store: Store, user_id: str, limit: int = MAX_PROMPT_FACTS) -> list[MemoryFact]:
    """Active facts for the system prompt: rules, then preferences, then contacts."""
    facts = store.list_facts(user_id)
    facts.sort(key=lambda f: (_TYPE_PRIORITY.get(f.fact_type, 9), f.key))
    return facts[:limit]
