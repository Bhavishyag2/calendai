"""Request tracing: what the agent did and why.

Every user request produces one trace_request row and a series of spans -
LLM calls (with token usage and latency), tool calls (with args, result, and
retry count), memory operations, and explicit decision rationales. The
/traces endpoint renders these; the eval suite asserts on them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime as _datetime
from typing import Any, Protocol

from pydantic import BaseModel

from calendai.core.clock import Clock, SystemClock
from calendai.core.models import SpanKind
from calendai.db.store import Store


def _json_default(value: Any) -> Any:
    """Normalize known types into machine-readable JSON; reject the rest.

    Evals parse trace payloads programmatically, so accidental str()
    coercion of arbitrary objects would silently corrupt them.
    """
    if isinstance(value, _datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Exception):
        return f"{type(value).__name__}: {value}"
    raise TypeError(
        f"trace payload contains non-JSON-safe value of type {type(value).__name__}; "
        "convert it before recording"
    )


class LiveSpan:
    """Mutable handle yielded by TraceEmitter.span(); fill payload/rationale inside the block."""

    def __init__(self, kind: SpanKind, name: str) -> None:
        self.kind = kind
        self.name = name
        self.payload: dict[str, Any] = {}
        self.rationale: str | None = None


class TraceEmitter(Protocol):
    def begin_request(self, request_id: str, user_id: str | None, user_message: str) -> None: ...

    def end_request(self, request_id: str) -> None: ...

    def span(
        self, request_id: str, kind: SpanKind, name: str
    ) -> _SpanContext: ...  # pragma: no cover


_SpanContext = Iterator[LiveSpan]


class SQLiteTraceEmitter:
    def __init__(self, store: Store, clock: Clock | None = None) -> None:
        self._store = store
        self._clock = clock or SystemClock()

    def _now(self) -> str:
        return self._clock.now().astimezone(UTC).isoformat()

    def begin_request(self, request_id: str, user_id: str | None, user_message: str) -> None:
        # Plain INSERT: a duplicate request_id is a bug and must fail loudly
        # rather than silently merging spans into an older trace.
        self._store.conn.execute(
            """INSERT INTO trace_requests
               (request_id, user_id, user_message, started_at) VALUES (?, ?, ?, ?)""",
            (request_id, user_id, user_message, self._now()),
        )
        self._store.conn.commit()

    def end_request(self, request_id: str) -> None:
        self._store.conn.execute(
            "UPDATE trace_requests SET ended_at = ? WHERE request_id = ?",
            (self._now(), request_id),
        )
        self._store.conn.commit()

    @contextmanager
    def span(self, request_id: str, kind: SpanKind, name: str) -> Iterator[LiveSpan]:
        live = LiveSpan(kind, name)
        started = self._now()
        try:
            yield live
        finally:
            self._store.conn.execute(
                """INSERT INTO trace_spans
                   (request_id, kind, name, started_at, ended_at, payload, rationale)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    live.kind.value,
                    live.name,
                    started,
                    self._now(),
                    json.dumps(live.payload, sort_keys=True, default=_json_default),
                    live.rationale,
                ),
            )
            self._store.conn.commit()

    # Read side (used by /traces viewer and eval trajectory checks)

    def spans_for(self, request_id: str) -> list[dict[str, Any]]:
        rows = self._store.conn.execute(
            "SELECT * FROM trace_spans WHERE request_id = ? ORDER BY id", (request_id,)
        ).fetchall()
        return [
            {
                "id": r["id"],
                "kind": r["kind"],
                "name": r["name"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "payload": json.loads(r["payload"]),
                "rationale": r["rationale"],
            }
            for r in rows
        ]

    def recent_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._store.conn.execute(
            # rowid tie-breaker keeps ordering deterministic under FrozenClock
            "SELECT * FROM trace_requests ORDER BY started_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


class NullTraceEmitter:
    """No-op emitter for tests that don't care about tracing."""

    def begin_request(self, request_id: str, user_id: str | None, user_message: str) -> None:
        pass

    def end_request(self, request_id: str) -> None:
        pass

    @contextmanager
    def span(self, request_id: str, kind: SpanKind, name: str) -> Iterator[LiveSpan]:
        yield LiveSpan(kind, name)


def new_request_id(clock: Clock | None = None) -> str:
    ts = (clock or SystemClock()).now().strftime("%Y%m%d%H%M%S")
    import secrets

    return f"req_{ts}_{secrets.token_hex(4)}"
