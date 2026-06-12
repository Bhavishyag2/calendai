from __future__ import annotations

import pytest

from calendai.core.models import SpanKind
from calendai.traces.emitter import NullTraceEmitter, SQLiteTraceEmitter


def test_request_and_spans_roundtrip(store, clock):
    emitter = SQLiteTraceEmitter(store, clock=clock)
    emitter.begin_request("req_1", "u_alice", "schedule a sync with alex")

    with emitter.span("req_1", SpanKind.LLM_CALL, "agent_turn") as span:
        span.payload = {"model": "claude-sonnet-4-6", "input_tokens": 1200, "output_tokens": 80}
    with emitter.span("req_1", SpanKind.TOOL_CALL, "create_event") as span:
        span.payload = {"args": {"title": "Sync"}, "retries": 0}
        span.rationale = "User asked for a sync; slot 10:30 is free for both."

    emitter.end_request("req_1")

    spans = emitter.spans_for("req_1")
    assert [s["kind"] for s in spans] == ["llm_call", "tool_call"]
    assert spans[1]["rationale"].startswith("User asked")
    assert spans[0]["payload"]["input_tokens"] == 1200

    requests = emitter.recent_requests()
    assert requests[0]["request_id"] == "req_1"
    assert requests[0]["ended_at"] is not None


def test_span_recorded_even_when_body_raises(store, clock):
    emitter = SQLiteTraceEmitter(store, clock=clock)
    emitter.begin_request("req_2", "u_alice", "boom")
    with pytest.raises(RuntimeError), emitter.span("req_2", SpanKind.TOOL_CALL, "explode") as span:
        span.payload = {"args": {}}
        raise RuntimeError("tool blew up")
    spans = emitter.spans_for("req_2")
    assert len(spans) == 1  # failure is still observable in the trace


def test_null_emitter_is_silent(clock):
    emitter = NullTraceEmitter()
    emitter.begin_request("r", None, "x")
    with emitter.span("r", SpanKind.DECISION, "noop") as span:
        span.rationale = "discarded"
    emitter.end_request("r")
