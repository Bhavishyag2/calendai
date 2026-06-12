"""The agent core: a hand-rolled tool-use loop on the Anthropic SDK.

Deliberately not a framework. The loop is small enough to audit line by
line, every step is traced (LLM calls with token usage, tool calls with
args/result/retries, the model's own per-call rationale), and failure
handling is explicit:
- invalid tool arguments are fed back as error results for self-correction;
- retryable provider errors are retried with backoff inside the Toolbox;
- a hard iteration guard prevents runaway loops;
- destructive operations pass through the code-enforced confirmation gate.
"""

from __future__ import annotations

import json
import time
from typing import Any

from calendai.agent.prompts import build_system_prompt
from calendai.agent.tools import Toolbox, anthropic_tool_schemas, execute_tool
from calendai.core.clock import Clock
from calendai.core.models import SpanKind, ToolOutcome, User
from calendai.db.store import Store
from calendai.traces.emitter import SQLiteTraceEmitter, new_request_id

MAX_ITERATIONS = 12
MAX_TOKENS = 1024
HISTORY_LIMIT = 20

BAIL_MESSAGE = (
    "I wasn't able to finish that request - I hit my internal step limit. "
    "Nothing was changed. Could you rephrase or split the request?"
)


class AgentLoop:
    def __init__(
        self,
        client: Any,  # anthropic.Anthropic or a scripted test double
        model: str,
        toolbox: Toolbox,
        store: Store,
        tracer: SQLiteTraceEmitter,
        clock: Clock,
        user: User,
    ) -> None:
        self.client = client
        self.model = model
        self.toolbox = toolbox
        self.store = store
        self.tracer = tracer
        self.clock = clock
        self.user = user
        self.last_request_id: str | None = None

    # -- public API --------------------------------------------------------

    def run_turn(self, user_text: str) -> str:
        """Process one user message and return the assistant's reply."""
        request_id = new_request_id(self.clock)
        self.last_request_id = request_id
        self.tracer.begin_request(request_id, self.user.id, user_text)
        # Evaluates consent for any pending confirmation against the user's
        # actual words, BEFORE the prompt is built below.
        self.toolbox.new_turn(user_text)

        self.store.add_message(self.user.id, "user", user_text)
        messages: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]}
            for m in self.store.recent_messages(self.user.id, limit=HISTORY_LIMIT)
        ]

        system = build_system_prompt(
            self.user,
            self.clock,
            self.store.list_facts(self.user.id),
            confirmation_context=self.toolbox.gate.prompt_context(),
        )
        tools = anthropic_tool_schemas()

        # Stays None unless the turn ends on purpose (final reply or loop-guard
        # bail). If an exception escapes, the user never saw a reply, so no
        # assistant message may enter history - but the trace is still closed.
        final_text: str | None = None
        try:
            for _ in range(MAX_ITERATIONS):
                response = self._llm_call(request_id, system, tools, messages)
                if response.stop_reason != "tool_use":
                    final_text = _text_of(response)
                    break
                messages.append({"role": "assistant", "content": response.content})
                results = self._run_tools(request_id, response)
                messages.append({"role": "user", "content": results})
            else:
                with self.tracer.span(request_id, SpanKind.DECISION, "loop_guard") as span:
                    span.rationale = f"hit MAX_ITERATIONS={MAX_ITERATIONS}; bailing gracefully"
                final_text = self._bail_message()
            return final_text
        finally:
            if final_text is not None:
                self.store.add_message(self.user.id, "assistant", final_text)
            self.tracer.end_request(request_id)

    def _bail_message(self) -> str:
        """Loop-guard bail text; honest about mutations already applied."""
        mutations = self.toolbox.mutations_this_turn
        if not mutations:
            return BAIL_MESSAGE
        return (
            "I hit my internal step limit before finishing. Heads up - these changes "
            f"were already applied: {'; '.join(mutations)}. Please review and tell me "
            "how you'd like to proceed."
        )

    # -- internals -----------------------------------------------------------

    def _llm_call(
        self,
        request_id: str,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> Any:
        with self.tracer.span(request_id, SpanKind.LLM_CALL, "agent_llm") as span:
            started = time.monotonic()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
            )
            span.payload = {
                "model": self.model,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        return response

    def _run_tools(self, request_id: str, response: Any) -> list[dict[str, Any]]:
        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            raw_input = dict(block.input)
            rationale = raw_input.get("rationale")
            with self.tracer.span(request_id, SpanKind.TOOL_CALL, block.name) as span:
                outcome = execute_tool(self.toolbox, block.name, raw_input)
                span.payload = {
                    "args": {k: v for k, v in raw_input.items() if k != "rationale"},
                    "ok": outcome.ok,
                    "error_type": outcome.error_type,
                    "retries": self.toolbox.last_retries,
                }
                span.rationale = rationale
            results.append(_tool_result_block(block.id, outcome))
        return results


def _text_of(response: Any) -> str:
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip() or "(no response)"


def _tool_result_block(tool_use_id: str, outcome: ToolOutcome) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(outcome.model_dump(mode="json"), sort_keys=True),
        "is_error": not outcome.ok,
    }
