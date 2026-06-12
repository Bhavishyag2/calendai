"""A scripted Anthropic-client double for agent loop tests.

Mimics exactly the slice of the SDK surface the loop touches:
client.messages.create(...) returning an object with .content (typed blocks),
.stop_reason and .usage. Responses are consumed in order; running out of
script is a test bug and raises.
"""

from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace
from typing import Any

_ids = itertools.count(1)


def text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def tool_call(name: str, tool_input: dict[str, Any], text: str | None = None) -> SimpleNamespace:
    blocks: list[SimpleNamespace] = []
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    blocks.append(
        SimpleNamespace(type="tool_use", id=f"toolu_{next(_ids):03d}", name=name, input=tool_input)
    )
    return SimpleNamespace(
        content=blocks,
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=30),
    )


class ScriptedClient:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # so client.messages.create resolves here

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Deep-copy: the loop mutates its messages list in place across
        # iterations; tests need the point-in-time state of each call.
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):  # scripted API failure
            raise response
        return response

    def queue(self, *responses: Any) -> ScriptedClient:
        """Append responses mid-test - e.g. after extracting a real
        confirmation token from an earlier turn."""
        self._responses.extend(responses)
        return self

    def repeat_forever(self, response: SimpleNamespace) -> ScriptedClient:
        """After any already-queued responses are consumed, every further
        call returns `response` (loop-guard tests)."""
        queued = list(self._responses)

        class _Repeater(list):
            def pop(self, _index: int = 0) -> SimpleNamespace:  # type: ignore[override]
                return queued.pop(0) if queued else response

            def __bool__(self) -> bool:
                return True

        self._responses = _Repeater()
        return self
