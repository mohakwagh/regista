"""A scripted provider for tests — public API, not a test-only hack.

Users unit-test their agents with it; contributors test loop changes with it;
~90% of regista's own suite runs on it. Zero network, zero keys, zero cost.
"""

from __future__ import annotations

from typing import Any

from regista.errors import ProviderError
from regista.providers.base import ModelRequest, ModelResponse
from regista.types import Message, TextBlock, ToolUseBlock, Usage


def text_response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    """A final assistant message ending the turn."""
    return ModelResponse(
        message=Message(role="assistant", content=[TextBlock(text=text)]),
        stop_reason="end_turn",
        usage=usage or Usage(input_tokens=10, output_tokens=10),
        model="fake-model",
    )


def tool_use_response(
    *calls: tuple[str, str, dict[str, Any]],
    text: str = "",
    usage: Usage | None = None,
) -> ModelResponse:
    """An assistant message requesting tool calls.

    Each call is ``(tool_use_id, tool_name, input)``.
    """
    blocks: list[TextBlock | ToolUseBlock] = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(ToolUseBlock(id=id_, name=name, input=input_) for id_, name, input_ in calls)
    return ModelResponse(
        message=Message(role="assistant", content=list(blocks)),
        stop_reason="tool_use",
        usage=usage or Usage(input_tokens=10, output_tokens=10),
        model="fake-model",
    )


class FakeProvider:
    """Serves a scripted sequence of responses and records every request."""

    name = "fake"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._served = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._served >= len(self._responses):
            raise ProviderError(
                f"FakeProvider script exhausted after {len(self._responses)} responses",
                provider=self.name,
            )
        response = self._responses[self._served]
        self._served += 1
        return response
