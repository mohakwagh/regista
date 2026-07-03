"""Anthropic Messages API adapter (official SDK).

Normalization is nearly free here — regista's message vocabulary is a superset
of the Anthropic shape — so this adapter mostly maps types 1:1 and handles the
provider-specific edges:

- the system prompt is sent with a ``cache_control`` breakpoint, so repeated
  turns of a session hit the prompt cache
- cache token counts map into ``Usage`` (read → cache_read_tokens,
  creation → cache_write_tokens) and flow into cost computation
- thinking blocks keep their ``signature``, which the API requires when
  thinking content is sent back in later turns
- SDK errors become ``ProviderError`` with an honest ``retryable`` flag
  (the SDK has already retried transient failures ``max_retries`` times)

Known v0.1 limitation: ``redacted_thinking`` blocks are dropped rather than
round-tripped; enable extended thinking with tool use only once that lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, get_args

import anthropic

from regista.errors import ProviderError
from regista.providers.base import ModelRequest, ModelResponse
from regista.streaming import TextDelta, ThinkingDelta
from regista.types import (
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from regista.streaming import ProviderDelta

_KNOWN_STOP_REASONS = frozenset(get_args(StopReason))


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    # regista blocks serialize to exactly the Anthropic wire shape
    return [
        {
            "role": message.role,
            "content": [
                block.model_dump(mode="json", exclude_none=True) for block in message.content
            ],
        }
        for message in messages
    ]


def _from_wire_block(block: Any) -> ContentBlock | None:
    if block.type == "text":
        return TextBlock(text=block.text)
    if block.type == "thinking":
        return ThinkingBlock(thinking=block.thinking, signature=block.signature)
    if block.type == "tool_use":
        return ToolUseBlock(id=block.id, name=block.name, input=cast("dict[str, Any]", block.input))
    return None  # e.g. redacted_thinking — not carried in v0.1 (see module docstring)


class AnthropicProvider:
    """``Provider`` backed by the Anthropic Messages API.

    The model is chosen here, explicitly — regista never defaults a model.
    Pass ``api_key`` or rely on ``ANTHROPIC_API_KEY``; ``client`` injection
    exists for tests and custom transports.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 2,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries
        )

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": _wire_messages(request.messages),
        }
        if request.system:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if request.tools:
            # parallel_safe is harness metadata and never reaches the wire
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        kwargs.update(request.params)
        return kwargs

    def _map_error(self, exc: anthropic.APIError) -> ProviderError:
        if isinstance(exc, anthropic.APIStatusError):
            retryable = exc.status_code in (408, 429) or exc.status_code >= 500
            return ProviderError(
                f"Anthropic API returned {exc.status_code}: {exc.message}",
                provider=self.name,
                retryable=retryable,
            )
        return ProviderError(
            f"connection to Anthropic failed: {exc}", provider=self.name, retryable=True
        )

    def _normalize(self, response: Any) -> ModelResponse:
        content = [b for b in map(_from_wire_block, response.content) if b is not None]
        stop_reason: StopReason = (
            cast("StopReason", response.stop_reason)
            if response.stop_reason in _KNOWN_STOP_REASONS
            else "other"
        )
        return ModelResponse(
            message=Message(role="assistant", content=content),
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_input_tokens or 0,
                cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            ),
            model=response.model,
            request_id=getattr(response, "_request_id", None),
            raw=response.model_dump(mode="json"),
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._build_kwargs(request))
        except anthropic.APIConnectionError as exc:
            raise self._map_error(exc) from exc
        except anthropic.APIStatusError as exc:
            raise self._map_error(exc) from exc
        return self._normalize(response)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderDelta | ModelResponse]:
        try:
            async with self._client.messages.stream(**self._build_kwargs(request)) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield TextDelta(event.delta.text)
                        elif event.delta.type == "thinking_delta":
                            yield ThinkingDelta(event.delta.thinking)
                message = await stream.get_final_message()
        except anthropic.APIConnectionError as exc:
            raise self._map_error(exc) from exc
        except anthropic.APIStatusError as exc:
            raise self._map_error(exc) from exc
        yield self._normalize(message)
