"""OpenAI-compatible Chat Completions adapter (raw httpx, non-streaming).

One adapter covers OpenAI itself and everything that speaks its dialect —
Ollama (``base_url="http://localhost:11434/v1"``), vLLM, LM Studio — which is
why it uses plain httpx rather than a vendor SDK, and why retries are handled
locally (backoff on 408/429/5xx and transport errors).

This is the lossy direction of translation (ARCHITECTURE.md §7): regista's
rich blocks flatten to OpenAI's string-content messages.

- thinking blocks are dropped on the way out (no wire equivalent)
- a tool_result's ``is_error`` flag has no wire field; the error text itself
  still reaches the model in the tool message content
- tool arguments arrive as a JSON string; if a model emits invalid JSON the
  input becomes ``{"raw_arguments": ...}`` so dispatch fails loudly as
  error-data the model can react to, instead of killing the session

Streaming arrives in step 12.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from regista.errors import ProviderError
from regista.providers.base import ModelRequest, ModelResponse
from regista.types import (
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _wire_messages(system: str | None, messages: list[Message]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    if system:
        wire.append({"role": "system", "content": system})
    for message in messages:
        # each tool result becomes its own role="tool" message
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                wire.append(
                    {"role": "tool", "tool_call_id": block.tool_use_id, "content": block.content}
                )
        text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        tool_calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": json.dumps(b.input)},
            }
            for b in message.content
            if isinstance(b, ToolUseBlock)
        ]
        if message.role == "assistant":
            if text or tool_calls:
                entry: dict[str, Any] = {"role": "assistant", "content": text or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                wire.append(entry)
        elif text:
            wire.append({"role": message.role, "content": text})
    return wire


def _parse_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_arguments": raw}
    return parsed if isinstance(parsed, dict) else {"raw_arguments": raw}


class OpenAICompatProvider:
    """``Provider`` for any Chat Completions-compatible endpoint.

    ``api_key`` is optional because local servers (Ollama, vLLM) don't need
    one. The model is chosen here, explicitly — regista never defaults one.
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
    ) -> None:
        self.model = model
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": _wire_messages(request.system, request.messages),
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
        payload.update(request.params)

        last_error = ProviderError("no attempts made", provider=self.name)
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = ProviderError(
                    f"connection to {self._client.base_url} failed: {exc}",
                    provider=self.name,
                    retryable=True,
                )
            else:
                if response.status_code == 200:
                    return self._normalize(response.json())
                retryable = response.status_code in (408, 429) or response.status_code >= 500
                last_error = ProviderError(
                    f"endpoint returned {response.status_code}: {response.text[:500]}",
                    provider=self.name,
                    retryable=retryable,
                )
                if not retryable:
                    raise last_error
            if attempt < self._max_retries:
                await asyncio.sleep(self._backoff_base_s * 2**attempt)
        raise last_error

    def _normalize(self, body: dict[str, Any]) -> ModelResponse:
        try:
            choice = body["choices"][0]
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"malformed response: no choices in {json.dumps(body)[:500]}",
                provider=self.name,
            ) from exc
        wire_message = choice.get("message") or {}

        content: list[ContentBlock] = []
        if wire_message.get("content"):
            content.append(TextBlock(text=wire_message["content"]))
        for call in wire_message.get("tool_calls") or []:
            content.append(
                ToolUseBlock(
                    id=call["id"],
                    name=call["function"]["name"],
                    input=_parse_arguments(call["function"].get("arguments") or ""),
                )
            )

        usage = body.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return ModelResponse(
            message=Message(role="assistant", content=content),
            stop_reason=_STOP_REASONS.get(choice.get("finish_reason"), "other"),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens") or 0,
                output_tokens=usage.get("completion_tokens") or 0,
                cache_read_tokens=cached,
            ),
            model=body.get("model") or self.model,
            request_id=body.get("id"),
            raw=body,
        )
