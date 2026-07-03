"""AnthropicProvider wire tests: real JSON at the httpx layer, zero network.

respx intercepts the official SDK's underlying httpx client, so these tests
exercise the exact bytes regista puts on (and reads off) the wire — the class
of bug FakeProvider can't catch: request assembly, usage extraction, stop
reason and block normalization, error classification.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from regista.errors import ProviderError
from regista.providers.anthropic import AnthropicProvider
from regista.providers.base import ModelRequest, Provider
from regista.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

API_URL = "https://api.anthropic.com/v1/messages"


def provider(**kwargs: Any) -> AnthropicProvider:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("max_retries", 0)
    return AnthropicProvider("claude-sonnet-4-6", **kwargs)


def wire_response(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 12, "output_tokens": 5},
    }


def simple_request(**kwargs: Any) -> ModelRequest:
    kwargs.setdefault("model", "claude-sonnet-4-6")
    kwargs.setdefault("messages", [Message.user("hi")])
    return ModelRequest(**kwargs)


def test_satisfies_the_provider_protocol() -> None:
    p = provider()
    assert isinstance(p, Provider)
    assert p.name == "anthropic"
    assert p.model == "claude-sonnet-4-6"


@respx.mock
async def test_request_assembly_on_the_wire() -> None:
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(200, json=wire_response([{"type": "text", "text": "ok"}]))
    )
    request = simple_request(
        system="You are terse.",
        messages=[
            Message.user("run it"),
            Message(
                role="assistant",
                content=[ToolUseBlock(id="tu_1", name="shell", input={"command": "ls"})],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="tu_1", content="a.txt", is_error=False)],
            ),
        ],
        tools=[
            ToolSpec(
                name="shell",
                description="Run a command.",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}}},
                parallel_safe=True,
            )
        ],
        max_tokens=1024,
        params={"temperature": 0.0},
    )
    await provider().complete(request)

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 1024
    assert sent["temperature"] == 0.0
    # system prompt carries a prompt-cache breakpoint
    assert sent["system"] == [
        {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
    ]
    # parallel_safe is harness metadata and must not reach the wire
    assert sent["tools"] == [
        {
            "name": "shell",
            "description": "Run a command.",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }
    ]
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "run it"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "shell", "input": {"command": "ls"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "a.txt",
                    "is_error": False,
                }
            ],
        },
    ]
    assert route.calls.last.request.headers["x-api-key"] == "test-key"


@respx.mock
async def test_response_normalization_with_tool_use_and_cache_usage() -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tu_9", "name": "read_file", "input": {"path": "a"}},
                ],
                stop_reason="tool_use",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 400,
                    "cache_creation_input_tokens": 50,
                },
            ),
            headers={"request-id": "req_abc"},
        )
    )
    response = await provider().complete(simple_request())

    assert response.stop_reason == "tool_use"
    assert response.message.text() == "Let me check."
    assert response.message.tool_uses() == [
        ToolUseBlock(id="tu_9", name="read_file", input={"path": "a"})
    ]
    assert response.usage == Usage(
        input_tokens=100, output_tokens=20, cache_read_tokens=400, cache_write_tokens=50
    )
    assert response.model == "claude-sonnet-4-6"
    assert response.request_id == "req_abc"
    assert response.raw is not None
    # and raw never leaks into the trace payload
    assert "raw" not in response.model_dump_trace()


@respx.mock
async def test_thinking_signature_round_trips() -> None:
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig_123"},
                    {"type": "text", "text": "answer"},
                ]
            ),
        )
    )
    response = await provider().complete(simple_request())
    thinking = response.message.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == "sig_123"

    # sending the block back preserves the signature on the wire
    await provider().complete(simple_request(messages=[Message.user("hi"), response.message]))
    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "hmm",
        "signature": "sig_123",
    }


@respx.mock
async def test_unknown_stop_reason_maps_to_other() -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                [{"type": "text", "text": "hi"}],
                stop_reason="model_context_window_exceeded",
            ),
        )
    )
    response = await provider().complete(simple_request())
    assert response.stop_reason == "other"


@respx.mock
async def test_unknown_block_types_are_dropped_not_fatal() -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                [
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "visible"},
                ]
            ),
        )
    )
    response = await provider().complete(simple_request())
    assert [type(b) for b in response.message.content] == [TextBlock]


@respx.mock
@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (529, True), (400, False), (401, False)],
)
async def test_api_errors_map_to_provider_error(status: int, retryable: bool) -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            status,
            json={"type": "error", "error": {"type": "api_error", "message": "nope"}},
        )
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider().complete(simple_request())
    assert excinfo.value.provider == "anthropic"
    assert excinfo.value.retryable is retryable
    assert str(status) in str(excinfo.value)


@respx.mock
async def test_connection_failure_is_retryable() -> None:
    respx.post(API_URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(ProviderError) as excinfo:
        await provider().complete(simple_request())
    assert excinfo.value.retryable is True
