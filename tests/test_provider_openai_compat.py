"""OpenAICompatProvider wire tests: the lossy rich→flat translation, verified."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from regista.errors import ProviderError
from regista.providers.base import ModelRequest, Provider
from regista.providers.openai_compat import OpenAICompatProvider
from regista.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

API_URL = "https://api.openai.com/v1/chat/completions"


def provider(**kwargs: Any) -> OpenAICompatProvider:
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("backoff_base_s", 0.0)
    return OpenAICompatProvider("gpt-4o", **kwargs)


def wire_response(
    message: dict[str, Any],
    *,
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 5},
    }


def simple_request(**kwargs: Any) -> ModelRequest:
    kwargs.setdefault("model", "gpt-4o")
    kwargs.setdefault("messages", [Message.user("hi")])
    return ModelRequest(**kwargs)


def test_satisfies_the_provider_protocol() -> None:
    p = provider()
    assert isinstance(p, Provider)
    assert p.name == "openai_compat"
    assert p.model == "gpt-4o"


@respx.mock
async def test_request_translation_on_the_wire() -> None:
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(200, json=wire_response({"role": "assistant", "content": "ok"}))
    )
    request = simple_request(
        system="You are terse.",
        messages=[
            Message.user("run it"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="private", signature="sig"),  # must be dropped
                    TextBlock(text="Checking."),
                    ToolUseBlock(id="tu_1", name="shell", input={"command": "ls"}),
                ],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="tu_1", content="a.txt", is_error=True)],
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
        max_tokens=512,
        params={"temperature": 0.5},
    )
    await provider().complete(request)

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "gpt-4o"
    assert sent["max_tokens"] == 512
    assert sent["temperature"] == 0.5
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    assert sent["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "tu_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tu_1", "content": "a.txt"},
    ]
    assert route.calls.last.request.headers["authorization"] == "Bearer test-key"


@respx.mock
async def test_tool_call_response_normalization() -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 60},
                },
            ),
        )
    )
    response = await provider().complete(simple_request())

    assert response.stop_reason == "tool_use"
    assert response.message.tool_uses() == [
        ToolUseBlock(id="call_1", name="shell", input={"command": "ls"})
    ]
    assert response.usage == Usage(input_tokens=100, output_tokens=20, cache_read_tokens=60)
    assert response.model == "gpt-4o"
    assert response.request_id == "chatcmpl-test"


@respx.mock
@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("content_filter", "refusal"),
        ("weird_new_reason", "other"),
    ],
)
async def test_finish_reason_mapping(finish_reason: str, expected: str) -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response({"role": "assistant", "content": "hi"}, finish_reason=finish_reason),
        )
    )
    response = await provider().complete(simple_request())
    assert response.stop_reason == expected


@respx.mock
async def test_invalid_tool_arguments_become_raw_arguments() -> None:
    respx.post(API_URL).mock(
        return_value=httpx.Response(
            200,
            json=wire_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{not json"},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        )
    )
    response = await provider().complete(simple_request())
    assert response.message.tool_uses()[0].input == {"raw_arguments": "{not json"}


@respx.mock
async def test_retries_transient_failures_then_succeeds() -> None:
    route = respx.post(API_URL)
    route.side_effect = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.ConnectError("blip"),
        httpx.Response(200, json=wire_response({"role": "assistant", "content": "ok"})),
    ]
    response = await provider(max_retries=2).complete(simple_request())
    assert response.message.text() == "ok"
    assert route.call_count == 3


@respx.mock
async def test_retries_exhausted_raises_last_error() -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
    with pytest.raises(ProviderError) as excinfo:
        await provider(max_retries=1).complete(simple_request())
    assert excinfo.value.retryable is True
    assert "500" in str(excinfo.value)


@respx.mock
async def test_client_errors_do_not_retry() -> None:
    route = respx.post(API_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider(max_retries=3).complete(simple_request())
    assert excinfo.value.retryable is False
    assert route.call_count == 1


@respx.mock
async def test_ollama_style_base_url_and_no_auth_header() -> None:
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=wire_response({"role": "assistant", "content": "local"})
        )
    )
    p = OpenAICompatProvider("llama3.1", base_url="http://localhost:11434/v1", max_retries=0)
    response = await p.complete(simple_request(model="llama3.1"))
    assert response.message.text() == "local"
    assert "authorization" not in route.calls.last.request.headers
    await p.aclose()


@respx.mock
async def test_malformed_response_is_a_provider_error() -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={"object": "whatever"}))
    with pytest.raises(ProviderError, match="no choices"):
        await provider().complete(simple_request())
