"""Streaming: same trace, same hashes — you just see it sooner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import respx

from regista import (
    Agent,
    RunCompleted,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnCompleted,
    replay,
    tool,
)
from regista.providers.anthropic import AnthropicProvider
from regista.providers.base import ModelRequest, ModelResponse
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.providers.openai_compat import OpenAICompatProvider
from regista.streaming import StreamEvent, synthetic_deltas
from regista.trace.events import LlmRequest
from regista.trace.reader import Trace
from regista.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

if TYPE_CHECKING:
    from pathlib import Path


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


def scripted() -> FakeProvider:
    return FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )


def make_agent(provider: FakeProvider, tmp_path: Path) -> Agent:
    return Agent(
        provider=provider,
        instructions="You are a test agent.",
        tools=[echo],
        trace_dir=tmp_path / "traces",
    )


async def collect(agent: Agent, task: str) -> list[StreamEvent]:
    return [event async for event in agent.stream(task)]


# --- the vocabulary ------------------------------------------------------------


def test_synthetic_deltas_chunk_text_and_thinking() -> None:
    message = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="pondering"), TextBlock(text="Hello world")],
    )
    deltas = list(synthetic_deltas(message, chunk_size=6))
    assert "".join(d.thinking for d in deltas if hasattr(d, "thinking")) == "pondering"
    assert "".join(d.text for d in deltas if hasattr(d, "text")) == "Hello world"
    assert len(deltas) == 4  # 2 thinking chunks + 2 text chunks


# --- agent.stream() -------------------------------------------------------------


async def test_stream_event_sequence(tmp_path: Path) -> None:
    events = await collect(make_agent(scripted(), tmp_path), "Say hi")

    # deltas for turn 1 (none: tool_use only), then tool events, then turn 2 text
    kinds = [type(event).__name__ for event in events]
    assert kinds == [
        "TurnCompleted",
        "ToolCallStarted",
        "ToolCallFinished",
        "TextDelta",
        "TurnCompleted",
        "RunCompleted",
    ]

    started = next(e for e in events if isinstance(e, ToolCallStarted))
    assert (started.name, started.input) == ("echo", {"text": "hi"})
    finished = next(e for e in events if isinstance(e, ToolCallFinished))
    assert finished.content == "echo: hi"
    assert not finished.is_error

    turns = [e for e in events if isinstance(e, TurnCompleted)]
    assert [t.turn for t in turns] == [1, 2]
    assert turns[0].stop_reason == "tool_use"
    assert turns[1].stop_reason == "end_turn"

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "done"

    final = events[-1]
    assert isinstance(final, RunCompleted)
    assert final.result.output == "done"
    assert final.result.stop_reason == "completed"


async def test_streamed_trace_matches_blocking_trace(tmp_path: Path) -> None:
    blocking = await make_agent(scripted(), tmp_path).run("Say hi")
    events = await collect(make_agent(scripted(), tmp_path), "Say hi")
    streamed = next(e for e in events if isinstance(e, RunCompleted)).result

    def shape(path: Path) -> list[str]:
        return [event.type for event in Trace.load(path)]

    def hashes(path: Path) -> list[str]:
        return [e.request_hash for e in Trace.load(path) if isinstance(e, LlmRequest)]

    assert shape(streamed.trace_path) == shape(blocking.trace_path)
    assert hashes(streamed.trace_path) == hashes(blocking.trace_path)


async def test_streamed_session_replays(tmp_path: Path) -> None:
    events = await collect(make_agent(scripted(), tmp_path), "Say hi")
    streamed = next(e for e in events if isinstance(e, RunCompleted)).result
    replayed = await replay(streamed.trace_path)
    assert replayed.output == "done"
    assert replayed.cost_usd == 0.0


async def test_provider_error_surfaces_in_run_completed(tmp_path: Path) -> None:
    events = await collect(make_agent(FakeProvider([]), tmp_path), "Doomed")
    final = events[-1]
    assert isinstance(final, RunCompleted)
    assert final.result.stop_reason == "error"
    assert final.result.error is not None


async def test_early_break_does_not_hang(tmp_path: Path) -> None:
    agent = make_agent(scripted(), tmp_path)
    seen = 0
    async for _event in agent.stream("Say hi"):
        seen += 1
        break  # consumer walks away; the generator's cleanup must cancel the run
    assert seen == 1


# --- provider-native streaming ---------------------------------------------------


def sse(*events: str) -> bytes:
    return ("\n\n".join(events) + "\n\n").encode()


@respx.mock
async def test_anthropic_stream_yields_deltas_then_final() -> None:
    body = sse(
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
        '"type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],'
        '"stop_reason":null,"stop_sequence":null,'
        '"usage":{"input_tokens":10,"output_tokens":1}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"Hel"}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"lo"}}',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
        'event: message_delta\ndata: {"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn","stop_sequence":null},'
        '"usage":{"output_tokens":5}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    provider = AnthropicProvider("claude-sonnet-4-6", api_key="test-key", max_retries=0)
    request = ModelRequest(model="claude-sonnet-4-6", messages=[Message.user("hi")])

    items = [item async for item in provider.stream(request)]

    deltas = [i for i in items if isinstance(i, TextDelta)]
    assert [d.text for d in deltas] == ["Hel", "lo"]
    final = items[-1]
    assert isinstance(final, ModelResponse)
    assert final.message.text() == "Hello"
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 10
    assert final.usage.output_tokens == 5


@respx.mock
async def test_openai_compat_stream_assembles_tool_calls() -> None:
    chunks = sse(
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"content":"lo"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"shell","arguments":"{\\"com"}}]},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,'
        '"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"mand\\": \\"ls\\"}"}}]},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,'
        '"delta":{},"finish_reason":"tool_calls"}]}',
        'data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[],'
        '"usage":{"prompt_tokens":50,"completion_tokens":9}}',
        "data: [DONE]",
    )
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=chunks, headers={"content-type": "text/event-stream"}
        )
    )
    provider = OpenAICompatProvider("gpt-4o", api_key="test-key", max_retries=0)
    request = ModelRequest(model="gpt-4o", messages=[Message.user("hi")])

    items = [item async for item in provider.stream(request)]

    assert [d.text for d in items if isinstance(d, TextDelta)] == ["Hel", "lo"]
    final = items[-1]
    assert isinstance(final, ModelResponse)
    assert final.message.text() == "Hello"
    assert final.message.tool_uses() == [
        ToolUseBlock(id="call_1", name="shell", input={"command": "ls"})
    ]
    assert final.stop_reason == "tool_use"
    assert final.usage.input_tokens == 50
    assert final.usage.output_tokens == 9

    import json

    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
