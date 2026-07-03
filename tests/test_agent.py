"""The heart test: a full FakeProvider run must produce the exact expected trace.

This file exercises the composition root end to end — Agent → Session →
run_loop → registry/policy/pricing → TraceWriter — with zero network. If the
event sequence here changes, either the loop changed behavior or the trace
contract did; both deserve a deliberate diff.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from regista import Agent, tool
from regista.errors import ConfigurationError
from regista.policy import Allow, Ask, Deny, PermissionRequest
from regista.pricing import ModelPrice
from regista.providers.base import ModelResponse
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.events import ErrorEvent, PermissionDecision, ToolResult, canonical_hash
from regista.trace.reader import Trace
from regista.types import Message, ToolResultBlock, Usage

if TYPE_CHECKING:
    from pathlib import Path


@tool
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echo: {text}"


@tool(parallel_safe=True)
def shout(text: str) -> str:
    """Return the text uppercased."""
    return text.upper()


@tool(parallel_safe=True)
def whisper(text: str) -> str:
    """Return the text lowercased."""
    return text.lower()


@tool
def broken(text: str) -> str:
    """Always fails."""
    raise ValueError(f"cannot handle {text!r}")


def make_agent(provider: FakeProvider, trace_dir: Path, **kwargs: object) -> Agent:
    kwargs.setdefault("tools", [echo])
    return Agent(
        provider=provider,
        instructions="You are a test agent.",
        trace_dir=trace_dir,
        **kwargs,  # type: ignore[arg-type]
    )


# --- the heart test -----------------------------------------------------------


async def test_full_run_produces_exact_trace(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )
    agent = make_agent(provider, tmp_path)
    result = await agent.run("Say hi")

    # the result summarizes the run
    assert result.output == "done"
    assert result.stop_reason == "completed"
    assert result.turns == 2
    assert result.usage == Usage(input_tokens=20, output_tokens=20)
    assert result.cost_usd == 0.0  # fake-model has no price; regista never guesses
    assert result.error is None
    assert len(result.session_id) == 26  # ULID
    assert result.trace_path == tmp_path / f"{result.session_id}.jsonl"

    # the trace records every step, in exactly this order
    trace = Trace.load(result.trace_path)
    assert [event.type for event in trace] == [
        "session.start",
        "llm.request",
        "llm.response",
        "tool.call",
        "permission.decision",
        "tool.result",
        "llm.request",
        "llm.response",
        "session.end",
    ]

    start = trace.start
    assert start.task == "Say hi"
    assert start.instructions == "You are a test agent."
    assert start.model == "fake-model"
    assert start.provider == "fake"
    assert start.policy == "allow_all"
    assert start.replay_of is None
    assert [schema["name"] for schema in start.tool_schemas] == ["echo"]

    end = trace.end
    assert end is not None
    assert end.stop_reason == "completed"
    assert end.turns == 2
    assert end.final_output == "done"

    # every event carries a fully stamped envelope
    assert all(event.session_id == result.session_id for event in trace)
    assert [event.seq for event in trace] == list(range(len(trace)))

    # request hashes verify against their recorded payloads — the replay contract
    for request, _response in trace.llm_calls():
        assert request.request_hash == canonical_hash(request.request)

    # the tool result made it back to the model verbatim
    assert trace.tool_results()["tu_1"].content == "echo: hi"
    second_request = provider.requests[1]
    (result_block,) = second_request.messages[-1].content
    assert isinstance(result_block, ToolResultBlock)
    assert result_block == ToolResultBlock(tool_use_id="tu_1", content="echo: hi", is_error=False)


# --- harness stops ------------------------------------------------------------


async def test_max_turns_stops_the_loop(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "again"})),
            text_response("never reached"),
        ]
    )
    agent = make_agent(provider, tmp_path, max_turns=1)
    result = await agent.run("Loop forever")

    assert result.stop_reason == "max_turns"
    assert result.turns == 1
    end = Trace.load(result.trace_path).end
    assert end is not None
    assert end.stop_reason == "max_turns"


async def test_budget_stops_the_loop(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("never reached"),
        ]
    )
    # 10 input + 10 output tokens at these prices = $15.00 per turn
    agent = make_agent(
        provider,
        tmp_path,
        max_cost_usd=10.0,
        price_overrides={"fake-model": ModelPrice(500_000, 1_000_000)},
    )
    result = await agent.run("Expensive task")

    assert result.stop_reason == "budget"
    assert result.turns == 1
    assert result.cost_usd == pytest.approx(15.0)


async def test_provider_error_ends_the_session(tmp_path: Path) -> None:
    agent = make_agent(FakeProvider([]), tmp_path)  # exhausted immediately
    result = await agent.run("Doomed")

    assert result.stop_reason == "error"
    assert result.error is not None
    assert "exhausted" in result.error

    trace = Trace.load(result.trace_path)
    errors = [event for event in trace if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].error_type == "ProviderError"
    end = trace.end
    assert end is not None
    assert end.stop_reason == "error"


# --- the permission gate ------------------------------------------------------


async def test_deny_becomes_error_tool_result(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("adapted"),
        ]
    )

    def deny_echo(request: PermissionRequest) -> Deny | Allow:
        return Deny(reason="echo is disabled") if request.tool_name == "echo" else Allow()

    agent = make_agent(provider, tmp_path, policy=deny_echo)
    result = await agent.run("Try echo")

    assert result.stop_reason == "completed"  # deny is data, not an exception
    trace = Trace.load(result.trace_path)
    decision = next(event for event in trace if isinstance(event, PermissionDecision))
    assert decision.decision == "deny"
    assert decision.policy == "deny_echo"
    assert decision.reason == "echo is disabled"

    denied = trace.tool_results()["tu_1"]
    assert denied.is_error
    assert denied.content == "Permission denied: echo is disabled"
    (result_block,) = provider.requests[1].messages[-1].content
    assert isinstance(result_block, ToolResultBlock)
    assert result_block.is_error


async def test_ask_without_handler_denies(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("ok"),
        ]
    )
    agent = make_agent(provider, tmp_path, policy=lambda request: Ask(prompt="allow echo?"))
    result = await agent.run("Try echo")

    trace = Trace.load(result.trace_path)
    decision = next(event for event in trace if isinstance(event, PermissionDecision))
    assert decision.decision == "ask"
    assert decision.resolution == "denied"
    assert decision.reason == "no ask handler configured"
    assert trace.tool_results()["tu_1"].is_error


async def test_ask_handler_resolution_is_traced(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("ok"),
        ]
    )

    async def approve(request: PermissionRequest) -> bool:
        return True

    agent = make_agent(
        provider, tmp_path, policy=lambda request: Ask(prompt="allow?"), ask_handler=approve
    )
    result = await agent.run("Try echo")

    trace = Trace.load(result.trace_path)
    decision = next(event for event in trace if isinstance(event, PermissionDecision))
    assert decision.decision == "ask"
    assert decision.resolution == "allowed"
    assert trace.tool_results()["tu_1"].content == "echo: hi"


async def test_async_policy_is_awaited(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("ok"),
        ]
    )

    async def async_allow(request: PermissionRequest) -> Allow:
        return Allow()

    agent = make_agent(provider, tmp_path, policy=async_allow)
    result = await agent.run("Try echo")
    assert Trace.load(result.trace_path).tool_results()["tu_1"].content == "echo: hi"


# --- tool execution -----------------------------------------------------------


async def test_tool_exception_is_error_data(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "broken", {"text": "hi"})),
            text_response("recovered"),
        ]
    )
    agent = make_agent(provider, tmp_path, tools=[broken])
    result = await agent.run("Break something")

    assert result.stop_reason == "completed"  # the model got to adapt
    failed = Trace.load(result.trace_path).tool_results()["tu_1"]
    assert failed.is_error
    assert failed.content == "ValueError: cannot handle 'hi'"


async def test_parallel_safe_batch_runs_concurrently(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(
                ("tu_1", "shout", {"text": "Hi"}),
                ("tu_2", "whisper", {"text": "Hi"}),
            ),
            text_response("ok"),
        ]
    )
    agent = make_agent(provider, tmp_path, tools=[shout, whisper])
    result = await agent.run("Both at once")

    trace = Trace.load(result.trace_path)
    assert trace.tool_results()["tu_1"].content == "HI"
    assert trace.tool_results()["tu_2"].content == "hi"
    # the reply preserves the model's call order regardless of completion order
    blocks = provider.requests[1].messages[-1].content
    assert [b.tool_use_id for b in blocks if isinstance(b, ToolResultBlock)] == ["tu_1", "tu_2"]


async def test_mixed_batch_falls_back_to_sequential(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(
                ("tu_1", "shout", {"text": "Hi"}),
                ("tu_2", "echo", {"text": "hi"}),  # echo is not parallel_safe
            ),
            text_response("ok"),
        ]
    )
    agent = make_agent(provider, tmp_path, tools=[shout, echo])
    result = await agent.run("Mixed batch")

    trace = Trace.load(result.trace_path)
    results = [event for event in trace if isinstance(event, ToolResult)]
    assert [event.tool_use_id for event in results] == ["tu_1", "tu_2"]  # strict sequence


async def test_pause_turn_continues_the_conversation(tmp_path: Path) -> None:
    paused = ModelResponse(
        message=Message.assistant("still working"),
        stop_reason="pause_turn",
        usage=Usage(input_tokens=10, output_tokens=10),
        model="fake-model",
    )
    provider = FakeProvider([paused, text_response("done")])
    agent = make_agent(provider, tmp_path)
    result = await agent.run("Long task")

    assert result.stop_reason == "completed"
    assert result.output == "done"
    assert result.turns == 2
    # the paused assistant message was re-sent as history
    assert provider.requests[1].messages[-1].text() == "still working"


# --- the sync facade ----------------------------------------------------------


def test_run_sync_outside_event_loop(tmp_path: Path) -> None:
    agent = make_agent(FakeProvider([text_response("done")]), tmp_path)
    result = agent.run_sync("Say hi")
    assert result.output == "done"


async def test_run_sync_inside_event_loop_raises(tmp_path: Path) -> None:
    agent = make_agent(FakeProvider([text_response("done")]), tmp_path)
    with pytest.raises(ConfigurationError, match=r"await agent\.run"):
        agent.run_sync("Say hi")


def test_run_sync_error_message_is_actionable(tmp_path: Path) -> None:
    # regression guard: the error must not suggest nest_asyncio or thread tricks
    async def call_inside_loop() -> str:
        agent = make_agent(FakeProvider([]), tmp_path)
        try:
            agent.run_sync("hi")
        except ConfigurationError as exc:
            return str(exc)
        raise AssertionError("expected ConfigurationError")

    message = asyncio.run(call_inside_loop())
    assert "running event loop" in message


# --- configuration validation -------------------------------------------------


def test_max_turns_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="max_turns"):
        make_agent(FakeProvider([]), tmp_path, max_turns=0)
