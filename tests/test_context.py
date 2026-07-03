"""Context compaction: budget-triggered, traced, and replayable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from regista import Agent, ContextConfig, replay, tool
from regista.context import COMPACTION_SYSTEM, _split_point, render_transcript
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.events import ContextCompaction, ErrorEvent, LlmRequest
from regista.trace.reader import Trace
from regista.types import Message, TextBlock, ThinkingBlock, ToolResultBlock, Usage

if TYPE_CHECKING:
    from pathlib import Path


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


def compacting_provider() -> FakeProvider:
    """Turn 1 blows the budget; the second scripted response is the summary."""
    return FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"}), usage=Usage(input_tokens=100)),
            text_response("SUMMARY: the agent echoed hi."),
            text_response("done"),
        ]
    )


def make_agent(provider: FakeProvider, tmp_path: Path, **kwargs: object) -> Agent:
    kwargs.setdefault("context", ContextConfig(max_input_tokens=50, keep_recent_messages=2))
    return Agent(
        provider=provider,
        instructions="You are a test agent.",
        tools=[echo],
        trace_dir=tmp_path / "traces",
        **kwargs,  # type: ignore[arg-type]
    )


# --- unit: transcript and split point ------------------------------------------


def test_transcript_is_deterministic_and_excludes_thinking() -> None:
    history = [
        Message.user("do the thing"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="private reasoning"),
                TextBlock(text="On it."),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="tu_1", content="x" * 3000, is_error=True)],
        ),
    ]
    transcript = render_transcript(history)
    assert transcript == render_transcript(history)  # deterministic
    assert "private reasoning" not in transcript
    assert "On it." in transcript
    assert "[tool_result error:" in transcript
    assert "... [truncated]" in transcript  # 3000-char result capped


def test_split_never_separates_a_tool_use_from_its_result() -> None:
    history = [
        Message.user("task"),
        tool_use_response(("tu_1", "echo", {"text": "a"})).message,
        Message(role="user", content=[ToolResultBlock(tool_use_id="tu_1", content="ok")]),
        Message.assistant("done with that"),
    ]
    # keep_recent=2 would split at index 2, a tool_result — must move to 1
    assert _split_point(history, keep_recent=2) == 1
    # keep_recent bigger than the history: nothing to drop
    assert _split_point(history, keep_recent=10) <= 0


# --- through the loop -----------------------------------------------------------


async def test_forced_compaction_is_traced_and_context_shrinks(tmp_path: Path) -> None:
    provider = compacting_provider()
    result = await make_agent(provider, tmp_path).run("Say hi")

    assert result.output == "done"
    assert result.stop_reason == "completed"
    assert result.turns == 2  # the compaction call is not a conversation turn
    # usage/cost include the summarization call
    assert result.usage.input_tokens == 100 + 10 + 10

    trace = Trace.load(result.trace_path)
    assert [event.type for event in trace] == [
        "session.start",
        "llm.request",
        "llm.response",
        "tool.call",
        "permission.decision",
        "tool.result",
        "llm.request",  # the compaction call
        "llm.response",
        "context.compaction",
        "llm.request",
        "llm.response",
        "session.end",
    ]
    compaction = next(e for e in trace if isinstance(e, ContextCompaction))
    assert compaction.tokens_before == 100
    assert compaction.dropped_messages == 1
    assert compaction.summary == "SUMMARY: the agent echoed hi."
    assert trace.start.context["max_input_tokens"] == 50

    # the summarizer saw its own system prompt and a transcript
    compaction_request = provider.requests[1]
    assert compaction_request.system == COMPACTION_SYSTEM
    assert compaction_request.tools == []
    assert "Summarize this agent conversation" in compaction_request.messages[0].text()

    # turn 2 ran on the compacted history: summary first, recent kept verbatim
    final_request = provider.requests[2]
    first_block = final_request.messages[0].content[0]
    assert isinstance(first_block, TextBlock)
    assert first_block.text.startswith("[Conversation so far, compacted by the harness]")
    assert "SUMMARY: the agent echoed hi." in first_block.text
    assert len(final_request.messages) == 3  # summary + kept tool_use/tool_result pair


async def test_no_compaction_when_nothing_safe_to_drop(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"}), usage=Usage(input_tokens=100)),
            text_response("done"),
        ]
    )
    agent = make_agent(
        provider, tmp_path, context=ContextConfig(max_input_tokens=50, keep_recent_messages=10)
    )
    result = await agent.run("Say hi")

    assert result.output == "done"
    trace = Trace.load(result.trace_path)
    assert not [e for e in trace if isinstance(e, ContextCompaction)]


async def test_compaction_disabled_by_default(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"}), usage=Usage(input_tokens=10**6)),
            text_response("done"),
        ]
    )
    result = await make_agent(provider, tmp_path, context=None).run("Say hi")
    assert not [e for e in Trace.load(result.trace_path) if isinstance(e, ContextCompaction)]


async def test_compaction_provider_error_ends_the_session(tmp_path: Path) -> None:
    provider = FakeProvider(
        [tool_use_response(("tu_1", "echo", {"text": "hi"}), usage=Usage(input_tokens=100))]
    )  # exhausted exactly when the summarizer is called
    result = await make_agent(provider, tmp_path).run("Say hi")

    assert result.stop_reason == "error"
    errors = [e for e in Trace.load(result.trace_path) if isinstance(e, ErrorEvent)]
    assert errors[0].phase == "context.compaction"


# --- replayability: the point of doing it this way -------------------------------


async def test_compacted_session_replays_with_identical_hashes(tmp_path: Path) -> None:
    original = await make_agent(compacting_provider(), tmp_path).run("Say hi")

    replayed = await replay(original.trace_path)

    assert replayed.output == "done"
    assert replayed.cost_usd == 0.0

    def hashes(path: Path) -> list[str]:
        return [e.request_hash for e in Trace.load(path) if isinstance(e, LlmRequest)]

    assert hashes(replayed.trace_path) == hashes(original.trace_path)
    compaction = next(
        e for e in Trace.load(replayed.trace_path) if isinstance(e, ContextCompaction)
    )
    assert compaction.summary == "SUMMARY: the agent echoed hi."
