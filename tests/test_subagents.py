"""Subagents: Agent.as_tool() — delegation with isolated context, linked traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from regista import Agent, replay
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path


def make_child(tmp_path: Path, provider: FakeProvider) -> Agent:
    return Agent(
        provider=provider,
        instructions="You are a research subagent.",
        trace_dir=tmp_path / "child-traces",
        max_cost_usd=0.50,  # the budget carve-out: the child polices itself
    )


def make_parent(tmp_path: Path, child: Agent, provider: FakeProvider) -> Agent:
    return Agent(
        provider=provider,
        instructions="You are an orchestrator. Delegate research.",
        tools=[
            child.as_tool(
                name="researcher",
                description="Delegate a research question to the research subagent.",
            )
        ],
        trace_dir=tmp_path / "parent-traces",
    )


async def test_parent_delegates_and_traces_are_linked(tmp_path: Path) -> None:
    child = make_child(tmp_path, FakeProvider([text_response("Messi has 8 Ballons d'Or.")]))
    parent = make_parent(
        tmp_path,
        child,
        FakeProvider(
            [
                tool_use_response(("tu_1", "researcher", {"task": "How many Ballons d'Or?"})),
                text_response("The answer is 8."),
            ]
        ),
    )

    result = await parent.run("Research the Ballon d'Or count")

    assert result.output == "The answer is 8."
    parent_trace = Trace.load(result.trace_path)
    assert parent_trace.tool_results()["tu_1"].content == "Messi has 8 Ballons d'Or."
    assert parent_trace.start.parent_session_id is None  # top-level session

    child_traces = list((tmp_path / "child-traces").glob("*.jsonl"))
    assert len(child_traces) == 1
    child_trace = Trace.load(child_traces[0])
    assert child_trace.start.parent_session_id == result.session_id
    assert child_trace.start.task == "How many Ballons d'Or?"


async def test_child_failure_is_error_data_for_the_parent(tmp_path: Path) -> None:
    child = make_child(tmp_path, FakeProvider([]))  # exhausted script → error outcome
    parent = make_parent(
        tmp_path,
        child,
        FakeProvider(
            [
                tool_use_response(("tu_1", "researcher", {"task": "anything"})),
                text_response("Could not research; answering from memory."),
            ]
        ),
    )

    result = await parent.run("Research something")

    assert result.stop_reason == "completed"  # the parent survived
    recorded = Trace.load(result.trace_path).tool_results()["tu_1"]
    assert recorded.is_error
    assert "SubagentError" in recorded.content
    assert "researcher" in recorded.content


async def test_parent_replay_is_hermetic_and_never_reruns_the_child(tmp_path: Path) -> None:
    child = make_child(tmp_path, FakeProvider([text_response("42")]))
    parent = make_parent(
        tmp_path,
        child,
        FakeProvider(
            [
                tool_use_response(("tu_1", "researcher", {"task": "the answer"})),
                text_response("done"),
            ]
        ),
    )
    result = await parent.run("Find the answer")
    child_traces_before = len(list((tmp_path / "child-traces").glob("*.jsonl")))

    replayed = await replay(result.trace_path)

    assert replayed.output == "done"
    assert replayed.cost_usd == 0.0
    # the child was not re-run: no new child trace appeared (its provider
    # script is exhausted anyway — a live re-run would have failed loudly)
    assert len(list((tmp_path / "child-traces").glob("*.jsonl"))) == child_traces_before


async def test_children_can_nest(tmp_path: Path) -> None:
    grandchild = Agent(
        provider=FakeProvider([text_response("leaf result")]),
        instructions="Leaf worker.",
        trace_dir=tmp_path / "grandchild-traces",
    )
    child = Agent(
        provider=FakeProvider(
            [
                tool_use_response(("tu_g", "leaf", {"task": "dig deeper"})),
                text_response("middle result"),
            ]
        ),
        instructions="Middle manager.",
        tools=[grandchild.as_tool(name="leaf", description="Delegate to the leaf worker.")],
        trace_dir=tmp_path / "child-traces",
    )
    parent = make_parent(
        tmp_path,
        child,
        FakeProvider(
            [
                tool_use_response(("tu_1", "researcher", {"task": "go"})),
                text_response("top result"),
            ]
        ),
    )

    result = await parent.run("Delegate all the way down")

    assert result.output == "top result"
    child_trace = Trace.load(next(iter((tmp_path / "child-traces").glob("*.jsonl"))))
    grandchild_trace = Trace.load(next(iter((tmp_path / "grandchild-traces").glob("*.jsonl"))))
    assert child_trace.start.parent_session_id == result.session_id
    assert grandchild_trace.start.parent_session_id == child_trace.start.session_id
