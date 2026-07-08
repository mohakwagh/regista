"""Agent.resume: continue an interrupted session from its trace.

Resume is hybrid replay wearing work clothes: the recorded prefix replays for
$0 with tool effects served from the recording, and the run goes live — real
provider, real tools, real policy — from the first request the recording
can't answer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from regista import Agent, RunResult
from regista.policy import Deny, PermissionRequest
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.tools import tool
from regista.trace.events import LlmResponse, PermissionDecision
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path

CALLS_LOG: list[str] = []


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    CALLS_LOG.append(text)
    return f"echo: {text}"


def make_agent(tmp_path: Path, provider: FakeProvider, **kwargs: object) -> Agent:
    kwargs.setdefault("instructions", "You are a test agent.")
    return Agent(
        provider=provider,
        tools=[echo],
        trace_dir=tmp_path / "traces",
        **kwargs,  # type: ignore[arg-type]
    )


async def record(tmp_path: Path) -> RunResult:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )
    result = await make_agent(tmp_path, provider).run("Say hi")
    CALLS_LOG.clear()
    return result


def truncate_after_first_turn(trace_path: Path) -> Path:
    """Drop turn 2 and session.end — a session that crashed between turns."""
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    kept = [
        e
        for e in events
        if e["type"] != "session.end"
        and not (e["type"] in ("llm.request", "llm.response") and e["turn"] == 2)
    ]
    truncated = trace_path.with_name("truncated.jsonl")
    truncated.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    return truncated


def truncate_mid_tools(trace_path: Path) -> Path:
    """Cut right after the first llm.response — crashed while running tools."""
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    kept = []
    for event in events:
        kept.append(event)
        if event["type"] == "llm.response":
            break
    truncated = trace_path.with_name("mid_tools.jsonl")
    truncated.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    return truncated


async def test_resume_continues_a_crashed_trace(tmp_path: Path) -> None:
    original = await record(tmp_path)
    truncated = truncate_after_first_turn(original.trace_path)

    resumer = make_agent(tmp_path, FakeProvider([text_response("done after resume")]))
    resumed = await resumer.resume(truncated)

    assert resumed.output == "done after resume"
    assert resumed.stop_reason == "completed"
    assert CALLS_LOG == []  # the recorded tool call was not re-executed

    trace = Trace.load(resumed.trace_path)
    assert trace.start.replay_of == original.session_id
    responses = [e for e in trace if isinstance(e, LlmResponse)]
    assert [e.replayed for e in responses] == [True, False]  # prefix free, ending live
    assert responses[0].cost_usd == 0.0


async def test_resume_of_a_completed_trace_is_a_free_replay(tmp_path: Path) -> None:
    original = await record(tmp_path)

    resumer = make_agent(tmp_path, FakeProvider([]))  # would raise if ever consulted
    resumed = await resumer.resume(original.trace_path)

    assert resumed.output == "done"
    assert resumed.cost_usd == 0.0
    assert CALLS_LOG == []
    trace = Trace.load(resumed.trace_path)
    assert all(e.replayed for e in trace if isinstance(e, LlmResponse))


async def test_resume_reexecutes_a_tool_the_crash_cut_short(tmp_path: Path) -> None:
    original = await record(tmp_path)
    truncated = truncate_mid_tools(original.trace_path)  # tu_1 has no recorded result

    resumer = make_agent(tmp_path, FakeProvider([text_response("finished live")]))
    resumed = await resumer.resume(truncated)

    assert CALLS_LOG == ["hi"]  # the interrupted call ran for real this time
    assert resumed.output == "finished live"
    trace = Trace.load(resumed.trace_path)
    assert trace.tool_results()["tu_1"].content == "echo: hi"


async def test_recorded_calls_bypass_the_policy_but_new_calls_face_it(tmp_path: Path) -> None:
    original = await record(tmp_path)
    truncated = truncate_after_first_turn(original.trace_path)

    def deny_all(request: PermissionRequest) -> Deny:
        return Deny(reason="locked down")

    deny_all.policy_name = "deny_all"  # type: ignore[attr-defined]

    provider = FakeProvider(
        [
            tool_use_response(("tu_live", "echo", {"text": "blocked?"})),
            text_response("gave up"),
        ]
    )
    resumer = make_agent(tmp_path, provider, policy=deny_all)
    resumed = await resumer.resume(truncated)

    assert CALLS_LOG == []  # recorded tu_1 served from the trace, live tu_live denied
    assert resumed.output == "gave up"
    trace = Trace.load(resumed.trace_path)
    decisions = {e.tool_use_id: e for e in trace if isinstance(e, PermissionDecision)}
    assert decisions["tu_1"].decision == "allow"
    assert decisions["tu_1"].policy == "resume(deny_all)"
    assert decisions["tu_live"].decision == "deny"
    assert trace.tool_results()["tu_live"].is_error


async def test_a_changed_agent_diverges_and_reruns_live(tmp_path: Path) -> None:
    original = await record(tmp_path)

    provider = FakeProvider(
        [
            tool_use_response(("tu_x", "echo", {"text": "yo"})),
            text_response("live done"),
        ]
    )
    resumer = make_agent(tmp_path, provider, instructions="You are a DIFFERENT agent.")
    resumed = await resumer.resume(original.trace_path)

    assert resumed.output == "live done"
    assert CALLS_LOG == ["yo"]  # nothing was served from the recording
    trace = Trace.load(resumed.trace_path)
    assert not any(e.replayed for e in trace if isinstance(e, LlmResponse))
    assert trace.start.replay_of == original.session_id  # lineage survives divergence
