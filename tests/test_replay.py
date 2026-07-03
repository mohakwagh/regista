"""Deterministic replay: the differentiator, proven.

The money assertion lives in the first test: a strict replay of a recorded
session produces the *same request-hash chain* as the original — the whole
conversation reconstructed byte-identically from the trace alone, with no
network, no tool execution, and $0 cost.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from regista import Agent, RunResult, replay, tool
from regista.errors import ConfigurationError, ReplayDivergence
from regista.policy import Allow, Deny, PermissionRequest
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.providers.replay import ReplayDivergenceWarning, diff_requests
from regista.trace.events import LlmRequest, LlmResponse
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path

CALLS_LOG: list[str] = []


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    CALLS_LOG.append(text)
    return f"echo: {text}"


def scripted_provider() -> FakeProvider:
    return FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )


async def record(tmp_path: Path, **agent_kwargs: object) -> RunResult:
    agent_kwargs.setdefault("provider", scripted_provider())
    agent = Agent(
        instructions="You are a test agent.",
        tools=[echo],
        trace_dir=tmp_path / "traces",
        **agent_kwargs,  # type: ignore[arg-type]
    )
    return await agent.run("Say hi")


def request_hashes(trace_path: Path) -> list[str]:
    return [e.request_hash for e in Trace.load(trace_path) if isinstance(e, LlmRequest)]


def tamper_instructions(trace_path: Path) -> Path:
    """Rewrite the recorded session.start with different instructions."""
    lines = trace_path.read_text().splitlines()
    start = json.loads(lines[0])
    start["instructions"] = "You are a DIFFERENT agent."
    tampered = trace_path.with_name("tampered.jsonl")
    tampered.write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")
    return tampered


def truncate_after_first_turn(trace_path: Path) -> Path:
    """Drop everything after the first turn's tool.result — a 'crashed' trace."""
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    kept = [
        e for e in events if not (e["type"] in ("llm.request", "llm.response") and e["turn"] == 2)
    ]
    kept = [e for e in kept if e["type"] != "session.end"]
    truncated = trace_path.with_name("truncated.jsonl")
    truncated.write_text("\n".join(json.dumps(e) for e in kept) + "\n")
    return truncated


# --- the money test -----------------------------------------------------------


async def test_strict_replay_reproduces_the_hash_chain(tmp_path: Path) -> None:
    original = await record(tmp_path)
    CALLS_LOG.clear()

    replayed = await replay(original.trace_path)

    # identical outcome, zero cost, zero tool execution
    assert replayed.output == original.output == "done"
    assert replayed.stop_reason == "completed"
    assert replayed.turns == original.turns
    assert replayed.usage == original.usage
    assert replayed.cost_usd == 0.0
    assert CALLS_LOG == []  # the real tool never ran

    # the replay's requests hash identically to the original's — byte-identical
    # conversation, reconstructed from the trace alone
    assert request_hashes(replayed.trace_path) == request_hashes(original.trace_path)

    replay_trace = Trace.load(replayed.trace_path)
    assert replay_trace.start.replay_of == original.session_id
    assert replay_trace.start.provider == "replay"
    responses = [e for e in replay_trace if isinstance(e, LlmResponse)]
    assert all(e.replayed for e in responses)
    assert all(e.cost_usd == 0.0 for e in responses)
    assert replay_trace.tool_results()["tu_1"].content == "echo: hi"
    # by default the replay trace lands next to the original
    assert replayed.trace_path.parent == original.trace_path.parent


async def test_denied_calls_replay_identically(tmp_path: Path) -> None:
    def deny_echo(request: PermissionRequest) -> Deny | Allow:
        return Deny(reason="echo is disabled") if request.tool_name == "echo" else Allow()

    original = await record(tmp_path, policy=deny_echo)
    replayed = await replay(original.trace_path)

    assert request_hashes(replayed.trace_path) == request_hashes(original.trace_path)
    denied = Trace.load(replayed.trace_path).tool_results()["tu_1"]
    assert denied.is_error
    assert denied.content == "Permission denied: echo is disabled"


# --- divergence modes ---------------------------------------------------------


async def test_strict_raises_on_tampered_recording(tmp_path: Path) -> None:
    original = await record(tmp_path)
    tampered = tamper_instructions(original.trace_path)

    with pytest.raises(ReplayDivergence) as excinfo:
        await replay(tampered)
    assert excinfo.value.seq >= 0
    assert "system" in excinfo.value.diff
    assert "DIFFERENT" in excinfo.value.diff


async def test_warn_mode_warns_and_serves_positionally(tmp_path: Path) -> None:
    original = await record(tmp_path)
    tampered = tamper_instructions(original.trace_path)

    with pytest.warns(ReplayDivergenceWarning, match="diverged"):
        replayed = await replay(tampered, mode="warn")
    assert replayed.output == "done"
    assert replayed.stop_reason == "completed"


async def test_hybrid_falls_through_to_live_provider(tmp_path: Path) -> None:
    original = await record(tmp_path)
    tampered = tamper_instructions(original.trace_path)
    fallback = FakeProvider(
        [tool_use_response(("tu_1", "echo", {"text": "hi"})), text_response("live now")]
    )

    replayed = await replay(tampered, mode="hybrid", fallback=fallback)

    assert replayed.output == "live now"
    # from the first divergence on, every call is live
    assert len(fallback.requests) == 2
    responses = [e for e in Trace.load(replayed.trace_path) if isinstance(e, LlmResponse)]
    assert [e.replayed for e in responses] == [False, False]


async def test_hybrid_requires_a_fallback(tmp_path: Path) -> None:
    original = await record(tmp_path)
    with pytest.raises(ConfigurationError, match="fallback"):
        await replay(original.trace_path, mode="hybrid")


# --- exhaustion and resume ----------------------------------------------------


async def test_strict_raises_when_recording_is_exhausted(tmp_path: Path) -> None:
    original = await record(tmp_path)
    truncated = truncate_after_first_turn(original.trace_path)

    with pytest.raises(ReplayDivergence, match="exhausted"):
        await replay(truncated)


async def test_hybrid_resumes_a_crashed_trace(tmp_path: Path) -> None:
    original = await record(tmp_path)
    truncated = truncate_after_first_turn(original.trace_path)
    fallback = FakeProvider([text_response("finished live")])

    resumed = await replay(truncated, mode="hybrid", fallback=fallback)

    assert resumed.output == "finished live"
    assert resumed.stop_reason == "completed"
    responses = [e for e in Trace.load(resumed.trace_path) if isinstance(e, LlmResponse)]
    assert [e.replayed for e in responses] == [True, False]  # recorded prefix, live tail


async def test_replay_needs_at_least_one_recorded_call(tmp_path: Path) -> None:
    crashed = await record(tmp_path, provider=FakeProvider([]))  # errors before any response
    with pytest.raises(ValueError, match="no recorded LLM calls"):
        await replay(crashed.trace_path)


# --- the diff -----------------------------------------------------------------


def test_diff_pinpoints_the_first_change() -> None:
    recorded = {"system": "a", "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
    live = {"system": "b", "messages": [{"role": "user", "content": [{"text": "yo"}]}]}
    diff = diff_requests(recorded, live)
    assert "request.system: recorded 'a' != live 'b'" in diff
    assert "request.messages[0].content[0].text: recorded 'hi' != live 'yo'" in diff


def test_diff_reports_structural_changes() -> None:
    assert "only in recording" in diff_requests({"a": 1}, {})
    assert "only in live request" in diff_requests({}, {"a": 1})
    assert "recorded 2 items, live has 1" in diff_requests([1, 2], [1])
    assert diff_requests({"a": 1}, {"a": 1}) == "(payloads are structurally identical)"
