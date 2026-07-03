"""Write→read round-trip tests for the flight recorder."""

from pathlib import Path

import pytest

from regista._ids import new_ulid
from regista.trace import Trace, TraceWriter, canonical_hash
from regista.trace.events import (
    LlmRequest,
    LlmResponse,
    PermissionDecision,
    SessionEnd,
    SessionStart,
    ToolCall,
    ToolResult,
)
from regista.types import Usage


def write_sample_session(path: Path, session_id: str) -> None:
    with TraceWriter(path, session_id) as w:
        w.emit(
            SessionStart(
                task="create hello.txt",
                instructions="You are a test agent.",
                model="fake-model",
                provider="fake",
                tool_schemas=[{"name": "write_file", "input_schema": {"type": "object"}}],
                policy="allow_all",
                regista_version="0.1.0.dev0",
            )
        )
        request = {"model": "fake-model", "messages": [{"role": "user"}]}
        w.emit(LlmRequest(turn=1, request=request, request_hash=canonical_hash(request)))
        w.emit(
            LlmResponse(
                turn=1,
                response={"stop_reason": "tool_use"},
                usage=Usage(input_tokens=10, output_tokens=5),
                cost_usd=0.001,
                latency_ms=120,
            )
        )
        w.emit(ToolCall(tool_use_id="tu_1", name="write_file", input={"path": "hello.txt"}))
        w.emit(PermissionDecision(tool_use_id="tu_1", decision="allow", policy="allow_all"))
        w.emit(ToolResult(tool_use_id="tu_1", content="ok", duration_ms=3))
        w.emit(
            SessionEnd(
                stop_reason="end_turn",
                turns=1,
                usage=Usage(input_tokens=10, output_tokens=5),
                cost_usd=0.001,
                wall_time_ms=500,
                final_output="done",
            )
        )


def test_write_read_round_trip(tmp_path: Path) -> None:
    session_id = new_ulid()
    trace_path = tmp_path / f"{session_id}.jsonl"
    write_sample_session(trace_path, session_id)

    trace = Trace.load(trace_path)
    assert len(trace) == 7
    assert all(e.session_id == session_id for e in trace)
    assert [e.seq for e in trace] == list(range(7))
    assert all(e.ts.endswith("Z") for e in trace)


def test_summary_and_indexes(tmp_path: Path) -> None:
    trace_path = tmp_path / "t.jsonl"
    write_sample_session(trace_path, new_ulid())
    trace = Trace.load(trace_path)

    summary = trace.summary()
    assert summary.task == "create hello.txt"
    assert summary.turns == 1
    assert summary.tool_calls == 1
    assert summary.stop_reason == "end_turn"
    assert summary.replay_of is None

    calls = trace.llm_calls()
    assert len(calls) == 1
    request, response = calls[0]
    assert request.request_hash == canonical_hash(request.request)
    assert response.response == {"stop_reason": "tool_use"}
    assert trace.tool_results()["tu_1"].content == "ok"


def test_crashed_session_has_no_end(tmp_path: Path) -> None:
    trace_path = tmp_path / "crash.jsonl"
    with TraceWriter(trace_path, new_ulid()) as w:
        w.emit(
            SessionStart(
                task="t",
                instructions="i",
                model="m",
                provider="p",
                tool_schemas=[],
                policy="allow_all",
                regista_version="0",
            )
        )
    trace = Trace.load(trace_path)
    assert trace.end is None
    assert trace.summary().stop_reason is None


def test_newer_schema_version_rejected(tmp_path: Path) -> None:
    trace_path = tmp_path / "future.jsonl"
    line = (
        '{"schema_version": 999, "session_id": "s", "seq": 0, "ts": "t",'
        ' "type": "tool.call", "tool_use_id": "x", "name": "n", "input": {}}'
    )
    trace_path.write_text(line + "\n")
    with pytest.raises(ValueError, match="upgrade regista"):
        Trace.load(trace_path)


def test_writer_rejects_emit_after_close(tmp_path: Path) -> None:
    writer = TraceWriter(tmp_path / "x.jsonl", "s")
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.emit(ToolCall(tool_use_id="a", name="b", input={}))


def test_canonical_hash_is_order_insensitive() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_ulids_sort_chronologically() -> None:
    ids = [new_ulid() for _ in range(50)]
    assert all(len(u) == 26 for u in ids)
    # same-millisecond ties can order arbitrarily; the timestamp prefix must be sorted
    assert [u[:10] for u in ids] == sorted(u[:10] for u in ids)
