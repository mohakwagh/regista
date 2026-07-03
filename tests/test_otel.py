"""OTel export: the span tree a recorded trace produces, via in-memory exporter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from regista import Agent, tool
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.otel import export_trace
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@tool
def broken(text: str) -> str:
    """Always fails."""
    raise ValueError("nope")


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


async def run_and_export(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter
) -> list[ReadableSpan]:
    agent = Agent(
        provider=FakeProvider(
            [
                tool_use_response(
                    ("tu_1", "echo", {"text": "hi"}), ("tu_2", "broken", {"text": "x"})
                ),
                text_response("done"),
            ]
        ),
        instructions="You are a test agent.",
        tools=[echo, broken],
        trace_dir=tmp_path,
    )
    result = await agent.run("Say hi")
    export_trace(Trace.load(result.trace_path), tracer_provider=provider)
    return list(exporter.get_finished_spans())


async def test_span_tree_shape(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    spans = await run_and_export(tmp_path, provider, exporter)
    by_name = {span.name: span for span in spans}

    session = next(span for span in spans if span.name.startswith("agent.session"))
    turn_1, turn_2 = by_name["turn 1"], by_name["turn 2"]
    assert turn_1.parent is not None and session.context is not None
    assert turn_1.parent.span_id == session.context.span_id
    assert turn_2.parent is not None
    assert turn_2.parent.span_id == session.context.span_id

    # two llm spans (one per turn), each a child of its turn
    llm_spans = [span for span in spans if span.name == "chat fake-model"]
    assert len(llm_spans) == 2
    assert turn_1.context is not None
    assert llm_spans[0].parent is not None
    assert llm_spans[0].parent.span_id == turn_1.context.span_id

    # both tool spans live under turn 1
    echo_span = by_name["execute_tool echo"]
    broken_span = by_name["execute_tool broken"]
    assert echo_span.parent is not None
    assert echo_span.parent.span_id == turn_1.context.span_id
    assert broken_span.parent is not None
    assert broken_span.parent.span_id == turn_1.context.span_id


async def test_span_attributes_and_statuses(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    spans = await run_and_export(tmp_path, provider, exporter)
    by_name = {span.name: span for span in spans}

    session = next(span for span in spans if span.name.startswith("agent.session"))
    assert session.attributes is not None
    assert session.attributes["gen_ai.request.model"] == "fake-model"
    assert session.attributes["regista.stop_reason"] == "completed"
    assert session.attributes["regista.turns"] == 2
    assert session.status.is_ok

    llm = next(span for span in spans if span.name == "chat fake-model")
    assert llm.attributes is not None
    assert llm.attributes["gen_ai.usage.input_tokens"] == 10
    assert llm.attributes["regista.replayed"] is False
    assert str(llm.attributes["regista.request_hash"])  # present and non-empty

    echo_span = by_name["execute_tool echo"]
    assert echo_span.attributes is not None
    assert echo_span.attributes["gen_ai.tool.name"] == "echo"
    assert echo_span.attributes["regista.permission.decision"] == "allow"
    assert echo_span.status.is_ok

    broken_span = by_name["execute_tool broken"]
    assert broken_span.attributes is not None
    assert broken_span.attributes["regista.tool.is_error"] is True
    assert not broken_span.status.is_ok


async def test_recorded_timestamps_drive_span_times(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    spans = await run_and_export(tmp_path, provider, exporter)
    session = next(span for span in spans if span.name.startswith("agent.session"))
    assert session.start_time is not None and session.end_time is not None
    assert session.start_time <= session.end_time
    for span in spans:
        assert span.start_time is not None and span.end_time is not None
        assert session.start_time <= span.start_time
        assert span.end_time <= session.end_time


async def test_error_session_is_marked(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    agent = Agent(
        provider=FakeProvider([]),  # exhausted immediately
        instructions="x",
        trace_dir=tmp_path,
    )
    result = await agent.run("Doomed")
    export_trace(Trace.load(result.trace_path), tracer_provider=provider)
    spans = exporter.get_finished_spans()

    session = next(span for span in spans if span.name.startswith("agent.session"))
    assert not session.status.is_ok
    error_span = next(span for span in spans if span.name.startswith("error"))
    assert error_span.attributes is not None
    assert error_span.attributes["regista.error.type"] == "ProviderError"
