"""OpenTelemetry export: one more consumer of the one trace.

Post-hoc by design — spans are built from recorded events with their recorded
timestamps, so exporting yesterday's session (or a $0 replay of it) paints the
same picture in Jaeger/Grafana as a live run would have. There is no second
instrumentation path to drift out of sync with the JSONL log.

Span hierarchy: session → turn → llm / tool. LLM spans carry ``gen_ai.*``
semantic-convention attributes plus regista extras (request_hash, cost,
replayed). Requires the ``otel`` extra: ``pip install "regista-harness[otel]"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from regista.errors import ConfigurationError
from regista.trace.events import (
    ContextCompaction,
    ErrorEvent,
    LlmRequest,
    LlmResponse,
    PermissionDecision,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from regista.trace.reader import Trace


def _otel() -> Any:
    try:
        from opentelemetry import trace as otel_trace
    except ImportError as exc:  # pragma: no cover — dev env always has it
        raise ConfigurationError(
            'OpenTelemetry export needs the otel extra: pip install "regista-harness[otel]"'
        ) from exc
    return otel_trace


def _ns(ts: str) -> int:
    """ISO-8601 event timestamp → epoch nanoseconds."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


def export_trace(trace: Trace, *, tracer_provider: Any = None) -> None:
    """Emit one recorded session as a span tree.

    Spans go to ``tracer_provider`` (or the global one), so any configured
    exporter — OTLP, Jaeger, console — receives them.
    """
    otel_trace = _otel()
    provider = tracer_provider or otel_trace.get_tracer_provider()
    tracer = provider.get_tracer("regista")

    start, end = trace.start, trace.end
    session_span = tracer.start_span(
        f"agent.session {start.session_id}",
        start_time=_ns(trace.events[0].ts),
        attributes={
            "regista.session_id": start.session_id,
            "regista.task": start.task,
            "regista.policy": start.policy,
            "regista.version": start.regista_version,
            "regista.replay_of": start.replay_of or "",
            "gen_ai.system": start.provider,
            "gen_ai.request.model": start.model,
        },
    )
    if end is not None:
        session_span.set_attributes(
            {
                "regista.stop_reason": end.stop_reason,
                "regista.turns": end.turns,
                "regista.cost_usd": end.cost_usd,
                "gen_ai.usage.input_tokens": end.usage.input_tokens,
                "gen_ai.usage.output_tokens": end.usage.output_tokens,
            }
        )
    if end is None or end.stop_reason == "error":
        session_span.set_status(otel_trace.StatusCode.ERROR)
    session_ctx = otel_trace.set_span_in_context(session_span)

    # group events by turn: llm events carry their turn; tool/permission/error
    # events belong to the most recent llm turn seen
    turns: dict[int, list[Any]] = {}
    current_turn = 0
    for event in trace.events:
        if isinstance(event, (LlmRequest, LlmResponse)):
            current_turn = event.turn
        span_worthy = (
            LlmRequest,
            LlmResponse,
            ToolCall,
            ToolResult,
            PermissionDecision,
            ErrorEvent,
            ContextCompaction,
        )
        if isinstance(event, span_worthy):
            turns.setdefault(current_turn, []).append(event)

    for turn_number, events in sorted(turns.items()):
        turn_start, turn_end = _ns(events[0].ts), _ns(events[-1].ts)
        turn_span = tracer.start_span(
            f"turn {turn_number}", context=session_ctx, start_time=turn_start
        )
        turn_ctx = otel_trace.set_span_in_context(turn_span)
        _export_turn_events(otel_trace, tracer, turn_ctx, events, start.model)
        turn_span.end(end_time=turn_end)

    session_span.end(end_time=_ns(trace.events[-1].ts))


def _export_turn_events(
    otel_trace: Any, tracer: Any, turn_ctx: Any, events: list[Any], model: str
) -> None:
    pending_request: LlmRequest | None = None
    calls: dict[str, ToolCall] = {}
    decisions: dict[str, PermissionDecision] = {}

    for event in events:
        if isinstance(event, LlmRequest):
            pending_request = event
        elif isinstance(event, LlmResponse) and pending_request is not None:
            span = tracer.start_span(
                f"chat {model}", context=turn_ctx, start_time=_ns(pending_request.ts)
            )
            span.set_attributes(
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": event.usage.input_tokens,
                    "gen_ai.usage.output_tokens": event.usage.output_tokens,
                    "regista.request_hash": pending_request.request_hash,
                    "regista.latency_ms": event.latency_ms,
                    "regista.replayed": event.replayed,
                    "regista.cost_usd": event.cost_usd if event.cost_usd is not None else -1.0,
                }
            )
            span.end(end_time=_ns(event.ts))
            pending_request = None
        elif isinstance(event, ToolCall):
            calls[event.tool_use_id] = event
        elif isinstance(event, PermissionDecision):
            decisions[event.tool_use_id] = event
        elif isinstance(event, ToolResult):
            call = calls.get(event.tool_use_id)
            span = tracer.start_span(
                f"execute_tool {call.name if call else event.tool_use_id}",
                context=turn_ctx,
                start_time=_ns(call.ts) if call else _ns(event.ts),
            )
            attributes: dict[str, Any] = {
                "gen_ai.tool.call.id": event.tool_use_id,
                "regista.tool.is_error": event.is_error,
                "regista.tool.duration_ms": event.duration_ms,
            }
            if call is not None:
                attributes["gen_ai.tool.name"] = call.name
            decision = decisions.get(event.tool_use_id)
            if decision is not None:
                attributes["regista.permission.decision"] = decision.decision
                if decision.resolution:
                    attributes["regista.permission.resolution"] = decision.resolution
            span.set_attributes(attributes)
            if event.is_error:
                span.set_status(otel_trace.StatusCode.ERROR)
            span.end(end_time=_ns(event.ts))
        elif isinstance(event, ContextCompaction):
            span = tracer.start_span(
                "context.compaction", context=turn_ctx, start_time=_ns(event.ts)
            )
            span.set_attributes(
                {
                    "regista.compaction.tokens_before": event.tokens_before,
                    "regista.compaction.dropped_messages": event.dropped_messages,
                }
            )
            span.end(end_time=_ns(event.ts))
        elif isinstance(event, ErrorEvent):
            span = tracer.start_span(
                f"error {event.phase}", context=turn_ctx, start_time=_ns(event.ts)
            )
            span.set_attributes(
                {"regista.error.type": event.error_type, "regista.error.message": event.message}
            )
            span.set_status(otel_trace.StatusCode.ERROR)
            span.end(end_time=_ns(event.ts))
