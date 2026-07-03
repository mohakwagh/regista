# The trace

**Durable state, verification, and observability — one artifact.** Module: `regista/trace/`.

Every session writes a JSONL trace: one JSON event per line, append-only, flushed per event
(a crash loses at most the event being written). The trace *is* the durable state: it holds
the full history, so resuming a session is just replaying its trace and continuing.

## The envelope and the events

```json
{"schema_version": 1, "session_id": "01J...", "seq": 17, "ts": "2026-07-02T18:04:11Z", "type": "tool.call", ...}
```

`seq` is the replay ordering key; `ts` is informational only. Event types:
`session.start` (task, rendered instructions, model, tool schemas, policy, context config),
`llm.request` (+ `request_hash`), `llm.response` (the full normalized response — **the
replay payload**), `tool.call`, `permission.decision`, `tool.result`,
`context.compaction`, `error`, `session.end`.

The schema is versioned; any change requires a `SCHEMA_VERSION` bump and a migration note.
Readers reject traces newer than they understand.

## Reading traces

```python
from regista.trace.reader import Trace

trace = Trace.load(result.trace_path)
trace.summary()        # session_id, task, turns, tool_calls, cost, stop_reason
trace.llm_calls()      # (LlmRequest, LlmResponse) pairs in order — the replay index
trace.tool_results()   # tool_use_id -> ToolResult, for stubbed-tool replay
```

## OpenTelemetry export

`trace/otel.py` builds a span tree from a recorded trace **post-hoc** — session → turn →
llm/tool spans, with the recorded timestamps and `gen_ai.*` attributes:

```python
from regista.trace.otel import export_trace

export_trace(Trace.load(result.trace_path))   # to the global tracer provider
```

Exporting yesterday's session (or a $0 replay of it) paints the same picture in Jaeger or
Grafana as a live run would have. One log; no second instrumentation path to drift.
Requires the `otel` extra: `pip install regista[otel]`.

## The rule

**If a behavior isn't in the trace, it's a bug.** Every subsystem writes to the trace;
replay, resume, and OTel export are consumers of it. This is the contract that makes
[deterministic replay](replay.md) possible.
