"""The flight recorder: event schema, JSONL writer, trace reader.

Replay, resume, and OTel export are consumers of this module — there is no
second instrumentation path anywhere in regista.
"""

from regista.trace.events import SCHEMA_VERSION, TraceEvent, canonical_hash, event_adapter
from regista.trace.reader import Trace, TraceSummary
from regista.trace.writer import TraceWriter

__all__ = [
    "SCHEMA_VERSION",
    "Trace",
    "TraceEvent",
    "TraceSummary",
    "TraceWriter",
    "canonical_hash",
    "event_adapter",
]
