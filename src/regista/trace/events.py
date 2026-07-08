"""The versioned trace event schema — regista's most important contract.

Every subsystem writes these events; replay, resume, and OTel export are all
consumers of them. The rule (ARCHITECTURE.md §2): if a behavior isn't in the
trace, it's a bug — and any schema change requires a SCHEMA_VERSION bump plus
a migration note.

LLM requests/responses are stored as plain dicts (the normalized ModelRequest/
ModelResponse dumps) rather than typed models: trace/ sits *below* providers/
in the dependency graph and must not import from it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from regista.types import Usage

SCHEMA_VERSION = 1


def canonical_hash(payload: Any) -> str:
    """SHA-256 over a canonical JSON serialization (sorted keys, no whitespace).

    This is the replay divergence detector: two requests are "the same" iff
    their canonical hashes match. Callers must exclude nondeterministic fields
    before hashing.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class _Event(BaseModel):
    """Shared envelope. ``seq`` is the replay ordering key; ``ts`` (ISO-8601
    UTC) is informational only — replay never depends on wall clocks.

    ``session_id``/``seq``/``ts`` are stamped by the TraceWriter at emit time;
    the placeholder defaults let event payloads be constructed without envelope
    boilerplate.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    session_id: str = ""
    seq: int = -1
    ts: str = ""


class SessionStart(_Event):
    """Everything needed to validate that a replay is configured like the original."""

    type: Literal["session.start"] = "session.start"
    task: str
    instructions: str
    model: str
    provider: str
    tool_schemas: list[dict[str, Any]]
    policy: str
    context: dict[str, Any] = {}
    """ContextConfig dump — replay needs it to compact at the same points."""
    regista_version: str
    replay_of: str | None = None
    skills: list[str] = []
    """Provenance only — the fragments/tools are already in instructions/tool_schemas."""
    parent_session_id: str | None = None
    """Set when this session ran as a subagent inside the named parent session."""


class LlmRequest(_Event):
    type: Literal["llm.request"] = "llm.request"
    turn: int
    request: dict[str, Any]
    request_hash: str


class LlmResponse(_Event):
    """The replay payload: the full normalized response, verbatim."""

    type: Literal["llm.response"] = "llm.response"
    turn: int
    response: dict[str, Any]
    usage: Usage
    cost_usd: float | None
    latency_ms: int
    replayed: bool = False


class ToolCall(_Event):
    type: Literal["tool.call"] = "tool.call"
    tool_use_id: str
    name: str
    input: dict[str, Any]


class ToolResult(_Event):
    type: Literal["tool.result"] = "tool.result"
    tool_use_id: str
    content: str
    is_error: bool = False
    duration_ms: int = 0


class PermissionDecision(_Event):
    """``ask`` decisions record their resolution so deny paths replay identically."""

    type: Literal["permission.decision"] = "permission.decision"
    tool_use_id: str
    decision: Literal["allow", "deny", "ask"]
    resolution: Literal["allowed", "denied"] | None = None
    policy: str = ""
    reason: str = ""


class ContextCompaction(_Event):
    """Compaction changes every subsequent request, so it must be traced."""

    type: Literal["context.compaction"] = "context.compaction"
    tokens_before: int
    tokens_after: int
    summary: str
    dropped_messages: int


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    phase: str
    error_type: str
    message: str
    retried: bool = False


class SessionEnd(_Event):
    """Summary only — replay never needs it, humans always read it first."""

    type: Literal["session.end"] = "session.end"
    stop_reason: str
    turns: int
    usage: Usage
    cost_usd: float
    wall_time_ms: int
    final_output: str = ""


TraceEvent = Annotated[
    SessionStart
    | LlmRequest
    | LlmResponse
    | ToolCall
    | ToolResult
    | PermissionDecision
    | ContextCompaction
    | ErrorEvent
    | SessionEnd,
    Field(discriminator="type"),
]

event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
