"""Load and inspect recorded traces.

``Trace`` is the read-side twin of ``TraceWriter`` and the foundation replay
builds on: it indexes LLM request/response pairs in seq order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from regista.trace.events import (
    SCHEMA_VERSION,
    LlmRequest,
    LlmResponse,
    SessionEnd,
    SessionStart,
    ToolResult,
    TraceEvent,
    event_adapter,
)


@dataclass(frozen=True)
class TraceSummary:
    session_id: str
    task: str
    model: str
    events: int
    turns: int
    tool_calls: int
    cost_usd: float | None
    stop_reason: str | None
    replay_of: str | None


class Trace:
    """An in-memory view of one session's JSONL trace file."""

    def __init__(self, events: list[TraceEvent], path: Path | None = None) -> None:
        if not events:
            raise ValueError("trace contains no events")
        self.events = events
        self.path = path

    @classmethod
    def load(cls, path: Path | str) -> Trace:
        path = Path(path)
        events: list[TraceEvent] = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                event = event_adapter.validate_json(line)
                if event.schema_version > SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}:{line_no}: trace schema v{event.schema_version} is newer "
                        f"than this regista (v{SCHEMA_VERSION}) — upgrade regista to read it"
                    )
                events.append(event)
        return cls(events, path=path)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def start(self) -> SessionStart:
        first = self.events[0]
        if not isinstance(first, SessionStart):
            raise ValueError("trace does not begin with session.start")
        return first

    @property
    def end(self) -> SessionEnd | None:
        """None if the session crashed before session.end was written."""
        last = self.events[-1]
        return last if isinstance(last, SessionEnd) else None

    def llm_calls(self) -> list[tuple[LlmRequest, LlmResponse]]:
        """Request/response pairs in seq order — the replay index."""
        requests = [e for e in self.events if isinstance(e, LlmRequest)]
        responses = [e for e in self.events if isinstance(e, LlmResponse)]
        # strict=False: a crash between request and response leaves one unpaired
        # trailing request, which is expected in a crashed trace
        return list(zip(requests, responses, strict=False))

    def tool_results(self) -> dict[str, ToolResult]:
        """tool_use_id → result, for stubbed-tool replay."""
        return {e.tool_use_id: e for e in self.events if isinstance(e, ToolResult)}

    def summary(self) -> TraceSummary:
        start, end = self.start, self.end
        return TraceSummary(
            session_id=start.session_id,
            task=start.task,
            model=start.model,
            events=len(self.events),
            turns=sum(1 for e in self.events if isinstance(e, LlmResponse)),
            tool_calls=len(self.tool_results()),
            cost_usd=end.cost_usd if end else None,
            stop_reason=end.stop_reason if end else None,
            replay_of=start.replay_of,
        )
