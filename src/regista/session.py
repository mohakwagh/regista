"""One run of one agent: identity, trace lifecycle, and the RunResult.

A Session owns exactly the bookkeeping the loop doesn't: it mints the session
id (a ULID, so trace files sort chronologically), opens the trace, brackets the
loop with session.start / session.end events, and folds the LoopOutcome into a
RunResult. Keeping this apart from loop.py is what makes replay cheap — a
replayed session is a new Session running the same loop with the provider
swapped, linked to the original via ``replay_of``.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import regista.trace.events as ev
from regista._ids import new_ulid
from regista._version import __version__
from regista.loop import run_loop
from regista.policy import policy_name
from regista.trace.writer import TraceWriter

if TYPE_CHECKING:
    from collections.abc import Callable

    from regista.loop import LoopConfig, StopReason
    from regista.streaming import StreamEvent
    from regista.types import Usage


# The session id of the run currently executing in this async context.
# A subagent's Session is constructed inside the parent's tool dispatch, so it
# sees the parent's id here — that one read is the whole trace-linkage story.
_current_session_id: ContextVar[str | None] = ContextVar("regista_session_id", default=None)


@dataclass(frozen=True)
class RunResult:
    """What ``await agent.run(task)`` returns. ``trace_path`` is the receipt:
    every detail this summary omits is in the trace."""

    session_id: str
    output: str
    stop_reason: StopReason
    usage: Usage
    cost_usd: float
    turns: int
    trace_path: Path
    error: str | None = None


class Session:
    """A single traced run. Construct, ``await run()`` once, discard."""

    def __init__(
        self,
        task: str,
        config: LoopConfig,
        trace_dir: Path | str,
        replay_of: str | None = None,
        skills: tuple[str, ...] = (),
    ) -> None:
        self.task = task
        self.config = config
        self.replay_of = replay_of
        self.skills = skills
        self.session_id = new_ulid()
        self.parent_session_id = _current_session_id.get()
        self.trace_path = Path(trace_dir) / f"{self.session_id}.jsonl"

    async def run(self, on_event: Callable[[StreamEvent], None] | None = None) -> RunResult:
        started = time.monotonic()
        with TraceWriter(self.trace_path, self.session_id) as writer:
            writer.emit(
                ev.SessionStart(
                    task=self.task,
                    instructions=self.config.system,
                    model=self.config.provider.model,
                    provider=self.config.provider.name,
                    tool_schemas=[
                        spec.model_dump(mode="json") for spec in self.config.registry.specs()
                    ],
                    policy=policy_name(self.config.policy),
                    context=self.config.context.model_dump(mode="json"),
                    regista_version=__version__,
                    replay_of=self.replay_of,
                    skills=list(self.skills),
                    parent_session_id=self.parent_session_id,
                )
            )
            token = _current_session_id.set(self.session_id)
            try:
                outcome = await run_loop(self.task, self.config, writer, on_event)
            finally:
                _current_session_id.reset(token)
            writer.emit(
                ev.SessionEnd(
                    stop_reason=outcome.stop_reason,
                    turns=outcome.turns,
                    usage=outcome.usage,
                    cost_usd=outcome.cost_usd,
                    wall_time_ms=int((time.monotonic() - started) * 1000),
                    final_output=outcome.output,
                )
            )
        return RunResult(
            session_id=self.session_id,
            output=outcome.output,
            stop_reason=outcome.stop_reason,
            usage=outcome.usage,
            cost_usd=outcome.cost_usd,
            turns=outcome.turns,
            trace_path=self.trace_path,
            error=outcome.error,
        )
