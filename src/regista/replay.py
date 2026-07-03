"""``replay(trace_path)``: re-run a recorded session for $0.

The whole configuration of the original run is reconstructed from the trace —
task and instructions from session.start, tool schemas from session.start,
model/max_tokens/params from the first recorded request — and driven through
the *same* loop with two swaps: ReplayProvider instead of the real provider,
and a stub registry that serves recorded tool results by tool_use_id instead
of executing anything. Hermetic by default: no network, no tool side effects.

A replay writes its own trace, linked to the original via ``replay_of``.
Faithfulness notes (v0.1):

- permission decisions are not re-fought: denials were recorded as error
  tool_results, so the conversation replays byte-identically, but the replay
  trace's own permission.decision events all read allow/replay_stub
- a budget-stopped recording replays with max_turns pinned to the recorded
  turn count, so it stops at the same place with stop_reason "max_turns"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from regista.loop import LoopConfig
from regista.policy import Allow, PermissionDecision
from regista.providers.base import ModelRequest
from regista.providers.replay import ReplayMode, ReplayProvider
from regista.session import Session
from regista.tools.registry import Tool, ToolExecution, ToolRegistry
from regista.trace.reader import Trace
from regista.types import ToolSpec

if TYPE_CHECKING:
    from regista.policy import PermissionRequest
    from regista.providers.base import Provider
    from regista.session import RunResult
    from regista.trace.events import ToolResult


def _replay_stub_policy(request: PermissionRequest) -> PermissionDecision:
    """Every call is allowed: gating already happened in the recorded run, and
    denied calls replay through their recorded error tool_results."""
    return Allow()


_replay_stub_policy.policy_name = "replay_stub"  # type: ignore[attr-defined]


class _StubToolRegistry(ToolRegistry):
    """Serves recorded tool.result events by tool_use_id — executes nothing."""

    def __init__(self, tool_schemas: list[dict[str, Any]], results: dict[str, ToolResult]) -> None:
        def _never_called(**_: Any) -> str:
            raise AssertionError("stubbed tool must not be invoked directly")

        super().__init__(
            [Tool(_never_called, ToolSpec.model_validate(schema)) for schema in tool_schemas]
        )
        self._results = results

    async def execute(
        self, name: str, input: dict[str, Any], *, tool_use_id: str = ""
    ) -> ToolExecution:
        recorded = self._results.get(tool_use_id)
        if recorded is None:
            return ToolExecution(
                content=f"[replay] no recorded result for tool call {tool_use_id!r}",
                is_error=True,
                duration_ms=0,
            )
        return ToolExecution(
            content=recorded.content,
            is_error=recorded.is_error,
            duration_ms=recorded.duration_ms,
        )


async def replay(
    trace_path: Path | str,
    *,
    mode: ReplayMode = "strict",
    fallback: Provider | None = None,
    trace_dir: Path | str | None = None,
    max_turns: int | None = None,
) -> RunResult:
    """Re-run a recorded session; each live request is hash-verified.

    ``mode``: "strict" raises ReplayDivergence with a structural diff,
    "warn" warns and continues positionally, "hybrid" falls through to
    ``fallback`` (required) from the first divergence onward.

    The replay's own trace lands next to the original unless ``trace_dir``
    says otherwise. Returns the replay's RunResult — cost is $0 for every
    call served from the recording.
    """
    trace_path = Path(trace_path)
    trace = Trace.load(trace_path)
    start = trace.start
    calls = trace.llm_calls()
    if not calls:
        raise ValueError(f"{trace_path} contains no recorded LLM calls to replay")
    first_request = ModelRequest.model_validate(calls[0][0].request)

    if max_turns is None:
        max_turns = trace.end.turns if trace.end else 50

    config = LoopConfig(
        provider=ReplayProvider(trace, mode=mode, fallback=fallback),
        registry=_StubToolRegistry(start.tool_schemas, trace.tool_results()),
        policy=_replay_stub_policy,
        ask_handler=None,
        system=start.instructions,
        max_turns=max_turns,
        max_cost_usd=None,
        max_tokens=first_request.max_tokens,
        params=dict(first_request.params),
    )
    session = Session(
        start.task, config, trace_dir or trace_path.parent, replay_of=start.session_id
    )
    return await session.run()
