"""The turn engine — regista's heart, meant to be read top-to-bottom.

One call to run_loop() drives one session's conversation: assemble request →
provider → parse blocks → per tool_use: policy gate → dispatch → results →
repeat until the model finishes or the harness stops it. The loop owns no I/O
of its own (providers, tools, and the trace writer do); it is pure
orchestration, and every step emits a trace event.

Session bookkeeping (session.start/session.end) lives one level up, in
session.py — which is also why replay works: a replayed session runs this
exact loop with only the provider swapped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import regista.trace.events as ev
from regista.errors import RegistaError, ReplayDivergence
from regista.policy import Allow, Ask, Deny, PermissionRequest, policy_name
from regista.pricing import cost_usd
from regista.providers.base import ModelRequest
from regista.types import Message, ToolResultBlock, Usage

if TYPE_CHECKING:
    from regista.policy import AskHandler, PermissionPolicy
    from regista.pricing import ModelPrice
    from regista.providers.base import Provider
    from regista.tools import ToolRegistry
    from regista.trace.writer import TraceWriter
    from regista.types import ToolUseBlock

StopReason = Literal["completed", "max_turns", "budget", "error"]

# model-level stop reasons that mean "the conversation is over"
_FINAL_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "refusal", "max_tokens", "other"})


@dataclass(frozen=True)
class LoopConfig:
    provider: Provider
    registry: ToolRegistry
    policy: PermissionPolicy
    ask_handler: AskHandler | None
    system: str
    max_turns: int
    max_cost_usd: float | None
    max_tokens: int
    params: dict[str, object]
    price_overrides: dict[str, ModelPrice] | None = None


@dataclass(frozen=True)
class LoopOutcome:
    stop_reason: StopReason
    output: str
    usage: Usage
    cost_usd: float
    turns: int
    error: str | None = None


async def run_loop(task: str, config: LoopConfig, writer: TraceWriter) -> LoopOutcome:
    history: list[Message] = [Message.user(task)]
    total_usage = Usage()
    total_cost = 0.0
    output = ""
    turn = 0

    while True:
        if turn >= config.max_turns:
            return LoopOutcome("max_turns", output, total_usage, total_cost, turn)
        if config.max_cost_usd is not None and total_cost >= config.max_cost_usd:
            return LoopOutcome("budget", output, total_usage, total_cost, turn)
        turn += 1

        request = ModelRequest(
            model=config.provider.model,
            system=config.system,
            messages=history,
            tools=config.registry.specs(),
            max_tokens=config.max_tokens,
            params=dict(config.params),
        )
        writer.emit(
            ev.LlmRequest(
                turn=turn,
                request=request.model_dump(mode="json"),
                request_hash=request.request_hash(),
            )
        )

        started = time.monotonic()
        try:
            response = await config.provider.complete(request)
        except RegistaError as exc:
            writer.emit(
                ev.ErrorEvent(phase="llm.request", error_type=type(exc).__name__, message=str(exc))
            )
            if isinstance(exc, ReplayDivergence):
                raise  # strict replay must surface to the caller, not become error data
            return LoopOutcome("error", output, total_usage, total_cost, turn, error=str(exc))
        latency_ms = int((time.monotonic() - started) * 1000)

        if response.replayed:
            turn_cost: float | None = 0.0  # a recording costs nothing to serve
        else:
            turn_cost = cost_usd(response.model, response.usage, config.price_overrides)
        total_usage = total_usage + response.usage
        total_cost += turn_cost or 0.0
        writer.emit(
            ev.LlmResponse(
                turn=turn,
                response=response.model_dump_trace(),
                usage=response.usage,
                cost_usd=turn_cost,
                latency_ms=latency_ms,
                replayed=response.replayed,
            )
        )

        history.append(response.message)
        if text := response.message.text():
            output = text

        if response.stop_reason == "tool_use":
            results = await _run_tools(response.message.tool_uses(), turn, config, writer)
            history.append(Message(role="user", content=list(results)))
        elif response.stop_reason == "pause_turn":
            continue  # provider paused a long turn; re-send accumulated history as-is
        elif response.stop_reason in _FINAL_STOP_REASONS:
            return LoopOutcome("completed", output, total_usage, total_cost, turn)


async def _run_tools(
    tool_uses: list[ToolUseBlock],
    turn: int,
    config: LoopConfig,
    writer: TraceWriter,
) -> list[ToolResultBlock]:
    """Gate every call, then execute — concurrently iff the whole batch opted in."""
    permitted: list[ToolUseBlock] = []
    denials: dict[str, str] = {}

    for block in tool_uses:
        writer.emit(ev.ToolCall(tool_use_id=block.id, name=block.name, input=block.input))
        reason = await _gate(block, turn, config, writer)
        if reason is None:
            permitted.append(block)
        else:
            denials[block.id] = reason

    async def execute(block: ToolUseBlock) -> ToolResultBlock:
        execution = await config.registry.execute(block.name, block.input, tool_use_id=block.id)
        writer.emit(
            ev.ToolResult(
                tool_use_id=block.id,
                content=execution.content,
                is_error=execution.is_error,
                duration_ms=execution.duration_ms,
            )
        )
        return ToolResultBlock(
            tool_use_id=block.id, content=execution.content, is_error=execution.is_error
        )

    if permitted and all(config.registry.get(b.name).spec.parallel_safe for b in permitted):
        executed = list(await asyncio.gather(*(execute(b) for b in permitted)))
    else:
        executed = [await execute(b) for b in permitted]

    by_id = {r.tool_use_id: r for r in executed}
    results: list[ToolResultBlock] = []
    for block in tool_uses:  # preserve the model's call order in the reply
        if block.id in denials:
            content = f"Permission denied: {denials[block.id]}"
            writer.emit(ev.ToolResult(tool_use_id=block.id, content=content, is_error=True))
            results.append(ToolResultBlock(tool_use_id=block.id, content=content, is_error=True))
        else:
            results.append(by_id[block.id])
    return results


async def _gate(
    block: ToolUseBlock, turn: int, config: LoopConfig, writer: TraceWriter
) -> str | None:
    """Consult the policy. Returns None if allowed, else the denial reason."""
    request = PermissionRequest(
        tool_name=block.name, tool_input=block.input, tool_use_id=block.id, turn=turn
    )
    decision = config.policy(request)
    if asyncio.iscoroutine(decision):
        decision = await decision
    name = policy_name(config.policy)

    match decision:
        case Allow():
            writer.emit(ev.PermissionDecision(tool_use_id=block.id, decision="allow", policy=name))
            return None
        case Deny(reason=reason):
            writer.emit(
                ev.PermissionDecision(
                    tool_use_id=block.id, decision="deny", policy=name, reason=reason
                )
            )
            return reason
        case Ask(prompt=prompt):
            if config.ask_handler is None:
                writer.emit(
                    ev.PermissionDecision(
                        tool_use_id=block.id,
                        decision="ask",
                        resolution="denied",
                        policy=name,
                        reason="no ask handler configured",
                    )
                )
                return "no ask handler configured"
            allowed = await config.ask_handler(request)
            writer.emit(
                ev.PermissionDecision(
                    tool_use_id=block.id,
                    decision="ask",
                    resolution="allowed" if allowed else "denied",
                    policy=name,
                    reason=prompt,
                )
            )
            return None if allowed else "denied by ask handler"
    raise AssertionError(f"unreachable: {decision!r}")
