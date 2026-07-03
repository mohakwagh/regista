"""Context management: keep the conversation inside the model's budget.

The trigger is provider-reported usage (never a local token estimate): when a
turn's observed input tokens cross ``max_input_tokens``, the oldest messages
are summarized — by the session's own provider — and replaced with a single
summary message, keeping the most recent messages verbatim.

Compaction is itself traced and replayable: the summarization call is a
regular llm.request/llm.response pair (with its own request_hash), followed by
a context.compaction event. A replayed session re-runs this exact logic and is
served the recorded summary, so post-compaction request hashes still match.
That is why compaction lives behind the same provider seam as everything else.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

import regista.trace.events as ev
from regista.pricing import cost_usd
from regista.providers.base import ModelRequest
from regista.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

if TYPE_CHECKING:
    from regista.pricing import ModelPrice
    from regista.providers.base import Provider
    from regista.trace.writer import TraceWriter

COMPACTION_SYSTEM = (
    "You are the context compactor inside an agent harness. Summarize the "
    "conversation transcript you are given so the agent can continue its task "
    "with your summary as its only memory of what happened. Preserve: the "
    "task, key decisions, facts and file paths discovered, tool call outcomes, "
    "and any unresolved errors or open questions. Be specific and concise."
)

# per-tool-result cap in the transcript sent to the summarizer
_TRANSCRIPT_RESULT_CAP = 2_000


class ContextConfig(BaseModel):
    """When and how to compact. Recorded in session.start so replay can
    reproduce the same compaction points."""

    model_config = ConfigDict(frozen=True)

    max_input_tokens: int | None = None
    """Compact when a turn's observed input tokens reach this; None disables."""

    keep_recent_messages: int = 4
    """Messages preserved verbatim at the end of the history."""

    summary_max_tokens: int = 2048


def render_transcript(history: list[Message]) -> str:
    """A deterministic plain-text transcript for the summarizer.

    Thinking blocks are excluded — private reasoning is not context the
    summary should carry forward.
    """
    lines: list[str] = []
    for message in history:
        parts: list[str] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[tool_use {block.name} {json.dumps(block.input, sort_keys=True)}]")
            elif isinstance(block, ToolResultBlock):
                content = block.content
                if len(content) > _TRANSCRIPT_RESULT_CAP:
                    content = f"{content[:_TRANSCRIPT_RESULT_CAP]}... [truncated]"
                status = "error" if block.is_error else "ok"
                parts.append(f"[tool_result {status}: {content}]")
            elif isinstance(block, ThinkingBlock):
                continue
        lines.append(f"{message.role}: " + "\n".join(parts))
    return "\n\n".join(lines)


def _split_point(history: list[Message], keep_recent: int) -> int:
    """How many leading messages to summarize away.

    Never splits between an assistant tool_use and its tool_result reply:
    if the first kept message carries tool results, the split moves earlier
    so the pair stays together.
    """
    split = len(history) - keep_recent
    while split > 0 and any(isinstance(block, ToolResultBlock) for block in history[split].content):
        split -= 1
    return split


async def compact_history(
    history: list[Message],
    *,
    provider: Provider,
    context: ContextConfig,
    writer: TraceWriter,
    turn: int,
    observed_input_tokens: int,
    price_overrides: dict[str, ModelPrice] | None = None,
) -> tuple[list[Message], Usage, float | None] | None:
    """Summarize the old history; returns (new_history, usage, cost) or None
    if there is nothing safe to drop. Provider errors propagate to the loop."""
    split = _split_point(history, context.keep_recent_messages)
    if split < 1:
        return None

    request = ModelRequest(
        model=provider.model,
        system=COMPACTION_SYSTEM,
        messages=[
            Message.user(
                "Summarize this agent conversation so far:\n\n" + render_transcript(history[:split])
            )
        ],
        max_tokens=context.summary_max_tokens,
    )
    writer.emit(
        ev.LlmRequest(
            turn=turn, request=request.model_dump(mode="json"), request_hash=request.request_hash()
        )
    )
    started = time.monotonic()
    response = await provider.complete(request)
    latency_ms = int((time.monotonic() - started) * 1000)

    if response.replayed:
        cost: float | None = 0.0
    else:
        cost = cost_usd(response.model, response.usage, price_overrides)
    writer.emit(
        ev.LlmResponse(
            turn=turn,
            response=response.model_dump_trace(),
            usage=response.usage,
            cost_usd=cost,
            latency_ms=latency_ms,
            replayed=response.replayed,
        )
    )

    summary = response.message.text()
    new_history = [
        Message.user(f"[Conversation so far, compacted by the harness]\n\n{summary}"),
        *history[split:],
    ]
    writer.emit(
        ev.ContextCompaction(
            tokens_before=observed_input_tokens,
            # a chars/4 estimate; the next llm.response reports the real count
            tokens_after=len(render_transcript(new_history)) // 4,
            summary=summary,
            dropped_messages=split,
        )
    )
    return new_history, response.usage, cost
