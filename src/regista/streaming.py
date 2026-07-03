"""The streaming event vocabulary for ``agent.stream()``.

Streaming changes *when* you see things, never *what happened*: the trace
records only final requests and responses, so a streamed session produces the
same trace — and replays with the same hashes — as a blocking one. Providers
yield ``TextDelta``/``ThinkingDelta`` items followed by their final
``ModelResponse``; the loop adds the tool and turn events around them.

``synthetic_deltas`` chunks a completed message into deltas — how FakeProvider
and ReplayProvider stream, and the reason replayed sessions can still drive a
live UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from regista.types import Message, TextBlock, ThinkingBlock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from regista.session import RunResult
    from regista.types import StopReason, Usage


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text, in order."""

    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    """A fragment of extended-thinking content."""

    thinking: str


@dataclass(frozen=True)
class ToolCallStarted:
    """The model requested a tool call; the gate and execution come next."""

    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolCallFinished:
    """A tool call resolved — executed, failed, or denied (``is_error``)."""

    tool_use_id: str
    name: str
    content: str
    is_error: bool


@dataclass(frozen=True)
class TurnCompleted:
    """One llm round-trip finished; usage and cost are for this turn only."""

    turn: int
    stop_reason: StopReason
    usage: Usage
    cost_usd: float | None


@dataclass(frozen=True)
class RunCompleted:
    """Always the final event: the same RunResult ``run()`` would return."""

    result: RunResult


ProviderDelta = TextDelta | ThinkingDelta
"""What providers yield before their final ModelResponse."""

StreamEvent = (
    TextDelta | ThinkingDelta | ToolCallStarted | ToolCallFinished | TurnCompleted | RunCompleted
)
"""Everything ``agent.stream()`` can yield."""


def synthetic_deltas(message: Message, *, chunk_size: int = 8) -> Iterator[ProviderDelta]:
    """Chunk a completed message into deltas, for providers that don't
    (or can't) stream natively — FakeProvider and ReplayProvider."""
    for block in message.content:
        if isinstance(block, TextBlock):
            for i in range(0, len(block.text), chunk_size):
                yield TextDelta(block.text[i : i + chunk_size])
        elif isinstance(block, ThinkingBlock):
            for i in range(0, len(block.thinking), chunk_size):
                yield ThinkingDelta(block.thinking[i : i + chunk_size])
