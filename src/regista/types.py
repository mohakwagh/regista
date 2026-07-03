"""Provider-neutral message vocabulary shared by every module.

The shapes here are a superset of the Anthropic Messages API (role plus a list
of typed content blocks). Adapters translate their wire formats *into* this
model, never out of it: rich-to-flat translation (for OpenAI-style APIs) is
easy, flat-to-rich is lossy. See ARCHITECTURE.md §7.

This module depends on nothing else in regista.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant", "system"]

StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "pause_turn",
    "other",
]


class TextBlock(BaseModel):
    """Plain assistant or user text."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Extended-thinking content.

    ``signature`` is preserved verbatim: the Anthropic API requires it when
    thinking blocks are sent back in multi-turn history.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(BaseModel):
    """The model requesting a tool call. The harness executes; the model only asks."""

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """The harness reporting a tool's outcome back to the model.

    A permission denial or tool failure is carried as ``is_error=True`` —
    it is data the model can adapt to, not an exception.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One conversation entry: a role and a list of typed content blocks."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)])

    def text(self) -> str:
        """Concatenated text of all TextBlocks (ignores tool/thinking blocks)."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [block for block in self.content if isinstance(block, ToolUseBlock)]


class ToolSpec(BaseModel):
    """What the model sees of a tool: name, description, and JSON Schema input.

    ``parallel_safe`` is harness metadata (read-only tools opt in to concurrent
    execution); it is never sent to the model.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    parallel_safe: bool = False


class Usage(BaseModel):
    """Token accounting as reported by the provider (never locally estimated).

    Cache fields map from Anthropic's ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``; providers without caching leave them 0.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )
