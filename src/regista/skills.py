"""Skills: reusable bundles of instructions + tools.

A Skill is declarative data — a named instruction fragment plus the tools
that make it actionable. Loading skills into an Agent appends each fragment
to the system prompt as an ``Instructions`` section titled ``Skill: {name}``
and registers the tools alongside the agent's own.

Nothing else changes: the rendered instructions and the full tool schemas
land in ``session.start`` exactly as they always do (plus the skill names,
for provenance), so a skilled session traces and replays like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regista.tools import Tool


@dataclass(frozen=True)
class Skill:
    """A named instruction fragment and the tools that go with it.

    >>> reviewer = Skill(
    ...     name="code-review",
    ...     instructions="Review diffs for correctness first, style second.",
    ...     tools=[fetch_diff, post_comment],
    ... )
    >>> agent = Agent(provider=..., instructions="...", skills=[reviewer])
    """

    name: str
    instructions: str
    tools: list[Tool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a Skill needs a non-empty name")
        if not self.instructions.strip():
            raise ValueError(f"skill '{self.name}' needs non-empty instructions")
