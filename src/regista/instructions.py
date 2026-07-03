"""The Instructions primitive: what the agent is told.

A layered system prompt — a base identity plus named sections — rendered once
per session and recorded verbatim in the session.start trace event. Skills
(v0.3) will contribute sections through this same structure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Instructions(BaseModel):
    """A composable system prompt.

    >>> Instructions(
    ...     base="You are a code-maintenance agent.",
    ...     sections={"Style": "Prefer small diffs."},
    ... ).render()
    'You are a code-maintenance agent.\\n\\n## Style\\n\\nPrefer small diffs.'
    """

    model_config = ConfigDict(frozen=True)

    base: str
    sections: dict[str, str] = {}

    @classmethod
    def coerce(cls, value: str | Instructions) -> Instructions:
        return value if isinstance(value, Instructions) else cls(base=value)

    def with_section(self, title: str, body: str) -> Instructions:
        return self.model_copy(update={"sections": {**self.sections, title: body}})

    def render(self) -> str:
        parts = [self.base.strip()]
        parts.extend(f"## {title}\n\n{body.strip()}" for title, body in self.sections.items())
        return "\n\n".join(parts)
