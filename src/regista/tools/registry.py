"""The @tool decorator and the registry that dispatches tool calls.

The registry defines *what* a capability is and how it's invoked — never
*where* its effects happen (that's environment/) and never *whether* it may
run (that's policy/, consulted by the loop before dispatch).
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from regista.errors import ConfigurationError, ToolError
from regista.tools.schema import build_input_schema
from regista.types import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ToolExecution:
    """The outcome of one tool call, ready to become a tool.result event."""

    content: str
    is_error: bool
    duration_ms: int


class Tool:
    """A registered tool: a spec (what the model sees) plus the function behind it.

    Calling the Tool calls the underlying function unchanged, so decorated
    functions remain directly usable in normal Python code.
    """

    def __init__(self, fn: Callable[..., Any], spec: ToolSpec) -> None:
        self.fn = fn
        self.spec = spec
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def invoke(self, input: dict[str, Any]) -> str:
        """Run the tool with model-supplied input; sync functions run in a thread."""
        if asyncio.iscoroutinefunction(self.fn):
            result = await self.fn(**input)
        else:
            result = await asyncio.to_thread(self.fn, **input)
        return result if isinstance(result, str) else str(result)


@overload
def tool(fn: Callable[..., Any]) -> Tool: ...
@overload
def tool(
    *, name: str | None = None, description: str | None = None, parallel_safe: bool = False
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parallel_safe: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function into a tool: ``@tool`` or ``@tool(name=..., ...)``.

    The JSON Schema is generated from the signature; the docstring summary
    becomes the description and ``Args:`` lines become parameter descriptions.
    Unsupported signatures raise ConfigurationError here, at decoration time.
    """

    def decorate(fn: Callable[..., Any]) -> Tool:
        summary, input_schema = build_input_schema(fn)
        tool_description = description or summary
        if not tool_description:
            raise ConfigurationError(
                f"tool '{fn.__name__}' needs a description: add a docstring "
                "or pass @tool(description=...)"
            )
        spec = ToolSpec(
            name=name or fn.__name__,
            description=tool_description,
            input_schema=input_schema,
            parallel_safe=parallel_safe,
        )
        return Tool(fn, spec)

    return decorate(fn) if fn is not None else decorate


class ToolRegistry:
    """Name → Tool, plus execution with timing and error capture."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ConfigurationError(f"duplicate tool name '{tool.spec.name}'")
        self._tools[tool.spec.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool '{name}'") from None

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def execute(self, name: str, input: dict[str, Any]) -> ToolExecution:
        """Execute a tool call, capturing failures as data.

        A tool that raises produces ``is_error=True`` content the model can
        adapt to — only an unknown tool name is a harness fault (ToolError,
        raised by ``get``).
        """
        tool = self.get(name)
        started = time.monotonic()
        try:
            content = await tool.invoke(input)
            is_error = False
        except Exception as exc:
            content = f"{type(exc).__name__}: {exc}"
            is_error = True
        duration_ms = int((time.monotonic() - started) * 1000)
        return ToolExecution(content=content, is_error=is_error, duration_ms=duration_ms)
