"""Schema generation, decorator, and dispatch tests for the tool interface."""

from typing import Literal

import pytest

from regista.errors import ConfigurationError, ToolError
from regista.instructions import Instructions
from regista.tools import Tool, ToolRegistry, tool

# --- schema generation (golden tests) --------------------------------------


@tool
def get_ticket(ticket_id: str, include_comments: bool = False) -> str:
    """Fetch a ticket from the issue tracker.

    Args:
        ticket_id: The ticket key, e.g. PROJ-123.
        include_comments: Whether to include the comment thread.
    """
    return f"{ticket_id}:{include_comments}"


def test_golden_schema_from_signature_and_docstring() -> None:
    assert get_ticket.spec.name == "get_ticket"
    assert get_ticket.spec.description == "Fetch a ticket from the issue tracker."
    assert get_ticket.spec.input_schema == {
        "type": "object",
        "properties": {
            "ticket_id": {
                "type": "string",
                "description": "The ticket key, e.g. PROJ-123.",
            },
            "include_comments": {
                "type": "boolean",
                "description": "Whether to include the comment thread.",
                "default": False,
            },
        },
        "required": ["ticket_id"],
    }


def test_golden_schema_literal_list_optional() -> None:
    @tool(description="search")
    def search(
        query: str,
        mode: Literal["exact", "fuzzy"] = "exact",
        tags: list[str] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> str:
        return query

    assert search.spec.input_schema["properties"] == {
        "query": {"type": "string"},
        "mode": {"enum": ["exact", "fuzzy"], "type": "string", "default": "exact"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer", "default": 10},
        "threshold": {"type": "number", "default": 0.5},
    }
    assert search.spec.input_schema["required"] == ["query"]


def test_decorator_overrides_and_parallel_safe() -> None:
    @tool(name="renamed", description="custom", parallel_safe=True)
    def original(x: int) -> int:
        return x

    assert original.spec.name == "renamed"
    assert original.spec.description == "custom"
    assert original.spec.parallel_safe is True
    assert original(3) == 3  # decorated function stays directly callable


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        (lambda: tool(description="d")(lambda x: x), "missing a type annotation"),
        (lambda: tool(description="d")(_unsupported_dict), "unsupported parameter type"),
        (lambda: tool(description="d")(_unsupported_union), "unions other than"),
        (lambda: tool(_no_description), "needs a description"),
    ],
    ids=["unannotated", "dict-param", "int-or-str-union", "no-description"],
)
def test_decoration_fails_fast(definition: object, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        definition()  # type: ignore[operator]


def _unsupported_dict(data: dict) -> str:  # type: ignore[type-arg]
    return str(data)


def _unsupported_union(value: int | str) -> str:
    return str(value)


def _no_description(x: int) -> int:
    return x


# --- registry & dispatch ----------------------------------------------------


async def test_registry_executes_sync_and_async_tools() -> None:
    @tool(description="sync")
    def double(x: int) -> int:
        return x * 2

    @tool(description="async")
    async def shout(text: str) -> str:
        return text.upper()

    registry = ToolRegistry([double, shout])
    assert [s.name for s in registry.specs()] == ["double", "shout"]

    sync_result = await registry.execute("double", {"x": 21})
    async_result = await registry.execute("shout", {"text": "goal"})
    assert (sync_result.content, sync_result.is_error) == ("42", False)
    assert async_result.content == "GOAL"


async def test_tool_exception_becomes_error_result_not_exception() -> None:
    @tool(description="explodes")
    def broken(x: int) -> str:
        raise RuntimeError("kaboom")

    result = await ToolRegistry([broken]).execute("broken", {"x": 1})
    assert result.is_error is True
    assert result.content == "RuntimeError: kaboom"


async def test_unknown_tool_is_a_harness_fault() -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        await ToolRegistry([]).execute("ghost", {})


def test_duplicate_registration_rejected() -> None:
    @tool(description="d")
    def twin(x: int) -> int:
        return x

    with pytest.raises(ConfigurationError, match="duplicate"):
        ToolRegistry([twin, Tool(twin.fn, twin.spec)])


# --- instructions -------------------------------------------------------------


def test_instructions_render_and_coerce() -> None:
    plain = Instructions.coerce("You are a triage agent.")
    assert plain.render() == "You are a triage agent."

    layered = plain.with_section("Style", "Be terse.").with_section("Safety", "Ask first.")
    assert layered.render() == (
        "You are a triage agent.\n\n## Style\n\nBe terse.\n\n## Safety\n\nAsk first."
    )
    assert Instructions.coerce(layered) is layered
