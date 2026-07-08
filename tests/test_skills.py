"""Skills: instruction fragments + tool bundles that load into an Agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from regista import Agent, Skill, replay, tool
from regista.errors import ConfigurationError
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path


@tool
def lookup(term: str) -> str:
    """Look up a term in the glossary.

    Args:
        term: The term to define.
    """
    return f"{term}: a very important concept"


GLOSSARY = Skill(
    name="glossary",
    instructions="When asked about terminology, always consult the lookup tool.",
    tools=[lookup],
)


def make_agent(tmp_path: Path, provider: FakeProvider) -> Agent:
    return Agent(
        provider=provider,
        instructions="You are a helpful agent.",
        skills=[GLOSSARY],
        trace_dir=tmp_path / "traces",
    )


def test_skill_fragment_becomes_an_instructions_section(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, FakeProvider([]))
    rendered = agent.instructions.render()
    assert "## Skill: glossary" in rendered
    assert "always consult the lookup tool" in rendered


async def test_skill_tools_execute_and_everything_is_traced(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "lookup", {"term": "regista"})),
            text_response("A regista is a very important concept."),
        ]
    )
    result = await make_agent(tmp_path, provider).run("What is a regista?")

    assert result.stop_reason == "completed"
    trace = Trace.load(result.trace_path)
    start = trace.start
    assert start.skills == ["glossary"]
    assert "## Skill: glossary" in start.instructions
    assert "lookup" in {schema["name"] for schema in start.tool_schemas}
    assert trace.tool_results()["tu_1"].content == "regista: a very important concept"


async def test_skilled_sessions_replay_like_any_other(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "lookup", {"term": "regista"})),
            text_response("done"),
        ]
    )
    result = await make_agent(tmp_path, provider).run("What is a regista?")

    replayed = await replay(result.trace_path)
    assert replayed.output == result.output
    assert replayed.cost_usd == 0.0


def test_skill_tool_colliding_with_agent_tool_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="duplicate tool name"):
        Agent(
            provider=FakeProvider([]),
            instructions="x",
            tools=[lookup],
            skills=[GLOSSARY],
            trace_dir=tmp_path,
        )


def test_blank_skills_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        Skill(name="  ", instructions="x")
    with pytest.raises(ValueError, match="non-empty instructions"):
        Skill(name="x", instructions="")
