"""The committed real-session fixture: a live run CI re-judges forever at $0.

``tests/fixtures/openai_add.jsonl`` is one real gpt-4o-mini session — a
tool-use round trip — recorded with the eval runner. The always-on test
strict-replays it on every CI run with no key and no network: if a harness
change breaks request assembly, hashing, tool stubbing, or replay itself
against *real* provider data, this is the test that catches it.

Re-record deliberately (after an intentional behavior change):

    REGISTA_RECORD_FIXTURES=1 OPENAI_API_KEY=... uv run pytest tests/test_real_trace.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from regista import Agent, tool
from regista.evals import (
    EvalSuite,
    EvalTask,
    max_cost_usd,
    max_turns_used,
    no_errors,
    output_contains,
    stop_reason_is,
    tool_was_called,
)
from regista.providers import OpenAICompatProvider

FIXTURE = Path(__file__).parent / "fixtures" / "openai_add.jsonl"


@tool
def add(a: int, b: int) -> str:
    """Add two integers exactly.

    Args:
        a: First addend.
        b: Second addend.
    """
    return str(a + b)


def suite() -> EvalSuite:
    return EvalSuite(
        [
            EvalTask(
                name="real gpt-4o-mini tool round trip",
                task="What is 17 + 25?",
                checks=[
                    output_contains("42"),
                    tool_was_called("add"),
                    stop_reason_is("completed"),
                    max_turns_used(4),
                    max_cost_usd(0.01),
                    no_errors(),
                ],
                trace=FIXTURE,
            )
        ]
    )


async def test_recorded_real_session_replays(tmp_path: Path) -> None:
    report = await suite().replay(trace_dir=tmp_path)
    assert report.passed, f"\n{report}"


@pytest.mark.skipif(
    os.environ.get("REGISTA_RECORD_FIXTURES") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="fixture recording needs REGISTA_RECORD_FIXTURES=1 and OPENAI_API_KEY",
)
async def test_record_fixture(tmp_path: Path) -> None:
    provider = OpenAICompatProvider("gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    agent = Agent(
        provider=provider,
        instructions="Use the add tool for any arithmetic. Answer with just the number.",
        tools=[add],
        trace_dir=tmp_path,
        max_turns=4,
        max_cost_usd=0.10,
    )
    try:
        report = await suite().record(agent)
    finally:
        await provider.aclose()
    assert report.passed, f"\n{report}"
    assert FIXTURE.exists()
