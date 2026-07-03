"""Opt-in live smoke test against the real Anthropic API.

Run with:  REGISTA_LIVE_TESTS=1 ANTHROPIC_API_KEY=... uv run pytest tests/test_live_smoke.py
Uses a haiku-class model; costs on the order of a cent. Skipped everywhere else.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from regista import Agent, replay, tool
from regista.providers import AnthropicProvider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("REGISTA_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live smoke needs REGISTA_LIVE_TESTS=1 and ANTHROPIC_API_KEY",
)


@tool
def add(a: int, b: int) -> str:
    """Add two integers exactly.

    Args:
        a: First addend.
        b: Second addend.
    """
    return str(a + b)


async def test_live_round_trip_and_replay(tmp_path: Path) -> None:
    agent = Agent(
        provider=AnthropicProvider("claude-haiku-4-5"),
        instructions="Use the add tool for any arithmetic. Answer with just the number.",
        tools=[add],
        trace_dir=tmp_path,
        max_turns=4,
        max_cost_usd=0.10,
    )
    result = await agent.run("What is 17 + 25?")

    assert result.stop_reason == "completed"
    assert "42" in result.output
    assert result.cost_usd is not None and result.cost_usd > 0
    assert result.trace_path.exists()

    # the recording replays hermetically, for free
    replayed = await replay(result.trace_path)
    assert replayed.output == result.output
    assert replayed.cost_usd == 0.0
