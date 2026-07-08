"""The eval/regression runner: record once, assert forever, $0 in CI.

A suite of tasks judged by checks on outcome AND trace shape. record() runs
live and saves passing traces as fixtures; replay() re-judges those fixtures
with no provider, no keys, no cost — and any divergence from the recording
fails the task. (Scripted model here, so this whole demo is $0 too.)

Run:  uv run python examples/07_evals.py
"""

import asyncio
import tempfile
from pathlib import Path

from regista import Agent, tool
from regista.evals import (
    EvalSuite,
    EvalTask,
    max_turns_used,
    no_errors,
    output_contains,
    tool_never_called,
    tool_was_called,
)
from regista.providers import FakeProvider, text_response, tool_use_response


@tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: Who to greet.
    """
    return f"Hello, {name}!"


def make_agent() -> Agent:
    return Agent(
        provider=FakeProvider(
            [
                tool_use_response(("tu_1", "greet", {"name": "world"})),
                text_response("Greeted the world. Done."),
            ]
        ),
        instructions="You are a greeter.",
        tools=[greet],
    )


async def main() -> None:
    fixtures = Path(tempfile.mkdtemp()) / "fixtures"
    suite = EvalSuite(
        [
            EvalTask(
                name="greets the world",
                task="Greet the world",
                checks=[
                    output_contains("Done"),
                    tool_was_called("greet"),
                    tool_never_called("shell"),
                    max_turns_used(3),
                    no_errors(),
                ],
                trace=fixtures / "greets_the_world.jsonl",
            )
        ]
    )

    recorded = await suite.record(make_agent())
    print("record():")
    print(recorded, "\n")

    # CI mode: no agent, no provider, no keys — judged against the fixture
    replayed = await suite.replay()
    print("replay() — $0:")
    print(replayed)


if __name__ == "__main__":
    asyncio.run(main())
