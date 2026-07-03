"""A real agent on the Anthropic API, read-only, on a tiny model.

Needs ANTHROPIC_API_KEY. Costs roughly a cent. Everything else in examples/
runs on FakeProvider for free — this one shows the exact same code shape
against a live model, and prints the trace path so you can replay it with
examples/05_replay.py at $0.

Run:  ANTHROPIC_API_KEY=... uv run python examples/04_real_provider.py
"""

import asyncio
import os
import sys
import tempfile

from regista import Agent
from regista.environment import LocalEnvironment
from regista.policy import read_only
from regista.providers import AnthropicProvider
from regista.tools.builtin import builtin_tools


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY to run this example (see examples 01-03 for $0 ones)")

    with tempfile.TemporaryDirectory() as workspace_dir:
        environment = LocalEnvironment(workspace_dir)
        await environment.write_file(
            "haiku.txt",
            "deep-lying playmaker\nthreads one pass through the middle\nthe game turns on it\n",
        )

        agent = Agent(
            provider=AnthropicProvider("claude-haiku-4-5"),
            instructions="You are a filesystem explorer. Be terse.",
            tools=builtin_tools(environment),
            policy=read_only(),  # write_file/shell/fetch are denied, and that's traced
            max_turns=6,
            max_cost_usd=0.10,
        )
        result = await agent.run("What files are in the workspace? Read one and describe it.")

        print("output:", result.output)
        print(f"cost:   ${result.cost_usd:.4f}")
        print("trace: ", result.trace_path)
        print("\nreplay it for free:")
        print(f"  uv run python examples/05_replay.py {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
