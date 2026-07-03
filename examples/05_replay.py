"""Deterministic replay: the flight recorder pays off.

Given a trace path (e.g. from example 04), re-runs the whole session with the
LLM served from the recording — no API key, no network, no tool side effects,
$0. Then deliberately tampers with the recording to show strict mode catching
the divergence with a structural diff.

With no argument, records a FakeProvider session first so the example is
self-contained.

Run:  uv run python examples/05_replay.py [path/to/trace.jsonl]
"""

import asyncio
import json
import sys
from pathlib import Path

from regista import Agent, replay, tool
from regista.errors import ReplayDivergence
from regista.providers import FakeProvider, text_response, tool_use_response


@tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: Who to greet.
    """
    return f"Hello, {name}!"


async def record_one() -> Path:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "greet", {"name": "world"})),
            text_response("Greeted the world."),
        ]
    )
    agent = Agent(provider=provider, instructions="You are a greeter.", tools=[greet])
    result = await agent.run("Greet the world")
    return result.trace_path


async def main() -> None:
    trace_path = Path(sys.argv[1]) if len(sys.argv) > 1 else await record_one()
    print("replaying:", trace_path)

    replayed = await replay(trace_path)  # strict mode is the default
    print(f"  output:    {replayed.output!r}")
    print(f"  cost:      ${replayed.cost_usd:.2f}  <- served from the recording")
    print(f"  new trace: {replayed.trace_path} (linked via replay_of)")

    # now tamper with the recording and watch strict mode catch it
    lines = trace_path.read_text().splitlines()
    start = json.loads(lines[0])
    start["instructions"] += " Always answer in French."
    tampered = trace_path.with_name("tampered.jsonl")
    tampered.write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")

    print("\ntampering with the recorded instructions, then replaying strictly...")
    try:
        await replay(tampered)
    except ReplayDivergence as exc:
        print("  ReplayDivergence! the diff points at exactly what changed:")
        for line in exc.diff.splitlines()[:4]:
            print("   ", line)


if __name__ == "__main__":
    asyncio.run(main())
