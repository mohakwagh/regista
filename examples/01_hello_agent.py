"""The smallest regista agent — zero API keys, zero cost.

FakeProvider is public API, not a test hack: you script the model, the real
loop runs, and a real trace is written. This is also how you unit-test your
own agents.

Run:  uv run python examples/01_hello_agent.py
"""

import asyncio

from regista import Agent, tool
from regista.providers import FakeProvider, text_response, tool_use_response
from regista.trace.reader import Trace


@tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: Who to greet.
    """
    return f"Hello, {name}!"


async def main() -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "greet", {"name": "world"})),
            text_response("Greeted the world. Done."),
        ]
    )
    agent = Agent(provider=provider, instructions="You are a greeter.", tools=[greet])

    result = await agent.run("Greet the world")

    print("output:     ", result.output)
    print("stop_reason:", result.stop_reason)
    print("turns:      ", result.turns)
    print("trace:      ", result.trace_path)

    print("\nevery step the harness took is in the trace:")
    for event in Trace.load(result.trace_path):
        print(f"  {event.seq:>2}  {event.type}")


if __name__ == "__main__":
    asyncio.run(main())
