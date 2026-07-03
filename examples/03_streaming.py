"""Streaming: see the session as it happens.

Streaming changes when you see things, never what happened — the trace records
final responses only, so a streamed session replays exactly like a blocking one.

Run:  uv run python examples/03_streaming.py
"""

import asyncio

from regista import (
    Agent,
    RunCompleted,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnCompleted,
    tool,
)
from regista.providers import FakeProvider, text_response, tool_use_response


@tool
def lookup(city: str) -> str:
    """Look up the weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: 24°C and sunny"


async def main() -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "lookup", {"city": "Barcelona"}), text="Checking..."),
            text_response("It's 24°C and sunny in Barcelona today."),
        ]
    )
    agent = Agent(provider=provider, instructions="You are a weather bot.", tools=[lookup])

    async for event in agent.stream("What's the weather in Barcelona?"):
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolCallStarted(name=name, input=input):
                print(f"\n[tool started]  {name}({input})")
            case ToolCallFinished(name=name, content=content):
                print(f"[tool finished] {name} -> {content}")
            case TurnCompleted(turn=turn, usage=usage):
                print(f"\n[turn {turn} done: {usage.input_tokens}in/{usage.output_tokens}out]")
            case RunCompleted(result=result):
                print(f"\n[run complete] {result.stop_reason}, trace: {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
