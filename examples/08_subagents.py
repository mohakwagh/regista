"""Subagents: one agent delegating to another, with linked traces — $0.

The child agent becomes a tool of the parent via as_tool(). It runs in its
own session: isolated context, its own policy and budgets, its own trace —
tagged with the parent's session id, so the whole delegation tree is
reconstructable from the trace files alone.

Run:  uv run python examples/08_subagents.py
"""

import asyncio

from regista import Agent
from regista.providers import FakeProvider, text_response, tool_use_response
from regista.trace.reader import Trace


async def main() -> None:
    researcher = Agent(
        provider=FakeProvider([text_response("Il regista dirige il gioco.")]),
        instructions="You are a translation subagent.",
        max_cost_usd=0.50,  # the child polices its own budget
        trace_dir=".regista/traces",
    )

    parent = Agent(
        provider=FakeProvider(
            [
                tool_use_response(
                    ("tu_1", "translator", {"task": "Translate: the regista directs the game"})
                ),
                text_response("Translated. Done."),
            ]
        ),
        instructions="You are an orchestrator. Delegate translations.",
        tools=[
            researcher.as_tool(
                name="translator",
                description="Delegate a translation task to the translation subagent.",
            )
        ],
        trace_dir=".regista/traces",
    )

    result = await parent.run("Translate a sentence for me")
    print("parent output:", result.output)
    print("parent trace: ", result.trace_path)

    # every session in the tree is linked by parent_session_id
    for path in sorted(result.trace_path.parent.glob("*.jsonl")):
        start = Trace.load(path).start
        if start.session_id == result.session_id or start.parent_session_id:
            arrow = f" (child of {start.parent_session_id})" if start.parent_session_id else ""
            print(f"  {start.session_id}: {start.task!r}{arrow}")


if __name__ == "__main__":
    asyncio.run(main())
