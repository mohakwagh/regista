"""Built-in tools, the execution environment, and the permission gate.

Three boundaries at work:
- built-in tools act only through an Environment (here: a workspace directory)
- the workspace() policy allows file tools but escalates shell to Ask
- an ask_handler decides escalations; denials become error results the model
  sees and adapts to — never exceptions

Run:  uv run python examples/02_environment_and_policy.py
"""

import asyncio
import tempfile

from regista import Agent
from regista.environment import LocalEnvironment
from regista.policy import PermissionRequest, workspace
from regista.providers import FakeProvider, text_response, tool_use_response
from regista.tools.builtin import builtin_tools
from regista.trace.events import PermissionDecision
from regista.trace.reader import Trace


async def ask_me(request: PermissionRequest) -> bool:
    """The Ask escalation path. A real app would prompt the user here."""
    command = request.tool_input.get("command", "")
    approved = "rm" not in command
    print(f"  [ask_handler] {request.tool_name} wants {request.tool_input} -> {approved}")
    return approved


async def main() -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "write_file", {"path": "notes.txt", "content": "hi"})),
            tool_use_response(("tu_2", "shell", {"command": "cat notes.txt"})),
            tool_use_response(("tu_3", "shell", {"command": "rm -rf /"})),  # will be denied
            text_response("Wrote the file; the destructive command was blocked."),
        ]
    )

    with tempfile.TemporaryDirectory() as workspace_dir:
        environment = LocalEnvironment(workspace_dir)
        agent = Agent(
            provider=provider,
            instructions="You are a careful assistant.",
            tools=builtin_tools(environment),
            policy=workspace(),  # file tools: Allow; shell/fetch/custom: Ask
            ask_handler=ask_me,
        )
        result = await agent.run("Take some notes, then clean up")

        print("\noutput:", result.output)
        print("\npermission decisions recorded in the trace:")
        for event in Trace.load(result.trace_path):
            if isinstance(event, PermissionDecision):
                print(
                    f"  {event.tool_use_id}: {event.decision}"
                    f" (resolution={event.resolution}, policy={event.policy})"
                )


if __name__ == "__main__":
    asyncio.run(main())
