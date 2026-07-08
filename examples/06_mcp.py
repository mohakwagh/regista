"""MCP tools with zero setup — this file is both the client and the server.

Run with "server" as an argument it becomes a tiny MCP server (one `shout`
tool); run normally it spawns itself as that server over stdio, hands the
server's tools to an agent, and — the regista twist — strictly replays the
session for $0 *after the server is gone*: MCP tool results live in the
trace like any other.

Run:  uv run --extra mcp python examples/06_mcp.py
"""

import asyncio
import sys


def run_server() -> None:
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("shouter")

    @server.tool()
    def shout(text: str) -> str:
        """Return the text, loudly."""
        return text.upper() + "!"

    server.run()


async def main() -> None:
    from regista import Agent, replay
    from regista.providers import FakeProvider, text_response, tool_use_response
    from regista.tools.mcp import MCPServer

    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "shout", {"text": "regista speaks mcp"})),
            text_response("Shouted it. Done."),
        ]
    )

    async with MCPServer.stdio(sys.executable, [__file__, "server"]) as server:
        agent = Agent(
            provider=provider,
            instructions="You are a town crier.",
            tools=await server.tools(),
        )
        result = await agent.run("Shout the news")

    print("output:", result.output)
    print("trace: ", result.trace_path)

    # the server subprocess is dead now — replay doesn't care
    replayed = await replay(result.trace_path)
    print(f"replayed for ${replayed.cost_usd:.2f}, tool result intact:", replayed.output)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        asyncio.run(main())
