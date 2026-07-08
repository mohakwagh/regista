"""A minimal stdio MCP server used as a test fixture (not a test module).

Spawned by tests/test_mcp.py via ``MCPServer.stdio(sys.executable, [this file])``
so the MCP client is exercised over the real wire protocol with no network.
"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("regista-echo")


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@server.tool()
def boom() -> str:
    """Always fails."""
    raise ValueError("kaboom")


if __name__ == "__main__":
    server.run()
