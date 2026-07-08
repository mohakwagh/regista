# Tools

**What the model can ask for.** Module: `regista/tools/`.

A tool is a typed Python function; `@tool` turns its signature and docstring into the JSON
Schema the model sees. The registry dispatches calls. The model only ever *requests* a
call — the harness executes.

```python
from regista import tool

@tool
def get_ticket(ticket_id: str, include_comments: bool = False) -> str:
    """Fetch a ticket.

    Args:
        ticket_id: The ticket key, e.g. PROJ-123.
        include_comments: Whether to include the comment thread.
    """
    ...
```

- Supported parameter types (v0.1): `str`, `int`, `float`, `bool`, `Literal[...]`,
  `list[...]`, `X | None`. Anything else raises `ConfigurationError` **at decoration
  time** — a tool that can't be described accurately should fail before any agent runs.
- The docstring summary becomes the tool description; `Args:` lines become per-parameter
  descriptions. A missing description is a decoration-time error too.
- `@tool(parallel_safe=True)` opts a read-only tool into concurrent execution: when *every*
  tool call in a model turn is parallel-safe, the loop runs them with `asyncio.gather`.
- Sync functions run in a thread (`asyncio.to_thread`); async functions are awaited.
- A tool that raises produces `tool_result(is_error=True)` — data the model adapts to,
  never an exception that kills the session.

## Built-in tools

`builtin_tools(environment)` returns the standard seven: `read_file`, `write_file`,
`list_dir`, `glob`, `search_files`, `shell`, `fetch`. Every effect goes through the
[environment](environment.md) — the built-ins never touch the OS directly. Outputs are
capped with a truncation marker so one `cat` of a huge file can't blow the context window.

## MCP servers

Any [Model Context Protocol](https://modelcontextprotocol.io) server's tools can join the
registry — no code per tool. Requires the `mcp` extra
(`pip install "regista-harness[mcp]"`):

```python
from regista.tools.mcp import MCPServer

async with MCPServer.stdio("uvx", ["mcp-server-fetch"]) as server:
    agent = Agent(provider=..., instructions=..., tools=await server.tools())
    result = await agent.run(task)
```

`MCPServer.stdio(command, args)` spawns a local server subprocess;
`MCPServer.http(url)` speaks streamable HTTP to a remote one. Keep the connection open
while the agent runs. `prefix="fs"` namespaces tool names (`fs__read_file`) so two servers
exporting the same name can coexist.

The wrapped tools are ordinary regista tools: same trace events, same policy gate, and a
server-side failure comes back as error-data the model can react to (never a crash).
`parallel_safe` is always `False` — the protocol doesn't declare side-effect safety.
Because results are recorded in the trace like any tool's, **a session that used MCP tools
replays hermetically with the server switched off** — `examples/06_mcp.py` demonstrates
exactly that.

## Boundaries

The registry defines *what* a capability is — never *where* its effects happen (that's
`environment/`) and never *whether* it may run (that's [`policy/`](policy.md), consulted by
the loop before dispatch).
