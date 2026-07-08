"""MCP client: any MCP server's tools join the registry.

Model Context Protocol servers expose tools over a standard wire protocol;
``MCPServer`` connects to one (a stdio subprocess or a streamable-HTTP
endpoint), lists its tools, and wraps each as a regular regista ``Tool`` —
same ToolSpec, same dispatch, same trace events, same policy gate. The rest
of the harness cannot tell an MCP tool from a local ``@tool`` function, which
is the point: traces record them identically, so replay stubs them
identically — a session that used MCP tools replays hermetically with the
server switched off.

Requires the ``mcp`` extra: ``pip install "regista-harness[mcp]"``.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from regista.errors import ConfigurationError
from regista.tools.registry import Tool
from regista.types import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


def _mcp() -> Any:
    try:
        import mcp
        import mcp.client.stdio
        import mcp.client.streamable_http
    except ImportError as exc:  # pragma: no cover — dev env always has it
        raise ConfigurationError(
            'MCP support needs the mcp extra: pip install "regista-harness[mcp]"'
        ) from exc
    return mcp


class MCPToolError(Exception):
    """The MCP server reported a failed tool call (``isError=True``).

    Raised inside the wrapped tool so the registry converts it to error-data
    the model can react to — exactly like a raising local tool.
    """


def _render_content(items: list[Any]) -> str:
    """MCP content blocks → the string the model gets. Text passes through;
    anything else (images, resources) becomes an honest placeholder."""
    parts = [
        item.text if getattr(item, "type", None) == "text" else f"[{item.type} content]"
        for item in items
    ]
    return "\n".join(parts)


class MCPServer:
    """A connection to one MCP server; an async context manager.

    >>> async with MCPServer.stdio("uvx", ["mcp-server-fetch"]) as server:
    ...     agent = Agent(provider=..., instructions=..., tools=await server.tools())
    ...     await agent.run(task)

    The connection must stay open for as long as the agent may call the
    tools. ``prefix`` namespaces tool names (``{prefix}__{name}``) so two
    servers exporting the same tool name can coexist in one registry.
    """

    def __init__(
        self,
        transport_factory: Callable[[], Any],
        *,
        prefix: str | None = None,
        timeout_s: float | None = 30.0,
    ) -> None:
        self._transport_factory = transport_factory
        self._prefix = prefix
        self._timeout_s = timeout_s
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        prefix: str | None = None,
        timeout_s: float | None = 30.0,
    ) -> MCPServer:
        """A server spawned as a subprocess, spoken to over stdin/stdout."""
        mcp = _mcp()
        params = mcp.StdioServerParameters(command=command, args=args or [], env=env)
        return cls(
            lambda: mcp.client.stdio.stdio_client(params), prefix=prefix, timeout_s=timeout_s
        )

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        prefix: str | None = None,
        timeout_s: float | None = 30.0,
    ) -> MCPServer:
        """A remote server spoken to over streamable HTTP."""
        mcp = _mcp()
        return cls(
            lambda: mcp.client.streamable_http.streamablehttp_client(url, headers=headers),
            prefix=prefix,
            timeout_s=timeout_s,
        )

    async def __aenter__(self) -> MCPServer:
        mcp = _mcp()
        self._stack = AsyncExitStack()
        try:
            # stdio yields (read, write); streamable HTTP appends a session-id getter
            streams = await self._stack.enter_async_context(self._transport_factory())
            timeout = timedelta(seconds=self._timeout_s) if self._timeout_s else None
            self._session = await self._stack.enter_async_context(
                mcp.ClientSession(streams[0], streams[1], read_timeout_seconds=timeout)
            )
            await self._session.initialize()
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise ConfigurationError(
                "MCPServer is not connected — use it as an async context manager"
            )
        return self._session

    async def tools(self) -> list[Tool]:
        """List the server's tools, each wrapped as a regista Tool."""
        listed = await self._require_session().list_tools()
        return [self._wrap(t) for t in listed.tools]

    def _wrap(self, mcp_tool: Any) -> Tool:
        mcp_name: str = mcp_tool.name
        name = f"{self._prefix}__{mcp_name}" if self._prefix else mcp_name
        spec = ToolSpec(
            name=name,
            description=mcp_tool.description or f"MCP tool '{mcp_name}'",
            input_schema=mcp_tool.inputSchema,
            parallel_safe=False,  # the protocol doesn't declare it; assume effects
        )

        async def call(**kwargs: Any) -> str:
            result = await self._require_session().call_tool(mcp_name, arguments=kwargs)
            text = _render_content(result.content)
            if result.isError:
                raise MCPToolError(text or f"MCP tool '{mcp_name}' failed")
            return text

        call.__name__ = name
        return Tool(call, spec)
