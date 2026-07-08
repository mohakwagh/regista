"""MCP client: a real stdio server's tools join the registry.

Every test spawns tests/mcp_echo_server.py as a subprocess and speaks the
actual MCP wire protocol — no mocks. The last test is the one that matters:
a session that used MCP tools replays hermetically with the server gone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from regista import Agent, replay
from regista.errors import ConfigurationError
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.trace.reader import Trace

pytest.importorskip("mcp")

from regista.tools.mcp import MCPServer, _render_content  # noqa: E402

if TYPE_CHECKING:
    from regista.session import RunResult

SERVER = str(Path(__file__).parent / "mcp_echo_server.py")


def echo_server(**kwargs: object) -> MCPServer:
    return MCPServer.stdio(sys.executable, [SERVER], **kwargs)  # type: ignore[arg-type]


def make_agent(tmp_path: Path, provider: FakeProvider, tools: list) -> Agent:  # type: ignore[type-arg]
    return Agent(
        provider=provider,
        instructions="You are a test agent.",
        tools=tools,
        trace_dir=tmp_path / "traces",
    )


async def test_lists_tools_as_toolspecs() -> None:
    async with echo_server() as server:
        tools = await server.tools()

    specs = {t.spec.name: t.spec for t in tools}
    assert set(specs) == {"echo", "boom"}
    assert specs["echo"].description == "Echo the text back."
    assert specs["echo"].input_schema["properties"]["text"]["type"] == "string"
    assert not specs["echo"].parallel_safe  # the protocol doesn't declare safety


async def test_prefix_namespaces_tool_names() -> None:
    async with echo_server(prefix="srv") as server:
        tools = await server.tools()
    assert {t.spec.name for t in tools} == {"srv__echo", "srv__boom"}


async def run_echo_session(tmp_path: Path) -> RunResult:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "echo", {"text": "hi"})),
            text_response("done"),
        ]
    )
    async with echo_server() as server:
        agent = make_agent(tmp_path, provider, await server.tools())
        return await agent.run("Say hi over MCP")


async def test_agent_calls_an_mcp_tool_like_any_other(tmp_path: Path) -> None:
    result = await run_echo_session(tmp_path)

    assert result.output == "done"
    trace = Trace.load(result.trace_path)
    assert trace.tool_results()["tu_1"].content == "echo: hi"
    # session.start records the MCP tool schema like any local tool's
    assert "echo" in {schema["name"] for schema in trace.start.tool_schemas}


async def test_server_errors_surface_as_error_data(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "boom", {})),
            text_response("recovered"),
        ]
    )
    async with echo_server() as server:
        agent = make_agent(tmp_path, provider, await server.tools())
        result = await agent.run("Trigger the failure")

    assert result.output == "recovered"  # the run survived; the model saw the error
    recorded = Trace.load(result.trace_path).tool_results()["tu_1"]
    assert recorded.is_error
    assert "kaboom" in recorded.content


async def test_mcp_sessions_replay_hermetically(tmp_path: Path) -> None:
    result = await run_echo_session(tmp_path)

    # the server is gone by now — strict replay must not need it
    replayed = await replay(result.trace_path)
    assert replayed.output == "done"
    assert replayed.cost_usd == 0.0
    assert Trace.load(replayed.trace_path).tool_results()["tu_1"].content == "echo: hi"


async def test_tools_require_an_open_connection() -> None:
    with pytest.raises(ConfigurationError, match="not connected"):
        await echo_server().tools()


def test_non_text_content_becomes_a_placeholder() -> None:
    class FakeText:
        type = "text"
        text = "hello"

    class FakeImage:
        type = "image"

    assert _render_content([FakeText(), FakeImage()]) == "hello\n[image content]"
