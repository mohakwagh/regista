"""Policy presets: read_only, workspace, compose — plus gate integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from regista import Agent
from regista.environment import LocalEnvironment
from regista.policy import Allow, Ask, Deny, PermissionRequest, compose, read_only, workspace
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.tools.builtin import builtin_tools
from regista.trace.events import PermissionDecision
from regista.trace.reader import Trace

if TYPE_CHECKING:
    from pathlib import Path


def request(tool_name: str, tool_input: dict[str, Any] | None = None) -> PermissionRequest:
    return PermissionRequest(
        tool_name=tool_name, tool_input=tool_input or {}, tool_use_id="tu_1", turn=1
    )


# --- read_only ------------------------------------------------------------


def test_read_only_allows_reading_builtins() -> None:
    policy = read_only()
    for name in ("read_file", "list_dir", "glob", "search_files"):
        assert isinstance(policy(request(name)), Allow)


def test_read_only_denies_everything_else() -> None:
    policy = read_only()
    for name in ("write_file", "shell", "fetch", "my_custom_tool"):
        decision = policy(request(name))
        assert isinstance(decision, Deny)
        assert name in decision.reason


def test_read_only_accepts_extra_allowed_names() -> None:
    policy = read_only(allow={"get_ticket"})
    assert isinstance(policy(request("get_ticket")), Allow)
    assert isinstance(policy(request("write_file")), Deny)


# --- workspace --------------------------------------------------------------


def test_workspace_allows_file_tools_and_asks_for_the_rest() -> None:
    policy = workspace()
    assert isinstance(policy(request("read_file")), Allow)
    assert isinstance(policy(request("write_file")), Allow)
    for name in ("shell", "fetch", "my_custom_tool"):
        decision = policy(request(name, {"command": "rm -rf /"}))
        assert isinstance(decision, Ask)
        assert name in decision.prompt


# --- compose ----------------------------------------------------------------


async def test_compose_first_non_allow_wins() -> None:
    def deny_shell(req: PermissionRequest) -> Deny | Allow:
        return Deny(reason="no shell") if req.tool_name == "shell" else Allow()

    policy = compose(deny_shell, workspace())
    assert isinstance(await policy(request("shell")), Deny)  # type: ignore[misc]
    assert isinstance(await policy(request("read_file")), Allow)  # type: ignore[misc]
    assert isinstance(await policy(request("fetch")), Ask)  # type: ignore[misc]


async def test_compose_awaits_async_members() -> None:
    async def async_deny(req: PermissionRequest) -> Deny:
        return Deny(reason="always")

    policy = compose(async_deny)
    assert isinstance(await policy(request("read_file")), Deny)  # type: ignore[misc]


def test_compose_name_lists_its_members() -> None:
    policy = compose(read_only(), workspace())
    assert policy.policy_name == "compose(read_only, workspace)"  # type: ignore[attr-defined]


# --- through the loop ---------------------------------------------------------


async def test_read_only_agent_cannot_write(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            tool_use_response(("tu_1", "write_file", {"path": "a.txt", "content": "hi"})),
            text_response("understood"),
        ]
    )
    environment = LocalEnvironment(tmp_path / "workspace")
    agent = Agent(
        provider=provider,
        instructions="You are a reader.",
        tools=builtin_tools(environment),
        policy=read_only(),
        trace_dir=tmp_path / "traces",
    )
    result = await agent.run("Try to write")

    assert not (environment.workspace / "a.txt").exists()
    trace = Trace.load(result.trace_path)
    decision = next(event for event in trace if isinstance(event, PermissionDecision))
    assert decision.decision == "deny"
    assert decision.policy == "read_only"
    denied = trace.tool_results()["tu_1"]
    assert denied.is_error
    assert denied.content.startswith("Permission denied: read_only policy")
