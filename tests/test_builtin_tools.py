"""Built-in tools: environment-backed effects, output caps, error-as-data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
import respx

from regista.environment import LocalEnvironment
from regista.tools import ToolRegistry
from regista.tools.builtin import builtin_tools

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(tmp_path / "workspace")


@pytest.fixture
def registry(env: LocalEnvironment) -> ToolRegistry:
    return ToolRegistry(builtin_tools(env))


def test_toolset_shape(registry: ToolRegistry) -> None:
    specs = {spec.name: spec for spec in registry.specs()}
    assert set(specs) == {
        "read_file",
        "write_file",
        "list_dir",
        "glob",
        "search_files",
        "shell",
        "fetch",
    }
    # read-only tools may run concurrently; mutating ones may not
    assert {name for name, spec in specs.items() if not spec.parallel_safe} == {
        "write_file",
        "shell",
    }
    assert all(spec.description for spec in specs.values())


async def test_write_then_read(registry: ToolRegistry, env: LocalEnvironment) -> None:
    written = await registry.execute("write_file", {"path": "a.txt", "content": "hello"})
    assert not written.is_error
    assert "a.txt" in written.content
    read = await registry.execute("read_file", {"path": "a.txt"})
    assert read.content == "hello"


async def test_read_output_is_capped(env: LocalEnvironment) -> None:
    registry = ToolRegistry(builtin_tools(env, max_output_chars=100))
    await env.write_file("big.txt", "x" * 500)
    result = await registry.execute("read_file", {"path": "big.txt"})
    assert result.content.startswith("x" * 100)
    assert "400 of 500 characters omitted" in result.content


async def test_escape_attempt_is_error_data(registry: ToolRegistry) -> None:
    result = await registry.execute("read_file", {"path": "../outside.txt"})
    assert result.is_error
    assert result.content.startswith("WorkspaceViolation:")


async def test_list_dir_and_empty_message(registry: ToolRegistry, env: LocalEnvironment) -> None:
    empty = await registry.execute("list_dir", {})
    assert "empty directory" in empty.content
    await env.write_file("src/app.py", "")
    listed = await registry.execute("list_dir", {})
    assert listed.content == "src/"


async def test_glob_matches_and_no_match_message(
    registry: ToolRegistry, env: LocalEnvironment
) -> None:
    await env.write_file("src/a.py", "")
    assert (await registry.execute("glob", {"pattern": "**/*.py"})).content == "src/a.py"
    assert "no files match" in (await registry.execute("glob", {"pattern": "*.rs"})).content


async def test_search_files_reports_path_line_text(
    registry: ToolRegistry, env: LocalEnvironment
) -> None:
    await env.write_file("src/a.py", "x = 1\ntodo: fix this\n")
    await env.write_file("notes.md", "TODO elsewhere\n")
    result = await registry.execute("search_files", {"pattern": r"todo", "glob": "**/*.py"})
    assert result.content == "src/a.py:2: todo: fix this"
    missing = await registry.execute("search_files", {"pattern": "xyzzy"})
    assert "no matches" in missing.content


async def test_search_files_can_match_case_insensitively(
    registry: ToolRegistry, env: LocalEnvironment
) -> None:
    await env.write_file("src/a.py", "TODO: fix this\n")
    result = await registry.execute(
        "search_files",
        {"pattern": r"todo", "glob": "**/*.py", "case_insensitive": True},
    )
    assert result.content == "src/a.py:1: TODO: fix this"


async def test_shell_formats_output(registry: ToolRegistry) -> None:
    result = await registry.execute("shell", {"command": "echo out && echo err >&2 && exit 4"})
    assert "exit code: 4" in result.content
    assert "stdout:\nout" in result.content
    assert "stderr:\nerr" in result.content


async def test_shell_timeout_is_reported(env: LocalEnvironment) -> None:
    registry = ToolRegistry(builtin_tools(env, shell_timeout_s=0.2))
    result = await registry.execute("shell", {"command": "sleep 5"})
    assert "timed out after 0.2s" in result.content


@respx.mock
async def test_fetch_returns_status_and_body(registry: ToolRegistry) -> None:
    respx.get("https://example.com/page").mock(return_value=httpx.Response(200, text="hello web"))
    result = await registry.execute("fetch", {"url": "https://example.com/page"})
    assert result.content == "HTTP 200\n\nhello web"


async def test_fetch_rejects_non_http_schemes(registry: ToolRegistry) -> None:
    result = await registry.execute("fetch", {"url": "file:///etc/passwd"})
    assert result.is_error
    assert "http(s)" in result.content
