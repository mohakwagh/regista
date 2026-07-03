"""LocalEnvironment: workspace scoping, file ops, and command execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from regista.environment import LocalEnvironment
from regista.errors import WorkspaceViolation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def env(tmp_path: Path) -> LocalEnvironment:
    return LocalEnvironment(tmp_path / "workspace")


def test_workspace_is_created_and_resolved(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path / "made" / "up")
    assert env.workspace.is_dir()
    assert env.workspace.is_absolute()


async def test_write_read_round_trip(env: LocalEnvironment) -> None:
    await env.write_file("src/deep/app.py", "print('hi')\n")
    assert await env.read_file("src/deep/app.py") == "print('hi')\n"


async def test_read_missing_file_raises(env: LocalEnvironment) -> None:
    with pytest.raises(FileNotFoundError):
        await env.read_file("nope.txt")


def test_resolve_allows_the_root_itself(env: LocalEnvironment) -> None:
    assert env.resolve(".") == env.workspace


@pytest.mark.parametrize("path", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
def test_escaping_paths_are_rejected(env: LocalEnvironment, path: str) -> None:
    with pytest.raises(WorkspaceViolation):
        env.resolve(path)


async def test_symlink_pointing_outside_is_rejected(env: LocalEnvironment, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (env.workspace / "innocent.txt").symlink_to(outside)
    with pytest.raises(WorkspaceViolation):
        await env.read_file("innocent.txt")


async def test_list_dir_marks_directories(env: LocalEnvironment) -> None:
    await env.write_file("b.txt", "")
    await env.write_file("a/nested.txt", "")
    assert await env.list_dir() == ["a/", "b.txt"]


async def test_glob_returns_sorted_relative_files(env: LocalEnvironment) -> None:
    await env.write_file("src/b.py", "")
    await env.write_file("src/a.py", "")
    await env.write_file("src/notes.md", "")
    assert await env.glob("**/*.py") == ["src/a.py", "src/b.py"]
    assert await env.glob("**/*.rs") == []


async def test_exec_runs_in_the_workspace(env: LocalEnvironment) -> None:
    result = await env.exec("pwd && echo err >&2")
    assert result.exit_code == 0
    assert result.stdout.strip() == str(env.workspace)
    assert result.stderr.strip() == "err"
    assert not result.timed_out


async def test_exec_reports_failure_exit_codes(env: LocalEnvironment) -> None:
    result = await env.exec("exit 3")
    assert result.exit_code == 3


# "sleep 5 & wait" forces the shell to fork a child that would survive a
# shell-only kill and hold the output pipes — the dash-on-Linux behavior that
# once made this hang, reproduced portably
@pytest.mark.parametrize("command", ["sleep 5", "sleep 5 & wait"])
async def test_exec_kills_the_whole_group_on_timeout(env: LocalEnvironment, command: str) -> None:
    result = await env.exec(command, timeout_s=0.2)
    assert result.timed_out
    assert result.duration_ms < 3000


async def test_exec_does_not_inherit_secrets(
    env: LocalEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_SECRET_KEY", "hunter2")
    result = await env.exec("echo ${SUPER_SECRET_KEY:-unset}")
    assert result.stdout.strip() == "unset"
