"""ContainerEnvironment: commands isolated in Docker, files via bind mount.

Needs a running Docker daemon (CI's Ubuntu runners have one; skipped
elsewhere). Uses alpine:3 — a ~3 MB pull on first run.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from regista.environment import ContainerEnvironment
from regista.errors import ConfigurationError, WorkspaceViolation

if TYPE_CHECKING:
    from pathlib import Path

IMAGE = "alpine:3"


def _docker_running() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _docker_running(), reason="needs a running docker daemon")


@pytest.fixture
async def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    async with ContainerEnvironment(tmp_path / "workspace", image=IMAGE) as environment:
        yield environment


async def test_exec_runs_inside_the_container(env: ContainerEnvironment) -> None:
    result = await env.exec("cat /etc/os-release && pwd")
    assert result.exit_code == 0
    assert "Alpine" in result.stdout  # definitely not the host
    assert result.stdout.strip().endswith("/workspace")


async def test_files_flow_both_ways_through_the_bind_mount(env: ContainerEnvironment) -> None:
    await env.write_file("from_host.txt", "written on the host\n")
    seen = await env.exec("cat from_host.txt")
    assert seen.stdout == "written on the host\n"

    made = await env.exec("echo made-in-container > from_container.txt")
    assert made.exit_code == 0
    assert (await env.read_file("from_container.txt")).strip() == "made-in-container"


async def test_host_environment_never_reaches_the_container(
    env: ContainerEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPER_SECRET_KEY", "hunter2")
    result = await env.exec("echo ${SUPER_SECRET_KEY:-unset}")
    assert result.stdout.strip() == "unset"


async def test_exec_times_out_inside_the_container(env: ContainerEnvironment) -> None:
    result = await env.exec("sleep 30", timeout_s=1)
    assert result.timed_out
    assert result.duration_ms < 10_000


async def test_exit_codes_propagate(env: ContainerEnvironment) -> None:
    assert (await env.exec("exit 3")).exit_code == 3


async def test_file_scoping_is_still_enforced(env: ContainerEnvironment) -> None:
    with pytest.raises(WorkspaceViolation):
        env.resolve("../escape.txt")


async def test_exec_requires_a_running_container(tmp_path: Path) -> None:
    stopped = ContainerEnvironment(tmp_path, image=IMAGE)
    with pytest.raises(ConfigurationError, match="not running"):
        await stopped.exec("true")


async def test_bad_image_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="could not start container"):
        async with ContainerEnvironment(tmp_path, image="regista-definitely-not-an-image:0"):
            pass  # pragma: no cover
