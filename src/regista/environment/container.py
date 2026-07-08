"""ContainerEnvironment: the drop-in the environment seam was built for.

Same ``Environment`` protocol, one difference that matters: **commands run
inside a Docker container**, not on your machine. The workspace directory is
bind-mounted into the container, so file operations keep LocalEnvironment's
exact semantics (host-side, workspace-pinned, symlink-escape rejection) while
``exec`` is isolated — a shell command can see the container, not your home
directory, and inherits none of your environment variables.

Built on the ``docker`` CLI (no SDK dependency): ``docker run -d`` a
long-lived container on ``__aenter__``, ``docker exec`` per command,
``docker kill`` on ``__aexit__``. Timeouts are enforced twice — the
``timeout`` binary inside the container (present in busybox/coreutils
images) and a client-side kill as backstop.

>>> async with ContainerEnvironment("./sandbox", image="alpine:3") as env:
...     agent = Agent(..., tools=builtin_tools(env))
...     await agent.run(task)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from typing import TYPE_CHECKING

from regista.environment.base import ExecResult
from regista.environment.local import LocalEnvironment
from regista.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

# `timeout -s KILL` yields 137 (128+SIGKILL) on both GNU coreutils and busybox;
# GNU also documents 124 for its non-KILL path. Checked against the observed
# duration so a command legitimately exiting with these codes isn't misread.
_TIMEOUT_EXIT_CODES = frozenset({124, 137})


async def _docker(*args: str, timeout_s: float = 60.0) -> tuple[int | None, str, str]:
    """Run one docker CLI command; kill the whole client group on timeout."""
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except asyncio.TimeoutError:  # noqa: UP041 — 3.10 compat
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = await process.communicate()
        return None, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    return process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class ContainerEnvironment(LocalEnvironment):
    """A workspace whose commands run inside a Docker container.

    ``workspace`` (host directory) is bind-mounted at ``container_workdir``;
    file operations act on the host side with LocalEnvironment's scoping,
    ``exec`` runs in the container. Use as an async context manager — the
    container lives for the duration of the ``async with`` block.
    """

    name = "container"

    def __init__(
        self,
        workspace: Path | str = ".",
        *,
        image: str = "python:3.12-slim",
        container_workdir: str = "/workspace",
    ) -> None:
        super().__init__(workspace)
        self.image = image
        self.container_workdir = container_workdir
        self._container_id: str | None = None

    async def __aenter__(self) -> ContainerEnvironment:
        code, out, err = await _docker(
            "run",
            "-d",
            "--rm",
            "-v",
            f"{self.workspace}:{self.container_workdir}",
            "-w",
            self.container_workdir,
            self.image,
            "sleep",
            "2147483647",
            timeout_s=300.0,  # first use may pull the image
        )
        if code != 0:
            raise ConfigurationError(
                f"could not start container from image '{self.image}': {err.strip() or out.strip()}"
            )
        self._container_id = out.strip()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._container_id is not None:
            await _docker("kill", self._container_id, timeout_s=30.0)
            self._container_id = None

    async def exec(self, command: str, *, timeout_s: float = 60.0) -> ExecResult:
        """Run a shell command inside the container, cwd = the workspace mount.

        The container's own (image-provided) environment applies — nothing
        from the host env is passed in. Timeout is enforced by the container's
        ``timeout`` binary, with a client-side kill as backstop.
        """
        if self._container_id is None:
            raise ConfigurationError(
                "ContainerEnvironment is not running — use it as an async context manager"
            )
        started = time.monotonic()
        code, stdout, stderr = await _docker(
            "exec",
            "-w",
            self.container_workdir,
            self._container_id,
            "timeout",
            "-s",
            "KILL",
            f"{timeout_s:g}",
            "sh",
            "-c",
            command,
            timeout_s=timeout_s + 10.0,  # backstop: in-container timeout fires first
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        timed_out = code is None or (
            code in _TIMEOUT_EXIT_CODES and duration_ms >= timeout_s * 1000
        )
        return ExecResult(
            exit_code=code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
