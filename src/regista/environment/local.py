"""LocalEnvironment: the host filesystem, scoped to one workspace directory.

Every path is resolved against the workspace root — symlinks included — and
anything that lands outside it raises WorkspaceViolation. Commands run with a
minimal environment (no inherited API keys) and a hard timeout. To be blunt
about the guarantee: this is scoping that keeps honest tools honest and
contains model mistakes, not a sandbox against adversarial code. SECURITY.md
has the full threat model; a container backend is the roadmap answer.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from regista.environment.base import ExecResult
from regista.errors import WorkspaceViolation

# Deliberate allowlist: enough for commands to run, nothing secret.
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR")


class LocalEnvironment:
    """File operations and command execution pinned to ``workspace``."""

    name = "local"

    def __init__(self, workspace: Path | str = ".") -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """Resolve a workspace-relative path, rejecting escapes (``..``,
        absolute paths, and symlinks pointing outside the root)."""
        candidate = (self.workspace / path).resolve()
        if candidate != self.workspace and not candidate.is_relative_to(self.workspace):
            raise WorkspaceViolation(
                f"path {path!r} resolves outside the workspace root {self.workspace}"
            )
        return candidate

    async def read_file(self, path: str) -> str:
        target = self.resolve(path)
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        target = self.resolve(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def list_dir(self, path: str = ".") -> list[str]:
        """Sorted entry names; directories carry a trailing ``/``."""
        target = self.resolve(path)

        def _list() -> list[str]:
            return sorted(
                entry.name + "/" if entry.is_dir() else entry.name for entry in target.iterdir()
            )

        return await asyncio.to_thread(_list)

    async def glob(self, pattern: str) -> list[str]:
        """Sorted workspace-relative file paths matching ``pattern``."""

        def _glob() -> list[str]:
            matches = []
            for match in self.workspace.glob(pattern):
                resolved = match.resolve()
                if resolved.is_file() and resolved.is_relative_to(self.workspace):
                    matches.append(match.relative_to(self.workspace).as_posix())
            return sorted(matches)

        return await asyncio.to_thread(_glob)

    async def exec(self, command: str, *, timeout_s: float = 60.0) -> ExecResult:
        started = time.monotonic()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ},
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:  # noqa: UP041 — not builtin TimeoutError until 3.11
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()
        return ExecResult(
            exit_code=process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            duration_ms=int((time.monotonic() - started) * 1000),
            timed_out=timed_out,
        )
