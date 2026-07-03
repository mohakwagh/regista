"""The execution environment: *where* tool effects happen.

Tools define what a capability is (schema + semantics); the environment
defines where its effects land. Built-in tools never touch the OS directly —
they call through this protocol, so swapping ``LocalEnvironment`` for a
container backend changes nothing about what the model sees (ARCHITECTURE.md
boundary rule: tools = what, environment = where).

Paths are workspace-relative strings on the tool side; each implementation
pins them to its workspace root and raises WorkspaceViolation on escapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one command execution.

    ``exit_code`` is None only when the process was killed before reporting
    one (e.g. on timeout, platform-dependent).
    """

    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@runtime_checkable
class Environment(Protocol):
    """Anything that can host file operations and command execution.

    All methods are async so remote/container backends are drop-in; local
    implementations delegate blocking I/O to threads.
    """

    @property
    def name(self) -> str: ...

    async def read_file(self, path: str) -> str: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def list_dir(self, path: str = ".") -> list[str]: ...

    async def glob(self, pattern: str) -> list[str]: ...

    async def exec(self, command: str, *, timeout_s: float = 60.0) -> ExecResult: ...
