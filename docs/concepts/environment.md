# Execution environment

**Where effects happen.** Module: `regista/environment/`.

The `Environment` protocol owns file operations and command execution, pinned to a
workspace root:

```python
class Environment(Protocol):
    name: str
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def list_dir(self, path: str = ".") -> list[str]: ...
    async def glob(self, pattern: str) -> list[str]: ...
    async def exec(self, command: str, *, timeout_s: float = 60.0) -> ExecResult: ...
```

`LocalEnvironment(workspace)` is the v0.1 implementation:

- Every path resolves against the workspace root — `..`, absolute paths, and symlinks
  pointing outside all raise `WorkspaceViolation` (which the registry converts to an error
  tool result when the path came from the model).
- `exec` runs with an **allowlisted environment** (`PATH`, `HOME`, `LANG`, …) so shell
  commands never inherit your API keys, and a hard timeout that kills the process.

## The boundary that pays for itself

Tools define *what* the model sees; the environment defines *where* effects happen. Because
built-in tools only act through this protocol, a `ContainerEnvironment` (roadmap) is a
drop-in constructor swap — no tool schemas change, no traces change shape.

## Honesty note

This is **scoping, not sandboxing**: an allowed shell command can still do anything your
user account can do. Pair `shell` with an asking [policy](policy.md), and run untrusted
tasks in a container. `SECURITY.md` has the full threat model.
