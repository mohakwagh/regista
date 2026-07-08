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
  commands never inherit your API keys, and a hard timeout that kills the whole process
  group — a forked child can't outlive the command.

## The boundary that pays for itself

Tools define *what* the model sees; the environment defines *where* effects happen. Because
built-in tools only act through this protocol, `ContainerEnvironment` is a drop-in
constructor swap — no tool schemas change, no traces change shape.

## ContainerEnvironment

`ContainerEnvironment` runs **commands inside a Docker container** while keeping file
operations on the host, workspace-pinned, through a bind mount:

```python
from regista.environment import ContainerEnvironment

async with ContainerEnvironment("./sandbox", image="python:3.12-slim") as env:
    agent = Agent(provider=..., instructions=..., tools=builtin_tools(env))
    await agent.run(task)
```

- Built on the `docker` CLI — no SDK dependency. The container lives for the `async with`
  block (`docker run -d` on enter, `docker kill` on exit; first use pulls the image).
- `exec` gets the container's environment only — **nothing from your host env is passed
  in** — and timeouts are enforced by the in-container `timeout` binary (busybox/coreutils)
  with a client-side kill as backstop.
- File ops keep `LocalEnvironment`'s exact semantics and scoping; the workspace directory
  is what the container sees at `/workspace`.

## Honesty note

`LocalEnvironment` is **scoping, not sandboxing**: an allowed shell command can still do
anything your user account can do. Pair `shell` with an asking [policy](policy.md) — or use
`ContainerEnvironment`, which makes command isolation a one-line change. The container is
the isolation boundary for *commands*; files you bind-mount are shared by design.
`SECURITY.md` has the full threat model.
