"""Built-in tools: files, shell, search, fetch.

Every effect goes through the Environment protocol — these functions never
touch the OS directly, which is exactly what makes a container backend a
drop-in swap. ``builtin_tools()`` is a factory rather than module-level tools
because each returned Tool closes over one environment instance.

All outputs are capped (``max_output_chars``) so a single ``cat`` of a huge
file can't blow the context window; the truncation marker tells the model how
much it didn't see.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import httpx

from regista.tools.registry import Tool, tool

if TYPE_CHECKING:
    from regista.environment.base import Environment

_MAX_SEARCH_MATCHES = 200


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [output truncated: {omitted} of {len(text)} characters omitted]"


def builtin_tools(
    environment: Environment,
    *,
    max_output_chars: int = 50_000,
    shell_timeout_s: float = 60.0,
    fetch_timeout_s: float = 30.0,
) -> list[Tool]:
    """The standard toolset, bound to one environment.

    Returns read_file, write_file, list_dir, glob, search_files, shell, and
    fetch. Read-only tools are ``parallel_safe``; write_file and shell are not.
    Pair with a policy — ``allow_all`` plus ``shell`` means the model can run
    anything your user account can.
    """

    @tool(parallel_safe=True)
    async def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Workspace-relative file path, e.g. "src/app.py".
        """
        return _truncate(await environment.read_file(path), max_output_chars)

    @tool
    async def write_file(path: str, content: str) -> str:
        """Write a text file in the workspace, creating parent directories
        and replacing any existing content.

        Args:
            path: Workspace-relative file path, e.g. "src/app.py".
            content: The full new file content.
        """
        await environment.write_file(path, content)
        return f"Wrote {len(content)} characters to {path}"

    @tool
    async def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace one exact string occurrence in a workspace file.

        Args:
            path: Workspace-relative file path, e.g. "src/app.py".
            old_string: The exact existing text to replace.
            new_string: The exact replacement text.
        """
        content = await environment.read_file(path)
        matches = content.count(old_string)
        if matches == 0:
            raise ValueError(f"{old_string!r} was not found in {path}")
        if matches > 1:
            raise ValueError(
                f"{old_string!r} occurs {matches} times in {path}; "
                "edit_file requires exactly one match"
            )
        updated = content.replace(old_string, new_string, 1)
        await environment.write_file(path, updated)
        return f"Edited {path}"

    @tool(parallel_safe=True)
    async def list_dir(path: str = ".") -> str:
        """List a workspace directory; names ending in "/" are directories.

        Args:
            path: Workspace-relative directory path; defaults to the root.
        """
        entries = await environment.list_dir(path)
        return "\n".join(entries) if entries else f"(empty directory: {path})"

    @tool(parallel_safe=True)
    async def glob(pattern: str) -> str:
        """Find workspace files matching a glob pattern.

        Args:
            pattern: A glob like "**/*.py" or "src/*.md", relative to the
                workspace root.
        """
        matches = await environment.glob(pattern)
        if not matches:
            return f"(no files match {pattern!r})"
        return _truncate("\n".join(matches), max_output_chars)

    @tool(parallel_safe=True)
    async def search_files(pattern: str, glob: str = "**/*") -> str:
        """Search workspace files for a regular expression; results are
        "path:line: text" lines.

        Args:
            pattern: A Python regular expression to search for.
            glob: Which files to search, as a glob pattern; defaults to all.
        """
        regex = re.compile(pattern)
        matches: list[str] = []
        for path in await environment.glob(glob):
            try:
                content = await environment.read_file(path)
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable files are not searchable
            for line_no, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{path}:{line_no}: {line.strip()}")
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        matches.append(f"... [stopped at {_MAX_SEARCH_MATCHES} matches]")
                        return _truncate("\n".join(matches), max_output_chars)
        if not matches:
            return f"(no matches for {pattern!r} in {glob!r})"
        return _truncate("\n".join(matches), max_output_chars)

    @tool
    async def shell(command: str) -> str:
        """Run a shell command in the workspace root and return its output.

        Args:
            command: The command line to execute.
        """
        result = await environment.exec(command, timeout_s=shell_timeout_s)
        parts = []
        if result.timed_out:
            parts.append(f"[command timed out after {shell_timeout_s:g}s and was killed]")
        parts.append(f"exit code: {result.exit_code}")
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        return _truncate("\n".join(parts), max_output_chars)

    @tool(parallel_safe=True)
    async def fetch(url: str) -> str:
        """Fetch a URL over HTTP(S) and return the response body as text.

        Args:
            url: An http:// or https:// URL.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"fetch only supports http(s) URLs, got {url!r}")
        async with httpx.AsyncClient(follow_redirects=True, timeout=fetch_timeout_s) as client:
            response = await client.get(url)
        body = _truncate(response.text, max_output_chars)
        return f"HTTP {response.status_code}\n\n{body}"

    return [read_file, write_file, edit_file, list_dir, glob, search_files, shell, fetch]
