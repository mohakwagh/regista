"""Ready-made permission policies.

Presets decide by tool *name* — they pair with the built-in toolset and treat
anything they don't recognize conservatively (deny or ask, never allow).
``Ask`` resolves to deny when no ask_handler is configured, so the safe
default costs nothing to adopt.

For anything richer, write a function: a policy is just
``(PermissionRequest) -> Allow | Deny | Ask``.
"""

from __future__ import annotations

from inspect import isawaitable
from typing import TYPE_CHECKING

from regista.policy.permissions import Allow, Ask, Deny, PermissionDecision, policy_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from regista.policy.permissions import PermissionPolicy, PermissionRequest

# Built-in tools whose only effect is reading the workspace. fetch is
# deliberately absent: a GET writes nothing locally but still reaches the
# network, and read_only promises no effects beyond reading.
_READ_ONLY_BUILTINS = frozenset({"read_file", "list_dir", "glob", "search_files"})

_WORKSPACE_BUILTINS = _READ_ONLY_BUILTINS | {"write_file"}


def read_only(*, allow: Iterable[str] = ()) -> Callable[[PermissionRequest], PermissionDecision]:
    """Permit only known read-only built-ins (plus ``allow`` extras); deny the rest.

    The strictest preset: right for exploration and summarization agents.
    """
    allowed = _READ_ONLY_BUILTINS | set(allow)

    def _read_only(request: PermissionRequest) -> PermissionDecision:
        if request.tool_name in allowed:
            return Allow()
        return Deny(reason=f"read_only policy: '{request.tool_name}' is not a known read-only tool")

    _read_only.policy_name = "read_only"  # type: ignore[attr-defined]
    return _read_only


def workspace() -> Callable[[PermissionRequest], PermissionDecision]:
    """Permit workspace-scoped file tools (reads and writes); ask for
    everything else — shell, fetch, and custom tools all escalate.

    The environment already pins file effects to the workspace root, so this
    preset draws the line at effects that can leave it.
    """

    def _workspace(request: PermissionRequest) -> PermissionDecision:
        if request.tool_name in _WORKSPACE_BUILTINS:
            return Allow()
        return Ask(prompt=f"Allow tool '{request.tool_name}' with input {request.tool_input!r}?")

    _workspace.policy_name = "workspace"  # type: ignore[attr-defined]
    return _workspace


def compose(*policies: PermissionPolicy) -> PermissionPolicy:
    """Consult policies in order; the first Deny or Ask decides.

    Allow requires unanimity, so composing only ever tightens: stack a
    targeted deny in front of a permissive preset, e.g.
    ``compose(deny_shell, workspace())``.
    """

    async def _composed(request: PermissionRequest) -> PermissionDecision:
        for policy in policies:
            decision = policy(request)
            if isawaitable(decision):
                decision = await decision
            if not isinstance(decision, Allow):
                return decision
        return Allow()

    names = ", ".join(policy_name(policy) for policy in policies)
    _composed.policy_name = f"compose({names})"  # type: ignore[attr-defined]
    return _composed
