"""The permission gate: a pure decision function consulted before every tool call.

A policy maps a PermissionRequest to Allow, Deny, or Ask. Deny is data, not an
exception — the model receives an error tool_result and adapts. Ask escalates
to the Agent's ask_handler; with no handler configured it resolves to deny,
so the harness never hangs waiting for input that can't arrive.

Presets beyond allow_all (read_only, workspace) land with the built-in tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class PermissionRequest:
    """Everything a policy (or a human, via Ask) needs to decide."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    turn: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str = "denied by policy"


@dataclass(frozen=True)
class Ask:
    prompt: str = ""


PermissionDecision: TypeAlias = Allow | Deny | Ask
PermissionPolicy: TypeAlias = Callable[
    [PermissionRequest], "PermissionDecision | Awaitable[PermissionDecision]"
]
AskHandler: TypeAlias = Callable[[PermissionRequest], Awaitable[bool]]


def policy_name(policy: object) -> str:
    """The name recorded in permission.decision events."""
    name = getattr(policy, "policy_name", None) or getattr(policy, "__name__", None)
    return str(name) if name else type(policy).__name__


def allow_all() -> Callable[[PermissionRequest], PermissionDecision]:
    """Permit every tool call. Fine for FakeProvider tests and trusted tools;
    prefer an asking policy for anything with side effects."""

    def _allow_all(request: PermissionRequest) -> PermissionDecision:
        return Allow()

    _allow_all.policy_name = "allow_all"  # type: ignore[attr-defined]
    return _allow_all
