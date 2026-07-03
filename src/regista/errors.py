"""The regista exception hierarchy.

Kept deliberately small. Two things are conspicuously NOT exceptions:
a permission denial (the model receives an error tool_result and adapts) and
a budget/turn-limit stop (the loop ends with a stop reason). Exceptions are
for conditions the loop cannot turn into data the model can act on.
"""

from __future__ import annotations


class RegistaError(Exception):
    """Base class for all regista errors."""


class ConfigurationError(RegistaError):
    """The harness was assembled incorrectly (bad tool signature, missing handler, ...)."""


class ProviderError(RegistaError):
    """An LLM provider call failed after any retries.

    ``retryable`` records whether the underlying condition was transient
    (rate limit, connection drop) — informational by the time it propagates,
    since retries have already been exhausted.
    """

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ToolError(RegistaError):
    """A tool implementation raised or misbehaved.

    The loop normally converts tool failures into ``tool_result(is_error=True)``
    so the model can adapt; this exception is reserved for harness-side faults
    (unknown tool name, dispatch failure).
    """


class WorkspaceViolation(RegistaError):
    """A path or command tried to act outside the environment's workspace root.

    Raised by Environment implementations; when the offending path came from
    the model, the registry converts it into ``tool_result(is_error=True)``
    like any other tool failure. This is scoping, not sandboxing — see
    SECURITY.md for the honest threat model.
    """


class ReplayDivergence(RegistaError):
    """A replayed request no longer matches the recording (strict mode).

    ``diff`` is a human-readable structural diff pointing at the first
    message/turn that changed relative to the recorded request.
    """

    def __init__(self, message: str, *, seq: int, diff: str = "") -> None:
        super().__init__(message)
        self.seq = seq
        self.diff = diff
