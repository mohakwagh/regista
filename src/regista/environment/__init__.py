"""The execution environment primitive: where tool effects happen."""

from regista.environment.base import Environment, ExecResult
from regista.environment.local import LocalEnvironment

__all__ = ["Environment", "ExecResult", "LocalEnvironment"]
