"""The execution environment primitive: where tool effects happen."""

from regista.environment.base import Environment, ExecResult
from regista.environment.container import ContainerEnvironment
from regista.environment.local import LocalEnvironment

__all__ = ["ContainerEnvironment", "Environment", "ExecResult", "LocalEnvironment"]
