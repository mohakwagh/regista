"""regista — the agent harness with a flight recorder.

The runtime layer that turns a stateless LLM API into an agent that can act,
recording every session as a structured, deterministically replayable trace.
See ARCHITECTURE.md for the full design.
"""

from regista._version import __version__
from regista.agent import Agent
from regista.instructions import Instructions
from regista.session import RunResult, Session
from regista.tools import Tool, tool
from regista.types import Message, Usage

__all__ = [
    "Agent",
    "Instructions",
    "Message",
    "RunResult",
    "Session",
    "Tool",
    "Usage",
    "__version__",
    "tool",
]
