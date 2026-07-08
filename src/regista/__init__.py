"""regista — the agent harness with a flight recorder.

The runtime layer that turns a stateless LLM API into an agent that can act,
recording every session as a structured, deterministically replayable trace.
See ARCHITECTURE.md for the full design.
"""

from regista._version import __version__
from regista.agent import Agent
from regista.context import ContextConfig
from regista.instructions import Instructions
from regista.replay import replay
from regista.session import RunResult, Session
from regista.skills import Skill
from regista.streaming import (
    RunCompleted,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnCompleted,
)
from regista.tools import Tool, tool
from regista.types import Message, Usage

__all__ = [
    "Agent",
    "ContextConfig",
    "Instructions",
    "Message",
    "RunCompleted",
    "RunResult",
    "Session",
    "Skill",
    "StreamEvent",
    "TextDelta",
    "ThinkingDelta",
    "Tool",
    "ToolCallFinished",
    "ToolCallStarted",
    "TurnCompleted",
    "Usage",
    "__version__",
    "replay",
    "tool",
]
