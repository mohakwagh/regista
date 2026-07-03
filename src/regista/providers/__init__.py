"""LLM provider adapters. See base.py for the protocol."""

from regista.providers.anthropic import AnthropicProvider
from regista.providers.base import ModelRequest, ModelResponse, Provider
from regista.providers.fake import FakeProvider, text_response, tool_use_response
from regista.providers.openai_compat import OpenAICompatProvider
from regista.providers.replay import ReplayDivergenceWarning, ReplayProvider

__all__ = [
    "AnthropicProvider",
    "FakeProvider",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatProvider",
    "Provider",
    "ReplayDivergenceWarning",
    "ReplayProvider",
    "text_response",
    "tool_use_response",
]
