"""LLM provider adapters. See base.py for the protocol."""

from regista.providers.anthropic import AnthropicProvider
from regista.providers.base import ModelRequest, ModelResponse, Provider
from regista.providers.fake import FakeProvider, text_response, tool_use_response

__all__ = [
    "AnthropicProvider",
    "FakeProvider",
    "ModelRequest",
    "ModelResponse",
    "Provider",
    "text_response",
    "tool_use_response",
]
