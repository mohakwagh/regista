"""The LLM boundary: one protocol, four implementations.

An adapter's whole job is normalization — wire format, tool-call shape, usage
extraction, retries. Because ``ReplayProvider`` implements this same protocol,
deterministic replay needs no special support anywhere else in the harness
(ARCHITECTURE.md §6, §7).

Streaming lands in v0.1 step 12; the protocol grows a ``stream()`` method then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from regista.trace.events import canonical_hash
from regista.types import Message, StopReason, ToolSpec, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from regista.streaming import ProviderDelta


class ModelRequest(BaseModel):
    """A fully-assembled, provider-neutral request — everything the model will see."""

    model_config = ConfigDict(frozen=True)

    model: str
    system: str | None = None
    messages: list[Message]
    tools: list[ToolSpec] = []
    max_tokens: int = 8192
    params: dict[str, Any] = {}
    """Provider-specific passthrough, e.g. ``{"thinking": {"type": "adaptive"}}``."""

    def request_hash(self) -> str:
        """The replay divergence detector: hash of everything deterministic.

        The full request is included — if any input that shapes the request
        changes (a message, a tool schema, a sampling param), the hash changes.
        """
        return canonical_hash(self.model_dump(mode="json"))


class ModelResponse(BaseModel):
    """A normalized response. ``raw`` is the provider's escape hatch and
    ``replayed`` marks responses served from a recording — both are excluded
    from trace serialization so a replay's payload stays byte-identical to
    the original's."""

    model_config = ConfigDict(frozen=True)

    message: Message
    stop_reason: StopReason
    usage: Usage = Usage()
    model: str = ""
    request_id: str | None = None
    raw: dict[str, Any] | None = None
    replayed: bool = False

    def model_dump_trace(self) -> dict[str, Any]:
        """The dict recorded in llm.response events (and served back by replay)."""
        return self.model_dump(mode="json", exclude={"raw", "replayed"})


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a ModelRequest into a ModelResponse.

    ``model`` is part of the protocol because the model is chosen where the
    provider is constructed — the loop only reads it to assemble requests.
    """

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ProviderDelta | ModelResponse]:
        """Yield deltas as they arrive, then the final ModelResponse last.

        Adapters without native streaming can wrap ``complete()`` with
        ``regista.streaming.synthetic_deltas``.
        """
        ...
