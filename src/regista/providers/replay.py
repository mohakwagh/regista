"""Replay is just another provider.

ReplayProvider serves recorded llm.response payloads in seq order, verifying
each live request's hash against the recording before it answers. Because it
implements the same Provider protocol as the real adapters, the rest of the
harness — loop, policy, trace — runs completely unchanged during a replay;
that one seam is the whole trick (ARCHITECTURE.md §6).

Divergence modes:

- ``strict``  — raise ReplayDivergence with a structural diff (CI, regression)
- ``warn``    — emit ReplayDivergenceWarning, keep serving positionally
  (debugging: "show me what it did anyway")
- ``hybrid``  — fall through to a real ``fallback`` provider from the first
  divergence (or when the recording runs out) and stay live; the basis of
  Session.resume in v0.2
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from regista.errors import ConfigurationError, ReplayDivergence
from regista.providers.base import ModelResponse
from regista.streaming import synthetic_deltas

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from regista.providers.base import ModelRequest, Provider
    from regista.streaming import ProviderDelta
    from regista.trace.reader import Trace

ReplayMode = Literal["strict", "warn", "hybrid"]


class ReplayDivergenceWarning(UserWarning):
    """Raised as a warning in ``warn`` mode when a request no longer matches."""


def diff_requests(recorded: Any, live: Any, *, max_lines: int = 20) -> str:
    """A human-readable structural diff, pointing at what changed first."""
    lines: list[str] = []
    _walk(recorded, live, "request", lines, max_lines)
    if not lines:
        return "(payloads are structurally identical)"
    return "\n".join(lines)


def _walk(recorded: Any, live: Any, path: str, out: list[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if isinstance(recorded, dict) and isinstance(live, dict):
        for key in sorted(recorded.keys() | live.keys()):
            if key not in live:
                out.append(f"{path}.{key}: only in recording")
            elif key not in recorded:
                out.append(f"{path}.{key}: only in live request")
            else:
                _walk(recorded[key], live[key], f"{path}.{key}", out, limit)
    elif isinstance(recorded, list) and isinstance(live, list):
        if len(recorded) != len(live):
            out.append(f"{path}: recorded {len(recorded)} items, live has {len(live)}")
        for index, (a, b) in enumerate(zip(recorded, live, strict=False)):
            _walk(a, b, f"{path}[{index}]", out, limit)
    elif recorded != live:
        out.append(f"{path}: recorded {recorded!r} != live {live!r}")


class ReplayProvider:
    """Serves one recorded trace's LLM responses, hash-verified call by call."""

    name = "replay"

    def __init__(
        self,
        trace: Trace,
        *,
        mode: ReplayMode = "strict",
        fallback: Provider | None = None,
    ) -> None:
        if mode == "hybrid" and fallback is None:
            raise ConfigurationError("hybrid replay needs a fallback provider to fall through to")
        self.model = trace.start.model
        self._calls = trace.llm_calls()
        self._mode: ReplayMode = mode
        self._fallback = fallback
        self._index = 0
        self._live = False  # hybrid: set on first divergence, never unset

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._live and self._fallback is not None:
            return await self._fallback.complete(request)

        if self._index >= len(self._calls):
            if self._mode == "hybrid" and self._fallback is not None:
                self._live = True
                return await self._fallback.complete(request)
            raise ReplayDivergence(
                f"recording exhausted: live run wants call {self._index + 1}, "
                f"only {len(self._calls)} were recorded",
                seq=self._calls[-1][0].seq if self._calls else -1,
            )

        recorded_request, recorded_response = self._calls[self._index]
        if request.request_hash() != recorded_request.request_hash:
            diff = diff_requests(recorded_request.request, request.model_dump(mode="json"))
            message = (
                f"request {self._index + 1} diverged from the recording "
                f"(trace seq {recorded_request.seq})"
            )
            if self._mode == "strict":
                raise ReplayDivergence(f"{message}\n{diff}", seq=recorded_request.seq, diff=diff)
            if self._mode == "hybrid" and self._fallback is not None:
                self._live = True
                return await self._fallback.complete(request)
            warnings.warn(f"{message}\n{diff}", ReplayDivergenceWarning, stacklevel=2)

        self._index += 1
        replayed = ModelResponse.model_validate(recorded_response.response)
        return replayed.model_copy(update={"replayed": True})

    async def stream(self, request: ModelRequest) -> AsyncIterator[ProviderDelta | ModelResponse]:
        """Synthetic deltas from the recorded response — a replay can drive a
        live UI exactly like the original run did."""
        response = await self.complete(request)
        for delta in synthetic_deltas(response.message):
            yield delta
        yield response
