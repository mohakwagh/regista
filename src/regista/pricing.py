"""Cost computation from provider-reported usage.

Best-effort by design: prices drift, so the table is overridable per-Agent
(``price_overrides``) and an unknown model yields ``None`` — regista records
"unknown", it never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from regista.types import Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. Cache defaults follow Anthropic's ratios
    (reads ~0.1x input, writes ~1.25x input)."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None

    @property
    def cache_read(self) -> float:
        return (
            self.cache_read_per_mtok
            if self.cache_read_per_mtok is not None
            else self.input_per_mtok * 0.1
        )

    @property
    def cache_write(self) -> float:
        return (
            self.cache_write_per_mtok
            if self.cache_write_per_mtok is not None
            else self.input_per_mtok * 1.25
        )


# Best-effort snapshot (2026-07); override via Agent(price_overrides=...).
PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-5": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
}


def resolve_price(model: str, overrides: dict[str, ModelPrice] | None = None) -> ModelPrice | None:
    """Exact match first (overrides shadow the table), then longest prefix match
    so dated ids like ``claude-haiku-4-5-20251001`` resolve."""
    table = {**PRICES, **(overrides or {})}
    if model in table:
        return table[model]
    prefixes = [known for known in table if model.startswith(known)]
    if prefixes:
        return table[max(prefixes, key=len)]
    return None


def cost_usd(
    model: str, usage: Usage, overrides: dict[str, ModelPrice] | None = None
) -> float | None:
    price = resolve_price(model, overrides)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.output_tokens * price.output_per_mtok
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / 1_000_000
