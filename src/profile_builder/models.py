"""OpenAI model registry: tier → model id and list prices used for cost estimates.

Prices are USD per 1M tokens (input, cached input, output) as published on
https://developers.openai.com/api/docs/pricing (checked 2026-10). They only feed the
cost *estimate* shown to the user and the optional budget cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Tier = Literal["fast", "quality"]

MODEL_TIERS: dict[Tier, str] = {
    "fast": "gpt-5-mini",
    "quality": "gpt-5.4",
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class Price:
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float


PRICES: dict[str, Price] = {
    "gpt-5-nano": Price(0.05, 0.005, 0.40),
    "gpt-5-mini": Price(0.25, 0.025, 2.00),
    "gpt-5": Price(1.25, 0.125, 10.00),
    "gpt-5.1": Price(1.25, 0.125, 10.00),
    "gpt-5.2": Price(1.75, 0.175, 14.00),
    "gpt-5.4-nano": Price(0.20, 0.02, 1.25),
    "gpt-5.4-mini": Price(0.75, 0.075, 4.50),
    "gpt-5.4": Price(2.50, 0.25, 15.00),
    "gpt-5.5": Price(5.00, 0.50, 30.00),
    "gpt-4.1-mini": Price(0.40, 0.10, 1.60),
    "text-embedding-3-small": Price(0.02, 0.02, 0.0),
    "text-embedding-3-large": Price(0.13, 0.13, 0.0),
}


def price_for(model_id: str) -> Price | None:
    """Exact match first, then longest known prefix (e.g. dated snapshots)."""
    if model_id in PRICES:
        return PRICES[model_id]
    best = None
    for known, price in PRICES.items():
        if model_id.startswith(known) and (best is None or len(known) > len(best[0])):
            best = (known, price)
    return best[1] if best else None


def estimate_cost_usd(model_id: str, usage: dict[str, Any] | None) -> float:
    """Estimate the cost of one model call from LangChain `usage_metadata`."""
    if not usage:
        return 0.0
    price = price_for(model_id)
    if price is None:
        return 0.0
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int((usage.get("input_token_details") or {}).get("cache_read") or 0)
    uncached = max(input_tokens - cached, 0)
    return (
        uncached * price.input_per_m
        + cached * price.cached_input_per_m
        + output_tokens * price.output_per_m
    ) / 1_000_000
