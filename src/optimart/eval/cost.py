"""Input-token cost model. Prices: OpenAI public list, September 2026."""

from __future__ import annotations

# USD per 1M input tokens. Source: https://developers.openai.com/api/docs/models
INPUT_USD_PER_MILLION = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
}


def usd_for_requests(avg_input_tokens: float, n_requests: int, usd_per_million: float) -> float:
    return (avg_input_tokens * n_requests * usd_per_million) / 1_000_000.0


def cost_table(avg_tokens_by_arm: dict[str, float], n_requests: int = 100_000) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    full = avg_tokens_by_arm.get("full") or 0.0
    for model, price in INPUT_USD_PER_MILLION.items():
        model_row: dict[str, float] = {}
        full_cost = usd_for_requests(full, n_requests, price)
        for arm, tokens in avg_tokens_by_arm.items():
            cost = usd_for_requests(tokens, n_requests, price)
            model_row[arm] = round(cost, 4)
            model_row[f"{arm}_saved_vs_full"] = round(full_cost - cost, 4)
        out[model] = model_row
    return out
