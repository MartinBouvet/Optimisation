"""0-1 knapsack under a token budget.

Maximize Σ x_i v_i  subject to Σ x_i ℓ_i ≤ B, x_i ∈ {0,1}.
Protected items are forced in; if they already exceed B they are still kept.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnapsackItem:
    index: int
    weight: int
    value: float
    protected: bool = False


@dataclass(frozen=True, slots=True)
class KnapsackSolution:
    chosen: frozenset[int]
    tokens_used: int
    protected_overflow: bool
    method: str


def solve_knapsack(items: list[KnapsackItem], budget: int) -> KnapsackSolution:
    if budget < 0:
        raise ValueError("budget must be >= 0")
    protected = [item for item in items if item.protected]
    free = [item for item in items if not item.protected]
    forced_weight = sum(max(int(item.weight), 0) for item in protected)
    chosen = {item.index for item in protected}
    remaining = budget - forced_weight
    overflow = remaining < 0
    if remaining <= 0:
        return KnapsackSolution(
            chosen=frozenset(chosen),
            tokens_used=forced_weight,
            protected_overflow=overflow,
            method="protected-only",
        )

    selectable: list[KnapsackItem] = []
    for item in free:
        weight = max(int(item.weight), 0)
        if weight == 0:
            chosen.add(item.index)
            continue
        selectable.append(
            KnapsackItem(index=item.index, weight=weight, value=max(float(item.value), 0.0), protected=False)
        )

    n_cells = len(selectable) * remaining
    if n_cells > 2_000_000:
        picked, extra = _greedy(selectable, remaining)
        chosen.update(picked)
        return KnapsackSolution(
            chosen=frozenset(chosen),
            tokens_used=forced_weight + extra,
            protected_overflow=False,
            method="greedy",
        )

    picked, extra = _dp(selectable, remaining)
    chosen.update(picked)
    return KnapsackSolution(
        chosen=frozenset(chosen),
        tokens_used=forced_weight + extra,
        protected_overflow=False,
        method="dp",
    )


def _dp(items: list[KnapsackItem], budget: int) -> tuple[set[int], int]:
    n = len(items)
    best = [0.0] * (budget + 1)
    take = [[False] * (budget + 1) for _ in range(n)]
    for i, item in enumerate(items):
        weight, value = item.weight, item.value
        for cap in range(budget, weight - 1, -1):
            candidate = best[cap - weight] + value
            if candidate > best[cap]:
                best[cap] = candidate
                take[i][cap] = True
    cap = max(range(budget + 1), key=lambda c: (best[c], -c))
    chosen: set[int] = set()
    tokens = 0
    for i in range(n - 1, -1, -1):
        if take[i][cap]:
            chosen.add(items[i].index)
            tokens += items[i].weight
            cap -= items[i].weight
    return chosen, tokens


def _greedy(items: list[KnapsackItem], budget: int) -> tuple[set[int], int]:
    ordered = sorted(
        items,
        key=lambda item: (item.value / item.weight, item.value),
        reverse=True,
    )
    chosen: set[int] = set()
    used = 0
    for item in ordered:
        if used + item.weight <= budget:
            chosen.add(item.index)
            used += item.weight
    return chosen, used
