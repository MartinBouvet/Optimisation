"""Full context vs head/tail truncation vs Optimart, same token budget B."""

from __future__ import annotations

import copy
import time
from typing import Literal

from optimart.prune.knapsack import KnapsackItem, solve_knapsack
from optimart.prune.pipeline import PruneEngine, PruneResult, WorkSegment, message_text

Policy = Literal["full", "trunc_head", "trunc_tail", "optimart"]


def run_arm(
    engine: PruneEngine,
    messages: list[dict],
    policy: Policy,
    budget_tokens: int | None = None,
) -> PruneResult:
    started = time.perf_counter()
    original = [copy.deepcopy(msg) for msg in messages]
    tokens_before = sum(engine.counter.count(message_text(msg)) for msg in original)
    query = engine._query_from_messages(original)
    segments = engine._segment_messages(original, query)
    if not segments:
        elapsed = (time.perf_counter() - started) * 1000
        return PruneResult(
            messages=original,
            query=query,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=tokens_before,
            kept=0,
            dropped=0,
            latency_ms=elapsed,
            method="noop",
            protected_overflow=False,
        )

    budget = engine.budget_for(segments, budget_tokens)
    if policy == "full":
        elapsed = (time.perf_counter() - started) * 1000
        return PruneResult(
            messages=original,
            query=query,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            budget=budget,
            kept=len(segments),
            dropped=0,
            latency_ms=elapsed,
            method="full",
            protected_overflow=False,
        )
    elif policy == "trunc_head":
        chosen = _take_unprotected_in_order(segments, budget, reverse=False)
        method = "trunc_head"
        overflow = False
    elif policy == "trunc_tail":
        chosen = _take_unprotected_in_order(segments, budget, reverse=True)
        method = "trunc_tail"
        overflow = False
    elif policy == "optimart":
        pruneable = [seg for seg in segments if not seg.protected]
        if pruneable:
            scores = engine.score_fn(query, [seg.text for seg in pruneable])
            for seg, score in zip(pruneable, scores):
                seg.score = float(score)
        items = [
            KnapsackItem(
                index=i,
                weight=seg.n_tokens,
                value=1.0 if seg.protected else max(seg.score, 1e-6),
                protected=seg.protected,
            )
            for i, seg in enumerate(segments)
        ]
        solution = solve_knapsack(items, budget)
        chosen = set(solution.chosen)
        method = "optimart"
        overflow = solution.protected_overflow
    else:
        raise ValueError(f"unknown policy {policy}")

    kept = [seg for i, seg in enumerate(segments) if i in chosen]
    rebuilt = engine.rebuild_messages(original, kept)
    tokens_after = sum(engine.counter.count(message_text(msg)) for msg in rebuilt)
    elapsed = (time.perf_counter() - started) * 1000
    return PruneResult(
        messages=rebuilt,
        query=query,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        budget=budget,
        kept=len(kept),
        dropped=len(segments) - len(kept),
        latency_ms=elapsed,
        method=method,
        protected_overflow=overflow,
    )


def _take_unprotected_in_order(segments: list[WorkSegment], budget: int, *, reverse: bool) -> set[int]:
    protected = [i for i, seg in enumerate(segments) if seg.protected]
    free = [i for i, seg in enumerate(segments) if not seg.protected]
    if reverse:
        free = list(reversed(free))
    chosen = set(protected)
    used = sum(segments[i].n_tokens for i in protected)
    for index in free:
        weight = segments[index].n_tokens
        if used + weight <= budget:
            chosen.add(index)
            used += weight
    return chosen
