"""Segment → score → knapsack → rebuilt OpenAI messages."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from optimart.data.segmentation import TokenCounter, split_sentences
from optimart.prune.constraints import (
    PROTECTED_ROLES,
    extract_query,
    is_multimodal,
    is_protected_segment,
    message_text,
)
from optimart.prune.knapsack import KnapsackItem, solve_knapsack

ScoreFn = Callable[[str, list[str]], list[float]]


@dataclass(slots=True)
class WorkSegment:
    message_index: int
    role: str
    text: str
    n_tokens: int
    protected: bool
    score: float = 0.0


@dataclass
class PruneResult:
    messages: list[dict[str, Any]]
    query: str
    tokens_before: int
    tokens_after: int
    budget: int
    kept: int
    dropped: int
    latency_ms: float
    method: str
    protected_overflow: bool

    @property
    def reduction(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)

    def stats(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "budget": self.budget,
            "reduction": round(self.reduction, 4),
            "kept": self.kept,
            "dropped": self.dropped,
            "latency_ms": round(self.latency_ms, 2),
            "method": self.method,
            "protected_overflow": self.protected_overflow,
        }


def lexical_scores(query: str, segments: list[str]) -> list[float]:
    query_toks = set(query.lower().split())
    if not query_toks:
        return [0.0] * len(segments)
    scores: list[float] = []
    for text in segments:
        toks = set(text.lower().split())
        if not toks:
            scores.append(0.0)
            continue
        scores.append(len(query_toks & toks) / len(query_toks | toks))
    return scores


class PruneEngine:
    def __init__(
        self,
        score_fn: ScoreFn | None = None,
        *,
        budget_ratio: float = 0.45,
        max_query_chars: int = 400,
        min_segment_chars: int = 12,
        encoding: str = "cl100k_base",
    ) -> None:
        if not 0.0 < budget_ratio <= 1.0:
            raise ValueError("budget_ratio must be in (0, 1]")
        self.score_fn = score_fn or lexical_scores
        self.budget_ratio = budget_ratio
        self.max_query_chars = max_query_chars
        self.min_segment_chars = min_segment_chars
        self.counter = TokenCounter(encoding)

    def prune_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        budget_tokens: int | None = None,
    ) -> PruneResult:
        started = time.perf_counter()
        original = [copy.deepcopy(msg) for msg in messages]
        tokens_before = sum(self.counter.count(message_text(msg)) for msg in original)
        query = self._query_from_messages(original)
        segments = self._segment_messages(original, query)
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

        pruneable = [seg for seg in segments if not seg.protected]
        if pruneable:
            scores = self.score_fn(query, [seg.text for seg in pruneable])
            if len(scores) != len(pruneable):
                raise ValueError("score_fn returned the wrong number of scores")
            for seg, score in zip(pruneable, scores):
                seg.score = float(score)

        budget = self.budget_for(segments, budget_tokens)

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
        kept_segments = [seg for i, seg in enumerate(segments) if i in solution.chosen]
        rebuilt = self.rebuild_messages(original, kept_segments)
        tokens_after = sum(self.counter.count(message_text(msg)) for msg in rebuilt)
        elapsed = (time.perf_counter() - started) * 1000
        return PruneResult(
            messages=rebuilt,
            query=query,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            budget=budget,
            kept=len(kept_segments),
            dropped=len(segments) - len(kept_segments),
            latency_ms=elapsed,
            method=solution.method,
            protected_overflow=solution.protected_overflow,
        )

    def _query_from_messages(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                query, _ = extract_query(message_text(message), self.max_query_chars)
                return query
        return message_text(messages[-1]) if messages else ""

    def _segment_messages(self, messages: list[dict[str, Any]], query: str) -> list[WorkSegment]:
        last_user = max((i for i, msg in enumerate(messages) if msg.get("role") == "user"), default=None)
        segments: list[WorkSegment] = []
        for index, message in enumerate(messages):
            role = str(message.get("role") or "user")
            if is_multimodal(message):
                text = message_text(message)
                segments.append(self._make_segment(index, role, text, protected=True, query=query))
                continue
            text = message_text(message)
            if not text.strip():
                continue
            if index == last_user:
                query_part, context_part = extract_query(text, self.max_query_chars)
                if context_part:
                    context_spans = split_sentences(context_part, min_chars=self.min_segment_chars)
                    if not context_spans:
                        context_spans = []
                        if context_part.strip():
                            segments.append(
                                self._make_segment(index, role, context_part, protected=False, query=query)
                            )
                    for span in context_spans:
                        segments.append(self._make_segment(index, role, span.text, protected=False, query=query))
                    segments.append(self._make_segment(index, role, query_part, protected=True, query=query))
                    continue
                segments.append(self._make_segment(index, role, text, protected=True, query=query))
                continue
            spans = split_sentences(text, min_chars=self.min_segment_chars)
            if not spans:
                protected = role in PROTECTED_ROLES or is_protected_segment(text, role=role, query=query)
                segments.append(self._make_segment(index, role, text, protected=protected, query=query))
                continue
            for span in spans:
                protected = is_protected_segment(span.text, role=role, query=query)
                segments.append(self._make_segment(index, role, span.text, protected=protected, query=query))
        return [seg for seg in segments if seg.text.strip() and seg.n_tokens > 0]

    def budget_for(self, segments: list[WorkSegment], budget_tokens: int | None = None) -> int:
        unprotected = sum(seg.n_tokens for seg in segments if not seg.protected)
        protected = sum(seg.n_tokens for seg in segments if seg.protected)
        budget = budget_tokens if budget_tokens is not None else protected + int(self.budget_ratio * unprotected)
        return max(budget, protected)

    def rebuild_messages(
        self, original: list[dict[str, Any]], kept: list[WorkSegment]
    ) -> list[dict[str, Any]]:
        return self._rebuild_messages(original, kept)

    def _make_segment(
        self, message_index: int, role: str, text: str, *, protected: bool, query: str
    ) -> WorkSegment:
        if not protected:
            protected = is_protected_segment(text, role=role, query=query)
        return WorkSegment(
            message_index=message_index,
            role=role,
            text=text.strip(),
            n_tokens=max(self.counter.count(text.strip()), 1),
            protected=protected,
        )

    def _rebuild_messages(
        self, original: list[dict[str, Any]], kept: list[WorkSegment]
    ) -> list[dict[str, Any]]:
        grouped: dict[int, list[str]] = {}
        for seg in kept:
            grouped.setdefault(seg.message_index, []).append(seg.text)
        rebuilt: list[dict[str, Any]] = []
        for index, message in enumerate(original):
            if is_multimodal(message):
                rebuilt.append(message)
                continue
            parts = grouped.get(index)
            if not parts:
                continue
            new_message = copy.deepcopy(message)
            new_message["content"] = " ".join(parts)
            rebuilt.append(new_message)
        return rebuilt or original
