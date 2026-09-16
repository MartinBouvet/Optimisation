"""Extractive metrics for context pruning. No LLM answers, no invented scores."""

from __future__ import annotations

import re
from statistics import mean


_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    lowered = (text or "").lower().replace("’", "'")
    return _SPACE.sub(" ", _NON_ALNUM.sub(" ", lowered)).strip()


def answer_covered(answers: list[str], text: str) -> bool:
    haystack = normalize(text)
    if not haystack:
        return False
    for answer in answers:
        needle = normalize(answer)
        if needle and needle in haystack:
            return True
    return False


def support_recall(supports: list[str], text: str) -> float:
    if not supports:
        return 1.0
    hits = sum(1 for sent in supports if answer_covered([sent], text))
    return hits / len(supports)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "n": float(len(rows)),
        "tokens_before_mean": mean(row["tokens_before"] for row in rows),
        "tokens_after_mean": mean(row["tokens_after"] for row in rows),
        "reduction_mean": mean(row["reduction"] for row in rows),
        "answer_coverage": mean(row["answer_coverage"] for row in rows),
        "support_recall": mean(row["support_recall"] for row in rows),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "latency_ms_mean": mean(latencies),
    }
