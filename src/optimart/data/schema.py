"""JSONL schema for scored context-pruning examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SegmentRecord:
    """One candidate span in a (query, context) example."""

    idx: int
    text: str
    n_tokens: int
    y_hard: float
    y_target: float
    paragraph_id: int
    source: str
    pmi: float | None = None
    is_protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SegmentRecord:
        return cls(
            idx=int(raw["idx"]),
            text=str(raw["text"]),
            n_tokens=int(raw["n_tokens"]),
            y_hard=float(raw["y_hard"]),
            y_target=float(raw["y_target"]),
            paragraph_id=int(raw.get("paragraph_id", 0)),
            source=str(raw.get("source", "unknown")),
            pmi=raw.get("pmi"),
            is_protected=bool(raw.get("is_protected", False)),
        )


@dataclass(slots=True)
class ExampleRecord:
    """One training / eval item after segmentation and labeling."""

    id: str
    split: str
    dataset: str
    query: str
    gold_answers: list[str]
    segments: list[SegmentRecord] = field(default_factory=list)

    @property
    def n_tokens_total(self) -> int:
        return int(sum(s.n_tokens for s in self.segments))

    @property
    def n_tokens_oracle(self) -> int:
        return int(sum(s.n_tokens for s in self.segments if s.y_hard >= 0.5))

    @property
    def n_positive(self) -> int:
        return int(sum(1 for s in self.segments if s.y_hard >= 0.5))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "dataset": self.dataset,
            "query": self.query,
            "gold_answers": self.gold_answers,
            "n_tokens_total": self.n_tokens_total,
            "n_tokens_oracle": self.n_tokens_oracle,
            "n_positive": self.n_positive,
            "oracle_keep_ratio": (
                self.n_tokens_oracle / self.n_tokens_total if self.n_tokens_total else 0.0
            ),
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExampleRecord:
        return cls(
            id=str(raw["id"]),
            split=str(raw.get("split", "train")),
            dataset=str(raw.get("dataset", "unknown")),
            query=str(raw["query"]),
            gold_answers=list(raw.get("gold_answers") or []),
            segments=[SegmentRecord.from_dict(item) for item in raw.get("segments") or []],
        )
