"""PyTorch dataset over Optimart JSONL records."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from optimart.data.schema import ExampleRecord, SegmentRecord


@dataclass
class ScorerBatch:
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    y_hard: Tensor
    y_target: Tensor
    lengths: Tensor
    mask: Tensor

    def to(self, device: torch.device) -> ScorerBatch:
        return ScorerBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            token_type_ids=self.token_type_ids.to(device),
            y_hard=self.y_hard.to(device),
            y_target=self.y_target.to(device),
            lengths=self.lengths.to(device),
            mask=self.mask.to(device),
        )


def load_jsonl(paths: Sequence[str | Path]) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(ExampleRecord.from_dict(json.loads(line)))
    return records


def subsample_segments(
    segments: list[SegmentRecord],
    max_segments: int,
    rng: random.Random,
) -> list[SegmentRecord]:
    """Keep every positive; fill the rest with random negatives."""
    if max_segments <= 0 or len(segments) <= max_segments:
        return list(segments)
    positives = [item for item in segments if item.y_hard >= 0.5]
    negatives = [item for item in segments if item.y_hard < 0.5]
    if len(positives) >= max_segments:
        chosen = positives[:max_segments]
    else:
        n_neg = max_segments - len(positives)
        chosen_neg = negatives[:]
        rng.shuffle(chosen_neg)
        chosen = positives + chosen_neg[:n_neg]
    rng.shuffle(chosen)
    return chosen


class JsonlPruneDataset(Dataset):
    def __init__(
        self,
        records: list[ExampleRecord],
        tokenizer: Any,
        max_seq_length: int = 160,
        max_segments: int = 24,
        seed: int = 42,
    ) -> None:
        self.records = [item for item in records if item.segments and item.n_positive > 0]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_segments = max_segments
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.records[index]
        segments = subsample_segments(example.segments, self.max_segments, self.rng)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        token_type_ids: list[list[int]] = []
        y_hard: list[float] = []
        y_target: list[float] = []
        lengths: list[int] = []
        for segment in segments:
            encoded = self.tokenizer(
                example.query,
                segment.text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_token_type_ids=True,
            )
            input_ids.append(list(encoded["input_ids"]))
            attention_mask.append(list(encoded["attention_mask"]))
            token_type_ids.append(list(encoded.get("token_type_ids") or [0] * len(encoded["input_ids"])))
            y_hard.append(float(segment.y_hard))
            y_target.append(float(segment.y_target))
            lengths.append(int(segment.n_tokens))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "y_hard": y_hard,
            "y_target": y_target,
            "lengths": lengths,
        }


def collate_scorer_batch(batch: list[dict[str, Any]], pad_id: int = 0) -> ScorerBatch:
    n_batch = len(batch)
    n_segments = max(len(item["input_ids"]) for item in batch)
    max_len = max(len(ids) for item in batch for ids in item["input_ids"])
    input_ids = torch.full((n_batch, n_segments, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(n_batch, n_segments, max_len, dtype=torch.long)
    token_type_ids = torch.zeros(n_batch, n_segments, max_len, dtype=torch.long)
    y_hard = torch.zeros(n_batch, n_segments)
    y_target = torch.zeros(n_batch, n_segments)
    lengths = torch.zeros(n_batch, n_segments)
    mask = torch.zeros(n_batch, n_segments)
    for i, item in enumerate(batch):
        for j, tokens in enumerate(item["input_ids"]):
            n_tok = len(tokens)
            input_ids[i, j, :n_tok] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, j, :n_tok] = torch.tensor(item["attention_mask"][j], dtype=torch.long)
            types = item["token_type_ids"][j]
            token_type_ids[i, j, : len(types)] = torch.tensor(types, dtype=torch.long)
            y_hard[i, j] = item["y_hard"][j]
            y_target[i, j] = item["y_target"][j]
            lengths[i, j] = item["lengths"][j]
            mask[i, j] = 1.0
    return ScorerBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        y_hard=y_hard,
        y_target=y_target,
        lengths=lengths,
        mask=mask,
    )
