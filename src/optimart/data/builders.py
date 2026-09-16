"""Dataset builders: HotpotQA supporting facts + SQuAD with packed distractors."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

from optimart.data.schema import ExampleRecord, SegmentRecord
from optimart.data.segmentation import TokenCounter, split_sentences


def _records_from_paragraphs(
    *,
    example_id: str,
    split: str,
    dataset: str,
    query: str,
    gold_answers: list[str],
    paragraphs: list[tuple[int, list[tuple[str, bool]], str]],
    counter: TokenCounter,
    max_segment_tokens: int,
    max_query_tokens: int,
) -> ExampleRecord:
    query = counter.truncate(query.strip(), max_query_tokens)
    segments: list[SegmentRecord] = []
    idx = 0
    for paragraph_id, sentences, source in paragraphs:
        for text, is_pos in sentences:
            clipped = counter.truncate(text.strip(), max_segment_tokens)
            if not clipped:
                continue
            y_hard = 1.0 if is_pos else 0.0
            segments.append(
                SegmentRecord(
                    idx=idx,
                    text=clipped,
                    n_tokens=counter.count(clipped),
                    y_hard=y_hard,
                    y_target=y_hard,
                    paragraph_id=paragraph_id,
                    source=source,
                )
            )
            idx += 1
    return ExampleRecord(
        id=example_id,
        split=split,
        dataset=dataset,
        query=query,
        gold_answers=gold_answers,
        segments=segments,
    )


def _hotpot_pairs(field: Any, key_a: str, key_b: str) -> list[tuple[Any, Any]]:
    """Accept both dict-of-lists (legacy) and list-of-dicts (Hub parquet)."""
    if isinstance(field, dict):
        return list(zip(field[key_a], field[key_b]))
    return [(item[key_a], item[key_b]) for item in field]


def iter_hotpotqa(
    *,
    split: str,
    counter: TokenCounter,
    max_segment_tokens: int,
    max_query_tokens: int,
    max_examples: int | None,
    min_segment_chars: int,
    seed: int,
    hf_id: str = "hotpotqa/hotpot_qa",
    config: str = "distractor",
) -> Iterator[ExampleRecord]:
    rng = random.Random(seed)
    ds = load_dataset(hf_id, config, split=split)
    n = len(ds) if max_examples is None else min(max_examples, len(ds))
    for i in range(n):
        row: dict[str, Any] = ds[i]
        support = {
            (str(title), int(sent_id))
            for title, sent_id in _hotpot_pairs(row["supporting_facts"], "title", "sent_id")
        }
        packed: list[tuple[int, list[tuple[str, bool]], str]] = []
        for p_id, (title, sents) in enumerate(_hotpot_pairs(row["context"], "title", "sentences")):
            labeled: list[tuple[str, bool]] = []
            for s_id, sent in enumerate(sents):
                text = str(sent).strip()
                if len(text) < min_segment_chars:
                    continue
                is_pos = (title, s_id) in support
                labeled.append((text, is_pos))
            if labeled:
                source = "gold" if any(flag for _, flag in labeled) else "distractor"
                packed.append((p_id, labeled, source))
        rng.shuffle(packed)
        record = _records_from_paragraphs(
            example_id=f"hotpotqa_{split}_{i}",
            split=split,
            dataset="hotpotqa",
            query=row["question"],
            gold_answers=[str(row["answer"])],
            paragraphs=packed,
            counter=counter,
            max_segment_tokens=max_segment_tokens,
            max_query_tokens=max_query_tokens,
        )
        if record.n_positive == 0 or record.n_tokens_total == 0:
            continue
        yield record


def _squad_sentence_labels(context: str, answer_starts: list[int], answers: list[str], min_chars: int) -> list[tuple[str, bool]]:
    spans = split_sentences(context, min_chars=min_chars)
    answer_ranges = [
        (start, start + len(ans))
        for start, ans in zip(answer_starts, answers)
        if ans
    ]
    labeled: list[tuple[str, bool]] = []
    for span in spans:
        is_pos = any(span.overlaps(a0, a1) for a0, a1 in answer_ranges)
        labeled.append((span.text, is_pos))
    if labeled:
        return labeled
    fallback = context.strip()
    is_pos = any(ans and ans in fallback for ans in answers)
    return [(fallback, is_pos)] if fallback else []


def iter_squad_packed(
    *,
    split: str,
    counter: TokenCounter,
    max_segment_tokens: int,
    max_query_tokens: int,
    max_examples: int | None,
    min_segment_chars: int,
    n_distractors: int,
    seed: int,
    hf_id: str = "squad",
) -> Iterator[ExampleRecord]:
    """Pack gold paragraph + random other paragraphs so pruning is non-trivial.

    Raw SQuAD contexts are short and almost entirely relevant. Without packing,
    a pruner has nothing to cut. Paragraph order is shuffled so position cannot
    become a shortcut.
    """
    rng = random.Random(seed + 17)
    ds = load_dataset(hf_id, split=split)
    contexts = [row["context"] for row in ds]
    n = len(ds) if max_examples is None else min(max_examples, len(ds))
    for i in range(n):
        row = ds[i]
        gold_sents = _squad_sentence_labels(
            row["context"],
            list(row["answers"]["answer_start"]),
            list(row["answers"]["text"]),
            min_segment_chars,
        )
        if not any(flag for _, flag in gold_sents):
            continue
        distractor_ids: list[int] = []
        attempts = 0
        while len(distractor_ids) < n_distractors and attempts < n_distractors * 8:
            attempts += 1
            j = rng.randrange(len(contexts))
            if j == i or j in distractor_ids:
                continue
            if contexts[j] == row["context"]:
                continue
            distractor_ids.append(j)
        paragraphs: list[tuple[int, list[tuple[str, bool]], str]] = [
            (0, gold_sents, "gold"),
        ]
        for k, j in enumerate(distractor_ids, start=1):
            spans = split_sentences(contexts[j], min_chars=min_segment_chars)
            sents = [(s.text, False) for s in spans] or [(contexts[j].strip(), False)]
            paragraphs.append((k, sents, "distractor"))
        rng.shuffle(paragraphs)
        record = _records_from_paragraphs(
            example_id=f"squad_{split}_{row['id']}",
            split=split,
            dataset="squad_packed",
            query=row["question"],
            gold_answers=list(dict.fromkeys(row["answers"]["text"])),
            paragraphs=paragraphs,
            counter=counter,
            max_segment_tokens=max_segment_tokens,
            max_query_tokens=max_query_tokens,
        )
        if record.n_positive == 0 or record.n_tokens_total == 0:
            continue
        yield record
