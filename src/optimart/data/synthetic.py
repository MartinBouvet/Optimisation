"""Tiny in-memory corpus so training can be smoke-tested without HuggingFace downloads."""

from __future__ import annotations

from optimart.data.schema import ExampleRecord, SegmentRecord

_PAIRS: list[tuple[str, list[str], list[str]]] = [
    (
        "What is the capital of France?",
        ["Paris is the capital of France.", "The French government is based in Paris."],
        ["The Nile is the longest river in Africa.", "Tokyo is a city in Japan.", "Python is a programming language."],
    ),
    (
        "Who wrote Hamlet?",
        ["William Shakespeare wrote Hamlet around 1600.", "Hamlet is a tragedy by Shakespeare."],
        ["Beethoven composed nine symphonies.", "The Amazon rainforest is in South America.", "Mars has two moons."],
    ),
    (
        "What gas do plants absorb?",
        ["Plants absorb carbon dioxide during photosynthesis.", "CO2 is taken in by plant leaves."],
        ["Helium is used in balloons.", "Granite is an igneous rock.", "The Pacific is the largest ocean."],
    ),
    (
        "Where is Mount Everest?",
        ["Mount Everest is on the border of Nepal and China.", "Everest is the highest peak in the Himalayas."],
        ["Kilimanjaro is in Tanzania.", "The Sahara is a desert in Africa.", "Saturn has prominent rings."],
    ),
]


def build_synthetic_records(n_train: int, n_val: int) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    train = [_make_record(i, "train") for i in range(n_train)]
    val = [_make_record(i + 10_000, "validation") for i in range(n_val)]
    return train, val


def _make_record(index: int, split: str) -> ExampleRecord:
    query, positives, negatives = _PAIRS[index % len(_PAIRS)]
    extra_neg = negatives[(index % len(negatives)) :] + negatives[: (index % len(negatives))]
    segments: list[SegmentRecord] = []
    order = [(text, 1.0, "gold") for text in positives] + [(text, 0.0, "distractor") for text in extra_neg]
    # Rotate so position cannot become a shortcut.
    pivot = index % max(len(order), 1)
    order = order[pivot:] + order[:pivot]
    for idx, (text, y_hard, source) in enumerate(order):
        n_tokens = max(len(text.split()), 1)
        segments.append(
            SegmentRecord(
                idx=idx,
                text=text,
                n_tokens=n_tokens,
                y_hard=y_hard,
                y_target=0.85 * y_hard + 0.05,
                paragraph_id=0 if y_hard >= 0.5 else 1,
                source=source,
            )
        )
    return ExampleRecord(
        id=f"synthetic_{split}_{index}",
        split=split,
        dataset="synthetic",
        query=query,
        gold_answers=[positives[0]],
        segments=segments,
    )
