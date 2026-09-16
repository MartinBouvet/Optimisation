"""Turn optional HotpotQA records into the same message format as crafted cases."""

from __future__ import annotations

from optimart.data.schema import ExampleRecord
from optimart.eval.cases import EvalCase


def record_to_case(record: ExampleRecord) -> EvalCase:
    supports = [seg.text for seg in record.segments if seg.y_hard >= 0.5]
    distractors = [seg.text for seg in record.segments if seg.y_hard < 0.5]
    return EvalCase(
        id=record.id,
        question=record.query,
        answers=list(record.gold_answers),
        supports=supports,
        distractors=distractors,
        place="mid",
    )


def load_hotpot_cases(max_examples: int, skip: int = 0) -> list[EvalCase]:
    from optimart.data.builders import iter_hotpotqa
    from optimart.data.segmentation import TokenCounter

    take = skip + max_examples
    records = list(
        iter_hotpotqa(
            split="validation",
            counter=TokenCounter(),
            max_segment_tokens=96,
            max_query_tokens=64,
            max_examples=take,
            min_segment_chars=12,
            seed=42,
        )
    )
    return [record_to_case(record) for record in records[skip : skip + max_examples]]
