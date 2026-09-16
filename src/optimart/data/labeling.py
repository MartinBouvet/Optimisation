"""Mix extractive gold labels with optional offline PMI."""

from __future__ import annotations

import math

from optimart.data.pmi import GPT2PMITeacher
from optimart.data.schema import ExampleRecord


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mix_targets(
    record: ExampleRecord,
    *,
    alpha: float,
    pmi_clip: tuple[float, float],
    teacher: GPT2PMITeacher | None,
) -> ExampleRecord:
    """Set y_target = α y_hard + (1-α) σ(clip(PMI)).

    When no teacher is provided, y_target == y_hard. PMI is min-max scaled
    per example after clipping so a few extreme sentences cannot dominate.
    """
    if teacher is None or not record.segments:
        for seg in record.segments:
            seg.y_target = seg.y_hard
        return record

    lo, hi = pmi_clip
    raw: list[float] = []
    for seg in record.segments:
        value = teacher.pmi(record.query, seg.text)
        clipped = min(max(value, lo), hi)
        seg.pmi = clipped
        raw.append(clipped)

    vmin, vmax = min(raw), max(raw)
    span = vmax - vmin
    for seg, value in zip(record.segments, raw):
        if span < 1e-6:
            pmi_norm = 0.5
        else:
            pmi_norm = (value - vmin) / span
        # Extra logistic squash keeps mixed labels strictly in (0, 1).
        soft = _sigmoid(4.0 * (pmi_norm - 0.5))
        seg.y_target = alpha * seg.y_hard + (1.0 - alpha) * soft
    return record
