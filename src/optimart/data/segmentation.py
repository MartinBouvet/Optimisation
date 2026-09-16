"""Sentence segmentation with character offsets and OpenAI-compatible token counts."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence-ending punctuation or a blank line. Conservative: keeps
# abbreviations inside a span rather than over-segmenting RAG dumps.
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextSpan:
    start: int
    end: int
    text: str

    def overlaps(self, other_start: int, other_end: int) -> bool:
        return self.start < other_end and other_start < self.end


def split_sentences(text: str, min_chars: int = 12) -> list[TextSpan]:
    """Return ordered sentence spans with offsets into `text`."""
    spans: list[TextSpan] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        inner = raw.strip()
        if len(inner) < min_chars:
            continue
        lead = len(raw) - len(raw.lstrip())
        start = match.start() + lead
        end = start + len(inner)
        spans.append(TextSpan(start=start, end=end, text=inner))
    if spans:
        return spans
    stripped = text.strip()
    if len(stripped) >= min_chars:
        start = text.find(stripped)
        return [TextSpan(start=start, end=start + len(stripped), text=stripped)]
    return []


class TokenCounter:
    """OpenAI-compatible token counts, with a whitespace fallback if tiktoken is missing."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is None:
            return max(1, len(text.split()))
        return len(self._enc.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        if self._enc is None:
            words = text.split()
            return " ".join(words[:max_tokens])
        ids = self._enc.encode(text, disallowed_special=())
        if len(ids) <= max_tokens:
            return text
        return self._enc.decode(ids[:max_tokens])
