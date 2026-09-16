"""Hard constraints: never drop system instructions or query-linked entities."""

from __future__ import annotations

import re

PROTECTED_ROLES = frozenset({"system", "developer"})

_QUESTION_SPLIT = re.compile(
    r"\b(?:question|query|consigne)\s*:\s*",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b\S+@\S+\.\S+\b")
# Lightweight proper-name detector (no spaCy): "New York", "Marie Curie".
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")
_STOP = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "If",
        "When",
        "What",
        "Who",
        "Where",
        "Why",
        "How",
        "Use",
        "Using",
        "Context",
        "Question",
        "Answer",
        "Please",
        "You",
        "Your",
        "Document",
        "Documents",
    }
)


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def is_multimodal(message: dict) -> bool:
    return isinstance(message.get("content"), list)


def proper_names(text: str) -> set[str]:
    found: set[str] = set()
    for match in _PROPER.finditer(text or ""):
        name = match.group(1)
        if name in _STOP:
            continue
        found.add(name)
    return found


def is_protected_segment(text: str, *, role: str, query: str) -> bool:
    if role in PROTECTED_ROLES:
        return True
    if _CODE_FENCE.search(text):
        return True
    if _URL.search(text) or _EMAIL.search(text):
        return True
    query_names = proper_names(query)
    if query_names and (query_names & proper_names(text)):
        return True
    return False


def extract_query(last_user_text: str, max_query_chars: int = 600) -> tuple[str, str]:
    """Return (query, leftover_context_in_same_message).

    Short last user turns are the query. Long RAG dumps keep a trailing
    Question: block, or the last two sentences, as the query.
    """
    text = (last_user_text or "").strip()
    if not text:
        return "", ""
    parts = _QUESTION_SPLIT.split(text)
    if len(parts) >= 2 and parts[-1].strip() and parts[0].strip():
        return parts[-1].strip(), parts[0].strip()
    if len(text) <= max_query_chars:
        return text, ""
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    if len(sentences) <= 2:
        return text, ""
    query = " ".join(sentences[-2:])
    context = " ".join(sentences[:-2])
    return query, context
