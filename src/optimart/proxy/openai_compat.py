"""OpenAI Chat Completions helpers for the Optimart proxy."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from optimart.prune.pipeline import PruneResult


def dry_run_completion(body: dict[str, Any], result: PruneResult) -> dict[str, Any]:
    preview = "\n\n".join(
        f"[{msg.get('role')}] {msg.get('content')}"
        for msg in result.messages
        if isinstance(msg.get("content"), str)
    )
    summary = (
        f"Optimart dry-run: {result.tokens_before} → {result.tokens_after} tokens "
        f"({result.reduction:.0%} cut), {result.dropped} segments dropped."
    )
    return {
        "id": f"chatcmpl-optimart-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": body.get("model") or "optimart-dry-run",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": summary + "\n\n" + preview},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.tokens_after,
            "completion_tokens": 0,
            "total_tokens": result.tokens_after,
        },
        "optimart": result.stats(),
    }


def attach_stats(payload: dict[str, Any], result: PruneResult) -> dict[str, Any]:
    payload = dict(payload)
    payload["optimart"] = result.stats()
    return payload
