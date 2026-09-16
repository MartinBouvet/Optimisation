"""Load a trained scorer and expose a (query, segments) → scores callable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from optimart.models.scoring import SegmentScorer, build_scorer
from optimart.train.engine import pick_device


class TorchScorer:
    def __init__(
        self,
        model: SegmentScorer,
        tokenizer: Any,
        *,
        max_seq_length: int = 160,
        device: torch.device | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device or next(model.parameters()).device

    def __call__(self, query: str, segments: list[str]) -> list[float]:
        if not segments:
            return []
        logits = self.model.score_texts(
            query,
            segments,
            self.tokenizer,
            self.max_seq_length,
            self.device,
        )
        return torch.sigmoid(logits).reshape(-1).tolist()


def load_checkpoint(path: str | Path, device: str = "auto") -> TorchScorer:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = payload["model_cfg"]
    if str(model_cfg.get("backbone", "")) == "tiny":
        resolved = torch.device("cpu")
    else:
        resolved = pick_device(device)
    model, tokenizer = build_scorer(model_cfg)
    model.load_state_dict(payload["model_state"])
    model.to(resolved)
    max_len = int(model_cfg.get("max_seq_length", 160))
    return TorchScorer(model, tokenizer, max_seq_length=max_len, device=resolved)
