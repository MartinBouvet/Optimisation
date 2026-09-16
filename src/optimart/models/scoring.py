"""Score every context segment against a query: (B, S, L) → (B, S) logits."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from optimart.models.encoder import (
    RelevanceEncoder,
    build_backbone,
    freeze_encoder_layers,
)


class SegmentScorer(nn.Module):
    """Cross-encoder: each (query, segment) pair gets one relevance logit."""

    def __init__(self, encoder: RelevanceEncoder, cls_token_id: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_token_id = cls_token_id

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        batch, n_segments, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * n_segments, length)
        flat_mask = attention_mask.reshape(batch * n_segments, length)
        # Fully padded slots make TransformerEncoder NaN; give them a dummy CLS.
        empty = flat_mask.sum(dim=-1) == 0
        if bool(empty.any()):
            flat_ids = flat_ids.clone()
            flat_mask = flat_mask.clone()
            flat_ids[empty, 0] = self.cls_token_id
            flat_mask[empty, 0] = 1
        flat_types = None
        if token_type_ids is not None:
            flat_types = token_type_ids.reshape(batch * n_segments, length)
        logits = self.encoder(flat_ids, flat_mask, flat_types).view(batch, n_segments)
        if segment_mask is not None:
            logits = logits.masked_fill(segment_mask == 0, 0.0)
        return logits

    @torch.inference_mode()
    def score_texts(
        self,
        query: str,
        segments: list[str],
        tokenizer: Any,
        max_length: int,
        device: torch.device,
    ) -> Tensor:
        """Convenience path for later proxy / knapsack (one example)."""
        if not segments:
            return torch.empty(0, device=device)
        encoded = [
            tokenizer(query, text, truncation=True, max_length=max_length, padding=False)
            for text in segments
        ]
        length = max(len(item["input_ids"]) for item in encoded)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        ids = torch.full((1, len(encoded), length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros(1, len(encoded), length, dtype=torch.long, device=device)
        types = torch.zeros(1, len(encoded), length, dtype=torch.long, device=device)
        for index, item in enumerate(encoded):
            tokens = item["input_ids"]
            ids[0, index, : len(tokens)] = torch.tensor(tokens, device=device)
            mask[0, index, : len(tokens)] = torch.tensor(item["attention_mask"], device=device)
            if "token_type_ids" in item:
                types[0, index, : len(tokens)] = torch.tensor(item["token_type_ids"], device=device)
        return self.forward(ids, mask, types)


def build_scorer(model_cfg: dict[str, Any]) -> tuple[SegmentScorer, Any]:
    backbone_name = str(model_cfg.get("backbone", "tiny"))
    backbone, tokenizer, hidden = build_backbone(backbone_name)
    freeze_encoder_layers(
        backbone,
        n_layers=int(model_cfg.get("freeze_layers", 0)),
        freeze_embeddings=bool(model_cfg.get("freeze_embeddings", False)),
    )
    encoder = RelevanceEncoder(
        backbone=backbone,
        hidden_size=hidden,
        dropout=float(model_cfg.get("dropout", 0.1)),
        head_hidden=int(model_cfg.get("head_hidden", 192)),
    )
    cls_id = getattr(tokenizer, "cls_token_id", None) or getattr(tokenizer, "bos_token_id", None) or 1
    return SegmentScorer(encoder, cls_token_id=int(cls_id)), tokenizer
