"""Zero-shot MiniLM scorer: cosine(query, sentence). No fine-tuning required."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from optimart.train.engine import pick_device

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbedScorer:
    """Bi-encoder relevance: higher cosine = more useful for the question."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def _embed(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            chunks.append(F.normalize(pooled, p=2, dim=1))
        return torch.cat(chunks, dim=0)

    def __call__(self, query: str, segments: list[str]) -> list[float]:
        if not segments:
            return []
        query_vec = self._embed([query or ""])
        segment_vec = self._embed(segments)
        cosine = (segment_vec @ query_vec.T).squeeze(-1)
        return ((cosine + 1.0) / 2.0).clamp(0.0, 1.0).tolist()
