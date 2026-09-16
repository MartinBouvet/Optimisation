"""Offline pointwise mutual information labels from a tiny causal LM.

I(c_i ; q) ≈ log p_teacher(q | c_i) − log p_teacher(q)

This is computed once during dataset preparation. The production scorer never
runs GPT-2: it distills these values (mixed with extractive gold labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True, slots=True)
class PMITeacherConfig:
    model_name: str = "gpt2"
    device: str = "cpu"
    max_length: int = 1024


class GPT2PMITeacher:
    """Prefix-LM PMI estimator. Intentionally small and offline-only."""

    def __init__(self, config: PMITeacherConfig | None = None) -> None:
        self.config = config or PMITeacherConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.to(self.config.device)
        self.model.eval()

    @torch.inference_mode()
    def pmi(self, query: str, context: str) -> float:
        log_p_q = -self._nll(query)
        log_p_q_given_c = -self._conditional_query_nll(context, query)
        return float(log_p_q_given_c - log_p_q)

    def _nll(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
        out = self.model(**encoded, labels=encoded["input_ids"])
        n_pred = max(int(encoded["input_ids"].size(1)) - 1, 1)
        return float(out.loss.item()) * n_pred

    def _conditional_query_nll(self, context: str, query: str) -> float:
        ctx_ids = self.tokenizer(
            context.strip(),
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_length - 32,
        )["input_ids"]
        q_ids = self.tokenizer(
            "\nQuestion: " + query.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )["input_ids"]
        input_ids = (ctx_ids + q_ids)[: self.config.max_length]
        q_len = min(len(q_ids), max(len(input_ids) - len(ctx_ids), 0))
        if q_len == 0:
            return self._nll(query)
        labels = [-100] * (len(input_ids) - q_len) + input_ids[-q_len:]
        tensor = torch.tensor([input_ids], device=self.config.device)
        label_tensor = torch.tensor([labels], device=self.config.device)
        out = self.model(tensor, labels=label_tensor)
        return float(out.loss.item()) * q_len
