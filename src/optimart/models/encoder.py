"""Compact pair encoder: MiniLM (or a tiny offline stand-in) + regression head."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn


class TinyPairEncoder(nn.Module):
    """Random transformer for tests and smoke runs. Does not download weights."""

    def __init__(
        self,
        vocab_size: int = 1024,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        max_position: int = 256,
    ) -> None:
        super().__init__()
        self.config = type("TinyConfig", (), {"hidden_size": hidden_size})()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos = nn.Embedding(max_position, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Any:
        del token_type_ids
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = self.embed(input_ids) + self.pos(positions)
        padding = attention_mask == 0
        encoded = self.encoder(hidden, src_key_padding_mask=padding)
        return type("EncoderOutput", (), {"last_hidden_state": encoded})()


class HashPairTokenizer:
    """Deterministic character-hash tokenizer. Used only with TinyPairEncoder."""

    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0
    vocab_size = 1024

    def __call__(
        self,
        text_a: str,
        text_b: str | None = None,
        truncation: bool = True,
        max_length: int = 160,
        padding: bool = False,
        return_token_type_ids: bool = True,
        **_: Any,
    ) -> dict[str, list[int]]:
        del padding
        first = self._encode(text_a)
        ids = [self.cls_token_id] + first + [self.sep_token_id]
        types = [0] * len(ids)
        if text_b is not None:
            second = self._encode(text_b)
            ids.extend(second + [self.sep_token_id])
            types.extend([1] * (len(second) + 1))
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
            types = types[:max_length]
            ids[-1] = self.sep_token_id
        attention = [1] * len(ids)
        out = {"input_ids": ids, "attention_mask": attention}
        if return_token_type_ids:
            out["token_type_ids"] = types
        return out

    def _encode(self, text: str) -> list[int]:
        raw = (text or "").encode("utf-8")
        if not raw:
            return [3]
        tokens: list[int] = []
        for index, byte in enumerate(raw[:96]):
            hashed = (byte * 131 + index * 17) % (self.vocab_size - 4) + 3
            tokens.append(hashed)
        return tokens or [3]


class RelevanceEncoder(nn.Module):
    """Backbone + MLP head. Consumes tokenized (query, segment) pairs of shape (N, L)."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        head_hidden: int = 192,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        signature = inspect.signature(backbone.forward)
        self._accepts_token_types = "token_type_ids" in signature.parameters

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self._accepts_token_types and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.backbone(**kwargs)
        cls = output.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def freeze_encoder_layers(backbone: nn.Module, n_layers: int, freeze_embeddings: bool) -> None:
    if freeze_embeddings and hasattr(backbone, "embeddings"):
        for param in backbone.embeddings.parameters():
            param.requires_grad = False
    layers = None
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        layers = encoder.layer
    elif hasattr(backbone, "layers"):
        layers = backbone.layers
    if layers is None or n_layers <= 0:
        return
    for layer in list(layers)[:n_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def build_backbone(name: str) -> tuple[nn.Module, Any, int]:
    """Return (backbone, tokenizer, hidden_size). `tiny` never hits the network."""
    if name == "tiny":
        tokenizer = HashPairTokenizer()
        backbone = TinyPairEncoder(vocab_size=tokenizer.vocab_size)
        return backbone, tokenizer, int(backbone.config.hidden_size)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    backbone = AutoModel.from_pretrained(name)
    hidden = int(backbone.config.hidden_size)
    return backbone, tokenizer, hidden
