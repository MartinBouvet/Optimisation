"""Training / evaluation loops for the Optimart scorer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimart.data.dataset import ScorerBatch
from optimart.models.loss import CompositePruneLoss, PruneLossOutput


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _running_mean(store: dict[str, float], values: dict[str, float], count: int) -> None:
    for key, value in values.items():
        store[key] = store.get(key, 0.0) + (value - store.get(key, 0.0)) / count


@torch.no_grad()
def threshold_metrics(logits: torch.Tensor, y_hard: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    valid = mask.bool()
    positives = (y_hard > 0.5) & valid
    predicted = (probs > 0.5) & valid
    n_pos = int(positives.sum().item())
    recall = float((predicted & positives).sum().item() / n_pos) if n_pos else 1.0
    n_pred = int(predicted.sum().item())
    precision = float((predicted & positives).sum().item() / n_pred) if n_pred else 0.0
    pos_scores = probs[positives]
    neg_scores = probs[(y_hard <= 0.5) & valid]
    margin = float(pos_scores.mean() - neg_scores.mean()) if pos_scores.numel() and neg_scores.numel() else 0.0
    return {
        "pos_recall": recall,
        "precision@0.5": precision,
        "score_margin": margin,
    }


def train_one_epoch(
    model: nn.Module,
    loss_fn: CompositePruneLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    *,
    grad_accum: int,
    max_grad_norm: float,
    log_every: int,
    max_steps: int | None,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    averages: dict[str, float] = {}
    seen = 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    step = 0
    for step, batch in enumerate(progress, start=1):
        batch = batch.to(device)
        logits = model(
            batch.input_ids,
            batch.attention_mask,
            batch.token_type_ids,
        )
        out: PruneLossOutput = loss_fn(logits, batch.y_hard, batch.y_target, batch.lengths, batch.mask)
        (out.total / grad_accum).backward()
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        seen += 1
        metrics = out.as_dict()
        metrics.update(threshold_metrics(logits.detach(), batch.y_hard, batch.mask))
        _running_mean(averages, metrics, seen)
        if step % log_every == 0:
            progress.set_postfix(
                loss=f"{averages['loss']:.3f}",
                rec=f"{averages['pos_recall']:.2f}",
                keep=f"{averages['expected_keep_ratio']:.2f}",
            )
        if max_steps is not None and global_step >= max_steps:
            break
    if seen and step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
    return averages, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: CompositePruneLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    averages: dict[str, float] = {}
    seen = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.input_ids, batch.attention_mask, batch.token_type_ids)
        out = loss_fn(logits, batch.y_hard, batch.y_target, batch.lengths, batch.mask)
        metrics = out.as_dict()
        metrics.update(threshold_metrics(logits, batch.y_hard, batch.mask))
        seen += 1
        _running_mean(averages, metrics, seen)
    return averages


def format_metrics(metrics: dict[str, float]) -> str:
    keys = [
        "loss",
        "loss_semantic",
        "loss_ranking",
        "loss_l1",
        "loss_budget",
        "loss_coverage",
        "pos_recall",
        "expected_keep_ratio",
        "score_margin",
    ]
    parts = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    return " ".join(parts)
