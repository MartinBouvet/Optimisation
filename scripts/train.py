#!/usr/bin/env python3
"""Train the Optimart relevance scorer.

Usage:
  python scripts/train.py --synthetic
  python scripts/train.py
  python scripts/train.py --synthetic --max-steps 20 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimart.data.dataset import JsonlPruneDataset, collate_scorer_batch, load_jsonl  # noqa: E402
from optimart.data.synthetic import build_synthetic_records  # noqa: E402
from optimart.models.loss import CompositePruneLoss  # noqa: E402
from optimart.models.scoring import build_scorer  # noqa: E402
from optimart.train.engine import (  # noqa: E402
    evaluate,
    format_metrics,
    pick_device,
    seed_all,
    train_one_epoch,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _linear_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup: int, total: int):
    warmup = max(warmup, 0)
    total = max(total, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(warmup, 1))
        return max(0.0, float(total - step) / float(max(total - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _existing_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    missing: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.exists():
            found.append(path)
        else:
            missing.append(raw)
    if missing:
        print("missing JSONL (run scripts/prepare_dataset.py):")
        for item in missing:
            print(f"  - {item}")
    return found


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-model", type=Path, default=ROOT / "configs" / "model.yaml")
    parser.add_argument("--config-train", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Train on toy data, no downloads.")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (tiny or HF id).")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    model_cfg = _load_yaml(args.config_model)
    train_cfg = _load_yaml(args.config_train)
    if args.backbone:
        model_cfg["backbone"] = args.backbone
    elif args.synthetic and model_cfg.get("backbone") != "tiny":
        # Keep the first smoke run offline unless the user asks otherwise.
        model_cfg["backbone"] = "tiny"

    seed_all(int(train_cfg.get("seed", 42)))
    if args.synthetic:
        train_cfg["optim"]["grad_accum"] = 1
    if args.device == "auto" and model_cfg.get("backbone") == "tiny":
        device = torch.device("cpu")
    else:
        device = pick_device(args.device)
    output_dir = args.output_dir or Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = train_cfg["data"]
    max_seq_length = int(model_cfg.get("max_seq_length", 160))
    max_segments = int(data_cfg.get("max_segments", 24))

    if args.synthetic:
        synth = train_cfg.get("synthetic", {})
        train_records, val_records = build_synthetic_records(
            int(synth.get("n_train", 48)),
            int(synth.get("n_val", 16)),
        )
        epochs = int(synth.get("epochs", 2))
        batch_size = int(synth.get("batch_size", 8))
        lr = float(synth.get("lr", 1e-3))
    else:
        train_files = _existing_files(list(data_cfg["train_files"]))
        val_files = _existing_files(list(data_cfg["val_files"]))
        if not train_files:
            raise SystemExit(
                "No training JSONL found. Run scripts/prepare_dataset.py or pass --synthetic."
            )
        train_records = load_jsonl(train_files)
        val_records = load_jsonl(val_files) if val_files else train_records[: max(1, len(train_records) // 10)]
        epochs = int(train_cfg["optim"]["epochs"])
        batch_size = int(train_cfg["optim"]["batch_size"])
        lr = float(train_cfg["optim"]["lr"])

    model, tokenizer = build_scorer(model_cfg)
    model.to(device)
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    train_ds = JsonlPruneDataset(
        train_records, tokenizer, max_seq_length, max_segments, seed=int(train_cfg.get("seed", 42))
    )
    val_ds = JsonlPruneDataset(
        val_records, tokenizer, max_seq_length, max_segments, seed=int(train_cfg.get("seed", 42)) + 1
    )
    if len(train_ds) == 0:
        raise SystemExit("Training set is empty after filtering.")

    collate = partial(collate_scorer_batch, pad_id=int(pad_id))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    loss_fn = CompositePruneLoss(**train_cfg["loss"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
    )
    steps_per_epoch = max(len(train_loader), 1)
    grad_accum = int(train_cfg["optim"]["grad_accum"])
    total_steps = max((steps_per_epoch // max(grad_accum, 1)) * epochs, 1)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
        epochs = max(1, min(epochs, args.max_steps))
    warmup = int(total_steps * float(train_cfg["optim"]["warmup_ratio"]))
    scheduler = _linear_warmup_scheduler(optimizer, warmup, total_steps)

    snapshot = {"model": model_cfg, "train": train_cfg, "synthetic": args.synthetic, "device": str(device)}
    (output_dir / "run_config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"

    print(f"device={device} backbone={model_cfg['backbone']} train={len(train_ds)} val={len(val_ds)}")
    best_val = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 3))
    stale = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_accum=grad_accum,
            max_grad_norm=float(train_cfg["optim"]["max_grad_norm"]),
            log_every=int(train_cfg.get("log_every_steps", 10)),
            max_steps=args.max_steps,
            epoch=epoch,
            global_step=global_step,
        )
        val_metrics = evaluate(model, loss_fn, val_loader, device)
        row = {"epoch": epoch, "step": global_step, "split": "train", **train_metrics}
        val_row = {"epoch": epoch, "step": global_step, "split": "val", **val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.write(json.dumps(val_row) + "\n")
        print(f"epoch {epoch} train {format_metrics(train_metrics)}")
        print(f"epoch {epoch} val   {format_metrics(val_metrics)}")

        payload = {
            "model_state": model.state_dict(),
            "model_cfg": model_cfg,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        _save_checkpoint(output_dir / "last.pt", payload)
        val_loss = float(val_metrics.get("loss", best_val))
        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            _save_checkpoint(output_dir / "best.pt", payload)
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(f"done. checkpoints in {output_dir}")


if __name__ == "__main__":
    main()
