#!/usr/bin/env python3
"""Build labeled JSONL corpora for the Optimart relevance scorer.

Usage:
  python scripts/prepare_dataset.py --config configs/data.yaml
  python scripts/prepare_dataset.py --config configs/data.yaml --max-examples 500
  python scripts/prepare_dataset.py --config configs/data.yaml --with-pmi --max-examples 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimart.data.builders import iter_hotpotqa, iter_squad_packed  # noqa: E402
from optimart.data.labeling import mix_targets  # noqa: E402
from optimart.data.pmi import GPT2PMITeacher, PMITeacherConfig  # noqa: E402
from optimart.data.schema import ExampleRecord  # noqa: E402
from optimart.data.segmentation import TokenCounter  # noqa: E402


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _iter_split(name: str, split: str, cfg: dict[str, Any], counter: TokenCounter, max_examples: int | None) -> Iterator[ExampleRecord]:
    data_cfg = cfg["datasets"][name]
    common = dict(
        split=split,
        counter=counter,
        max_segment_tokens=int(cfg["max_segment_tokens"]),
        max_query_tokens=int(cfg["max_query_tokens"]),
        max_examples=max_examples if max_examples is not None else data_cfg.get("max_examples"),
        min_segment_chars=int(cfg["min_segment_chars"]),
        seed=int(cfg["seed"]),
        hf_id=data_cfg["hf_id"],
    )
    if name == "hotpotqa":
        yield from iter_hotpotqa(config=data_cfg.get("config", "distractor"), **common)
        return
    if name == "squad":
        yield from iter_squad_packed(n_distractors=int(data_cfg.get("n_distractors", 3)), **common)
        return
    raise ValueError(f"Unknown dataset '{name}'")


def _write_jsonl(path: Path, records: list[ExampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _summarize(records: list[ExampleRecord]) -> dict[str, Any]:
    if not records:
        return {"n_examples": 0}
    totals = [r.n_tokens_total for r in records]
    oracles = [r.n_tokens_oracle for r in records]
    keep = [r.n_tokens_oracle / r.n_tokens_total for r in records if r.n_tokens_total]
    pos_rate = [r.n_positive / max(len(r.segments), 1) for r in records]
    return {
        "n_examples": len(records),
        "n_segments_mean": round(statistics.mean(len(r.segments) for r in records), 2),
        "tokens_mean": round(statistics.mean(totals), 1),
        "tokens_p50": int(statistics.median(totals)),
        "oracle_tokens_mean": round(statistics.mean(oracles), 1),
        "oracle_keep_ratio_mean": round(statistics.mean(keep), 4),
        "oracle_reduction_pct": round((1.0 - statistics.mean(keep)) * 100.0, 2),
        "positive_segment_rate": round(statistics.mean(pos_rate), 4),
        "datasets": sorted({r.dataset for r in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="Cap per dataset split (smoke tests).")
    parser.add_argument("--with-pmi", action="store_true", help="Override config and enable GPT-2 PMI labels.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Subset: hotpotqa squad")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = args.output_dir or Path(cfg["output_dir"])
    counter = TokenCounter(cfg.get("encoding", "cl100k_base"))

    pmi_cfg = cfg["labeling"]["pmi"]
    teacher: GPT2PMITeacher | None = None
    use_pmi = bool(args.with_pmi or pmi_cfg.get("enabled"))
    if use_pmi:
        teacher = GPT2PMITeacher(
            PMITeacherConfig(
                model_name=str(pmi_cfg.get("teacher", "gpt2")),
                device=str(pmi_cfg.get("device", "cpu")),
                max_length=int(pmi_cfg.get("max_length", 1024)),
            )
        )

    clip = tuple(cfg["labeling"].get("pmi_clip", [-8.0, 8.0]))
    alpha = float(cfg["labeling"]["alpha"])
    enabled = args.datasets or [
        name for name, spec in cfg["datasets"].items() if spec.get("enabled", True)
    ]

    grouped: dict[str, list[ExampleRecord]] = {}
    for name in enabled:
        splits = cfg["datasets"][name].get("splits", ["train", "validation"])
        for split in splits:
            key = f"{name}_{split}"
            bucket: list[ExampleRecord] = []
            cap = args.max_examples
            if cap is not None and split != "train":
                cap = max(80, cap // 5)
            iterator = _iter_split(name, split, cfg, counter, cap)
            for record in tqdm(iterator, desc=key, unit="ex"):
                mix_targets(record, alpha=alpha, pmi_clip=clip, teacher=teacher)
                bucket.append(record)
            grouped[key] = bucket
            out_path = output_dir / f"{key}.jsonl"
            _write_jsonl(out_path, bucket)
            print(f"wrote {out_path}  ({len(bucket)} examples)")

    report = {key: _summarize(recs) for key, recs in grouped.items()}
    print(json.dumps(report, indent=2))
    if cfg.get("report", {}).get("write_json", True):
        report_path = output_dir / "dataset_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
