#!/usr/bin/env python3
"""Compare full context vs naive truncation vs Optimart.

Usage:
  python scripts/run_benchmarks.py
  python scripts/run_benchmarks.py --hotpot 24
  python scripts/run_benchmarks.py --no-checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimart.eval.baselines import Policy, run_arm  # noqa: E402
from optimart.eval.cases import EvalCase, crafted_cases  # noqa: E402
from optimart.eval.cost import cost_table  # noqa: E402
from optimart.eval.metrics import answer_covered, summarize, support_recall  # noqa: E402
from optimart.prune.pipeline import PruneEngine, message_text  # noqa: E402

ARMS: list[Policy] = ["full", "trunc_head", "trunc_tail", "optimart"]


def _joined(messages: list[dict]) -> str:
    return " ".join(message_text(msg) for msg in messages)


def _evaluate_case(engine: PruneEngine, case: EvalCase) -> dict[str, dict[str, Any]]:
    messages = case.messages()
    query = engine._query_from_messages(messages)
    segments = engine._segment_messages(messages, query)
    budget = engine.budget_for(segments)
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        result = run_arm(engine, messages, arm, budget_tokens=None if arm == "full" else budget)
        text = _joined(result.messages)
        per_arm[arm] = {
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "reduction": result.reduction,
            "latency_ms": result.latency_ms,
            "budget": result.budget,
            "answer_coverage": 1.0 if answer_covered(case.answers, text) else 0.0,
            "support_recall": support_recall(case.supports, text),
        }
    return per_arm


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Optimart benchmark note",
        "",
        "Extractive protocol: a gold answer counts as preserved if it still appears in the pruned prompt.",
        "This is **not** LLM Exact Match. Same token budget B for truncation and Optimart.",
        "",
        f"- Suite: `{report['suite']}`",
        f"- Cases: {report['n_cases']}",
        f"- Scorer: `{report['scorer']}`",
        f"- Budget ratio: {report['budget_ratio']}",
        "",
        "| Arm | Tokens in (mean) | Tokens out (mean) | Reduction | Answer coverage | Support recall | Latency p50 (ms) | Latency p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, stats in report["arms"].items():
        lines.append(
            "| {arm} | {tin:.1f} | {tout:.1f} | {red:.1%} | {cov:.1%} | {sup:.1%} | {p50:.2f} | {p95:.2f} |".format(
                arm=arm,
                tin=stats["tokens_before_mean"],
                tout=stats["tokens_after_mean"],
                red=stats["reduction_mean"],
                cov=stats["answer_coverage"],
                sup=stats["support_recall"],
                p50=stats["latency_ms_p50"],
                p95=stats["latency_ms_p95"],
            )
        )
    lines.extend(["", "## Cost model · 100 000 requests · input tokens only", ""])
    lines.append("| Model | Full | Optimart | Saved | Trunc head saved |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, row in report["cost_100k"].items():
        lines.append(
            f"| {model} | ${row['full']:.2f} | ${row['optimart']:.2f} | ${row['optimart_saved_vs_full']:.2f} | ${row['trunc_head_saved_vs_full']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Input prices used: gpt-4o-mini $0.15 / 1M, gpt-4o $2.50 / 1M (OpenAI public list, Sept 2026).",
            "Output tokens are ignored: pruning changes the prompt, not the completion length in this model.",
            "",
            "## Limits",
            "",
            "- Extractive coverage is not LLM Exact Match.",
            "- MiniLM was fine-tuned on 800 HotpotQA train examples (2 epochs).",
            "- Hotpot cases in this suite skip the first 200 validation items so they are not the training val slice.",
            "- Latency is PyTorch MiniLM, not ONNX; long Hotpot contexts are well above the 25 ms design target.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "train_smoke" / "best.pt")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--hotpot", type=int, default=0, help="Extra HotpotQA validation examples.")
    parser.add_argument("--hotpot-skip", type=int, default=200, help="Skip first N val examples (avoid train overlap).")
    parser.add_argument("--budget-ratio", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()

    score_fn = None
    scorer_name = "lexical"
    if not args.no_checkpoint and args.checkpoint.exists():
        from optimart.runtime.infer import load_checkpoint

        score_fn = load_checkpoint(args.checkpoint)
        scorer_name = f"checkpoint:{args.checkpoint}"

    engine = PruneEngine(score_fn, budget_ratio=args.budget_ratio)
    cases = crafted_cases()
    suite = "crafted"
    if args.hotpot:
        from optimart.eval.hotpot import load_hotpot_cases

        cases = cases + load_hotpot_cases(args.hotpot, skip=args.hotpot_skip)
        suite = f"crafted+hotpot{args.hotpot}"

    # Warmup so p50/p95 are not dominated by the first CUDA/MPS/CPU init.
    run_arm(engine, cases[0].messages(), "optimart")

    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for case in cases:
        result = _evaluate_case(engine, case)
        for arm, row in result.items():
            by_arm[arm].append(row)

    summaries = {arm: summarize(rows) for arm, rows in by_arm.items()}
    avg_tokens = {arm: stats["tokens_after_mean"] for arm, stats in summaries.items()}
    report = {
        "suite": suite,
        "n_cases": len(cases),
        "scorer": scorer_name,
        "budget_ratio": args.budget_ratio,
        "protocol": "extractive_answer_coverage",
        "arms": summaries,
        "cost_100k": cost_table(avg_tokens, n_requests=100_000),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "latest.json"
    md_path = ROOT / "paper" / "BENCHMARK.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
