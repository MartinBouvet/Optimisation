#!/usr/bin/env python3
"""Open the Optimart live demo at http://127.0.0.1:8000."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimart.proxy.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "proxy.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lexical", action="store_true", help="Skip MiniLM, Jaccard only.")
    parser.add_argument("--checkpoint", action="store_true", help="Use the local trained weights if present.")
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.lexical:
        cfg["scorer"] = "lexical"
        cfg.pop("checkpoint", None)
    elif not args.checkpoint:
        # The Hotpot checkpoint is English-QA; cosine MiniLM is the wow demo.
        cfg["scorer"] = "minilm"
        cfg.pop("checkpoint", None)
    print(f"Optimart demo → http://{args.host}:{args.port}")
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
