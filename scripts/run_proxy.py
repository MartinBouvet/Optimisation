#!/usr/bin/env python3
"""Run the Optimart OpenAI-compatible proxy.

Usage:
  python scripts/run_proxy.py
  python scripts/run_proxy.py --no-checkpoint --port 8080
"""

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
    parser.add_argument("--no-checkpoint", action="store_true", help="Lexical scorer only.")
    parser.add_argument("--live", action="store_true", help="Forward to the real OpenAI-compatible API.")
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if args.no_checkpoint:
        cfg.pop("checkpoint", None)
    if args.live:
        cfg["dry_run"] = False
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
