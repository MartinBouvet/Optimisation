"""OpenAI-compatible FastAPI reverse proxy that prunes context before upstream."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from optimart.prune.pipeline import PruneEngine
from optimart.proxy.openai_compat import attach_stats, dry_run_completion


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_engine(cfg: dict[str, Any]) -> PruneEngine:
    prune_cfg = cfg.get("prune") or {}
    checkpoint = cfg.get("checkpoint")
    score_fn = None
    if checkpoint:
        from optimart.runtime.infer import load_checkpoint

        path = Path(checkpoint)
        if not path.exists():
            root = Path(__file__).resolve().parents[3]
            path = root / checkpoint
        if path.exists():
            score_fn = load_checkpoint(path, device=str(cfg.get("device", "auto")))
        else:
            print(f"checkpoint missing ({checkpoint}), using lexical scorer")
    return PruneEngine(
        score_fn,
        budget_ratio=float(prune_cfg.get("budget_ratio", 0.45)),
        max_query_chars=int(prune_cfg.get("max_query_chars", 400)),
        min_segment_chars=int(prune_cfg.get("min_segment_chars", 12)),
        encoding=str(prune_cfg.get("encoding", "cl100k_base")),
    )


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    cfg = cfg or {}
    engine = build_engine(cfg)
    dry_run = bool(cfg.get("dry_run", True))
    upstream = str(cfg.get("upstream_base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "")
    timeout = float(cfg.get("timeout_s", 120))

    app = FastAPI(title="Optimart", version="0.1.0")
    app.state.engine = engine
    app.state.cfg = cfg

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "dry_run": dry_run, "has_checkpoint": bool(cfg.get("checkpoint"))}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "optimart-proxy", "object": "model", "owned_by": "optimart"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        messages = list(body.get("messages") or [])
        result = engine.prune_messages(messages)
        body["messages"] = result.messages

        if dry_run:
            return JSONResponse(dry_run_completion(body, result))

        headers = {"Content-Type": "application/json"}
        auth = request.headers.get("authorization") or (f"Bearer {api_key}" if api_key else None)
        if auth:
            headers["Authorization"] = auth

        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_response = await client.post(
                f"{upstream}/v1/chat/completions",
                json=body,
                headers=headers,
            )
        try:
            payload = upstream_response.json()
        except ValueError:
            return JSONResponse(
                {"error": {"message": upstream_response.text, "type": "upstream_error"}},
                status_code=upstream_response.status_code,
            )
        if isinstance(payload, dict):
            payload = attach_stats(payload, result)
        return JSONResponse(payload, status_code=upstream_response.status_code)

    return app


def app_from_env() -> FastAPI:
    root = Path(__file__).resolve().parents[3]
    config_path = Path(os.getenv("OPTIMART_PROXY_CONFIG") or root / "configs" / "proxy.yaml")
    cfg = _load_yaml(config_path) if config_path.exists() else {}
    if os.getenv("OPTIMART_CHECKPOINT"):
        cfg["checkpoint"] = os.getenv("OPTIMART_CHECKPOINT")
    if os.getenv("OPTIMART_DRY_RUN"):
        cfg["dry_run"] = os.getenv("OPTIMART_DRY_RUN") not in {"0", "false", "False"}
    return create_app(cfg)
