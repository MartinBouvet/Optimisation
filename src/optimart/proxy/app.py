"""OpenAI-compatible FastAPI reverse proxy that prunes context before upstream."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from optimart.eval.baselines import run_arm
from optimart.eval.cost import INPUT_USD_PER_MILLION, usd_for_requests
from optimart.prune.pipeline import PruneEngine, PruneResult
from optimart.proxy.openai_compat import attach_stats, dry_run_completion

ROOT = Path(__file__).resolve().parents[3]
DEMO_HTML = Path(__file__).resolve().parents[1] / "demo" / "index.html"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _checkpoint_path(cfg: dict[str, Any]) -> Path | None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if path.exists():
        return path
    candidate = ROOT / checkpoint
    return candidate if candidate.exists() else None


def resolve_score_fn(cfg: dict[str, Any]) -> tuple[Any, str]:
    """Prefer a trained checkpoint, else MiniLM embeddings, else lexical overlap."""
    path = _checkpoint_path(cfg)
    if path is not None:
        from optimart.runtime.infer import load_checkpoint

        return load_checkpoint(path, device=str(cfg.get("device", "auto"))), "checkpoint"
    scorer = str(cfg.get("scorer") or "lexical")
    if scorer == "minilm":
        from optimart.runtime.embed import MiniLMEmbedScorer

        print("loading MiniLM embed scorer (no checkpoint)")
        return MiniLMEmbedScorer(device=str(cfg.get("device", "auto"))), "minilm"
    return None, "lexical"


def build_engine(cfg: dict[str, Any]) -> tuple[PruneEngine, str]:
    prune_cfg = cfg.get("prune") or {}
    score_fn, name = resolve_score_fn(cfg)
    engine = PruneEngine(
        score_fn,
        budget_ratio=float(prune_cfg.get("budget_ratio", 0.45)),
        max_query_chars=int(prune_cfg.get("max_query_chars", 400)),
        min_segment_chars=int(prune_cfg.get("min_segment_chars", 12)),
        encoding=str(prune_cfg.get("encoding", "cl100k_base")),
    )
    return engine, name


def _arm_payload(result: PruneResult) -> dict[str, Any]:
    saved = usd_for_requests(
        max(result.tokens_before - result.tokens_after, 0),
        100_000,
        INPUT_USD_PER_MILLION["gpt-4o"],
    )
    return {
        "stats": result.stats(),
        "segments": [trace.as_dict() for trace in result.traces],
        "pruned_text": result.pruned_text,
        "saved_per_100k_gpt4o": round(saved, 2),
    }


def create_app(cfg: dict[str, Any] | None = None) -> FastAPI:
    cfg = cfg or {}
    engine, scorer_name = build_engine(cfg)
    dry_run = bool(cfg.get("dry_run", True))
    upstream = str(cfg.get("upstream_base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    api_key = str(cfg.get("api_key") or os.getenv("OPENAI_API_KEY") or "")
    timeout = float(cfg.get("timeout_s", 120))

    app = FastAPI(title="Optimart", version="0.2.0")
    app.state.engine = engine
    app.state.cfg = cfg
    app.state.scorer_name = scorer_name

    @app.get("/", response_class=HTMLResponse)
    def demo() -> HTMLResponse:
        return HTMLResponse(DEMO_HTML.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "dry_run": dry_run, "scorer": scorer_name}

    @app.get("/v1/examples")
    def list_examples() -> dict[str, Any]:
        from optimart.demo.examples import examples

        return {"examples": examples()}

    @app.post("/v1/prune")
    async def prune_document(request: Request) -> JSONResponse:
        body = await request.json()
        question = str(body.get("question") or "").strip()
        context = str(body.get("context") or "").strip()
        if not question or not context:
            return JSONResponse({"error": "question and context are required"}, status_code=400)
        ratio = body.get("budget_ratio")
        previous = engine.budget_ratio
        messages = [
            {"role": "system", "content": "Tu réponds uniquement à partir des documents fournis."},
            {"role": "user", "content": f"{context}\nQuestion: {question}"},
        ]
        if ratio is not None:
            engine.budget_ratio = float(ratio)
        try:
            result = engine.prune_messages(messages)
            trunc = run_arm(engine, messages, "trunc_head", budget_tokens=result.budget)
        finally:
            engine.budget_ratio = previous
        optimart = _arm_payload(result)
        return JSONResponse(
            {
                "question": question,
                "scorer": scorer_name,
                "optimart": optimart,
                "trunc": _arm_payload(trunc),
                "cost": {"model": "gpt-4o", "saved_per_100k": optimart["saved_per_100k_gpt4o"]},
            }
        )

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
    config_path = Path(os.getenv("OPTIMART_PROXY_CONFIG") or ROOT / "configs" / "proxy.yaml")
    cfg = _load_yaml(config_path) if config_path.exists() else {}
    if os.getenv("OPTIMART_CHECKPOINT"):
        cfg["checkpoint"] = os.getenv("OPTIMART_CHECKPOINT")
    if os.getenv("OPTIMART_DRY_RUN"):
        cfg["dry_run"] = os.getenv("OPTIMART_DRY_RUN") not in {"0", "false", "False"}
    if os.getenv("OPTIMART_SCORER"):
        cfg["scorer"] = os.getenv("OPTIMART_SCORER")
    return create_app(cfg)
