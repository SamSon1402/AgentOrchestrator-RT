"""FastAPI application — wires the orchestrator, health monitor, and routes.

Run locally::

    cp .env.example .env  # add at least one provider API key
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import get_settings
from .health import HealthMonitor
from .providers import (
    AnthropicProvider,
    BaseProvider,
    MistralProvider,
    OpenAIProvider,
    ProviderError,
)
from .router import Orchestrator
from .schemas import CompletionRequest, CompletionResponse, HealthReport

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Provider wiring                                                             #
# --------------------------------------------------------------------------- #

def _build_providers(http: httpx.AsyncClient) -> list[BaseProvider]:
    """Construct provider adapters for whichever API keys are configured.

    At least one key must be set, otherwise the gateway is useless and we
    fail fast at startup rather than 503-ing on the first request.
    """
    s = get_settings()
    kwargs = dict(
        breaker_threshold=s.circuit_breaker_threshold,
        breaker_reset_s=s.circuit_breaker_reset_s,
        timeout_s=s.request_timeout_s,
    )

    providers: list[BaseProvider] = []
    if s.openai_api_key:
        providers.append(OpenAIProvider(s.openai_api_key, http, **kwargs))
    if s.anthropic_api_key:
        providers.append(AnthropicProvider(s.anthropic_api_key, http, **kwargs))
    if s.mistral_api_key:
        providers.append(MistralProvider(s.mistral_api_key, http, **kwargs))

    if not providers:
        raise RuntimeError(
            "no providers configured — set at least one of "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY"
        )
    return providers


# --------------------------------------------------------------------------- #
# Lifespan                                                                    #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up shared resources for the app lifetime.

    One ``httpx.AsyncClient`` is shared across providers so HTTP/2 + keep-alive
    connection pooling actually kicks in. The health monitor runs as a
    background task and is cancelled cleanly on shutdown.
    """
    s = get_settings()
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(s.request_timeout_s, connect=s.connect_timeout_s),
        limits=httpx.Limits(max_keepalive_connections=32, max_connections=64),
    )
    providers = _build_providers(http)
    orchestrator = Orchestrator(providers)
    monitor = HealthMonitor(providers, interval_s=s.health_check_interval_s)
    await monitor.start()

    app.state.http = http
    app.state.orchestrator = orchestrator
    app.state.monitor = monitor

    logger.info("startup_complete providers=%s",
                [p.name.value for p in providers])
    try:
        yield
    finally:
        await monitor.stop()
        await http.aclose()
        logger.info("shutdown_complete")


app = FastAPI(
    title="AgentOrchestrator-RT",
    version="0.1.0",
    description=(
        "Multi-LLM real-time routing engine with pluggable strategies, "
        "per-provider circuit breakers, and per-request tracing."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #

@app.get("/healthz", response_model=HealthReport, tags=["ops"])
async def healthz(request: Request) -> HealthReport:
    """Liveness + per-provider status. Safe to hit from a K8s probe."""
    monitor: HealthMonitor = request.app.state.monitor
    return monitor.report()


@app.post("/v1/completions", response_model=CompletionResponse, tags=["completions"])
async def completions(req: CompletionRequest, request: Request) -> CompletionResponse:
    """Synchronous completion.

    The orchestrator picks a provider according to ``req.strategy``, falls
    back once on failure, and returns the response with a :class:`Trace`
    so the caller can attribute latency and detect failovers.
    """
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="use /v1/completions/stream for streaming",
        )

    req_id = str(uuid4())
    orchestrator: Orchestrator = request.app.state.orchestrator

    try:
        content, trace = await orchestrator.route(
            req_id=req_id,
            messages=req.messages,
            strategy=req.strategy,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except ProviderError as exc:
        logger.error("route_failed req=%s reason=%s", req_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return CompletionResponse(content=content, trace=trace)


@app.post("/v1/completions/stream", tags=["completions"])
async def completions_stream(req: CompletionRequest, request: Request):
    """Streaming completion via Server-Sent Events.

    Emits three event types::

        event: trace   data: <Trace JSON>
        event: token   data: {"content": "..."}    (one or more)
        event: done    data: [DONE]

    True token-by-token streaming requires each provider's SSE parser,
    which the orchestrator can be extended with. For now the contract is
    in place and the chunking can be swapped in without changing the
    route signature.
    """
    req_id = str(uuid4())
    orchestrator: Orchestrator = request.app.state.orchestrator

    async def event_stream():
        try:
            content, trace = await orchestrator.route(
                req_id=req_id,
                messages=req.messages,
                strategy=req.strategy,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except ProviderError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        yield f"event: trace\ndata: {trace.model_dump_json()}\n\n"
        yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
