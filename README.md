# AgentOrchestrator-RT

Multi-LLM real-time routing engine. A small FastAPI gateway that picks a provider per request, falls back on failure, and returns a trace so the caller can attribute latency.

Companion to the live demo at `samson1402.github.io/agent-orchestrator-rt`.

## What's in the box

```
app/
├── main.py        FastAPI app, routes, lifespan
├── config.py      pydantic-settings; env-driven config
├── schemas.py     Pydantic request / response / trace models
├── providers.py   BaseProvider + OpenAI / Anthropic / Mistral adapters
├── router.py      RoutingStrategy interface + 4 strategies + Orchestrator
└── health.py      Async background health monitor
```

## Endpoints

| Method | Path                      | Purpose                                    |
|--------|---------------------------|--------------------------------------------|
| POST   | `/v1/completions`         | Synchronous completion, returns trace      |
| POST   | `/v1/completions/stream`  | SSE: emits `trace`, `token`, `done` events |
| GET    | `/healthz`                | Per-provider status for K8s / monitoring   |

## Routing strategies

| Strategy      | Picks                                | Use case                          |
|---------------|--------------------------------------|-----------------------------------|
| `latency`     | Lowest observed p50                  | Real-time voice agents            |
| `cost`        | Cheapest healthy provider            | Background / batch                |
| `quality`     | Highest `quality_score`              | Customer-facing first turn        |
| `round_robin` | Stable rotation                      | Even load distribution, A/B tests |

Strategy selection runs only over *healthy* providers — anything with an open circuit is filtered out before the strategy sees the list.

## Failure handling

- **Per-provider circuit breaker**: N consecutive failures → opens for T seconds. No external deps; lives in `ProviderStats`.
- **Single-hop failover in the orchestrator**: try chosen → try one other healthy → return 503. Retrying further is the queue's job, not the router's (chaining retries multiplies tail latency).
- **Rolling latency window**: bounded `deque(maxlen=64)` per provider; p50 is fast to read and stays memory-bounded.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # set at least one provider API key
uvicorn app.main:app --reload
```

Smoke test:

```bash
curl -sX POST http://localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "say hi in 5 words"}],
    "strategy": "latency"
  }' | jq
```

```bash
curl http://localhost:8000/healthz | jq
```

## Design notes

- **One shared `httpx.AsyncClient`** across providers — connection pooling and HTTP/2 reuse only kick in if the client is shared. Created once in the lifespan handler.
- **Lifespan, not deprecated startup/shutdown events** — also lets the test client manage state cleanly.
- **Strategies are objects, not if/elif** — adding a new one (e.g. `quality_by_task_type`) is a class + a registry entry. The orchestrator doesn't change.
- **The trace is part of the contract** — every response includes which provider actually served the request and whether failover happened. Easier to debug than log diving.
- **Health probes are intentionally cheap** — short prompt, 8 max tokens, 5s timeout. They exist for liveness, not benchmarking. Real performance measurement lives in `ConvoStream-Bench`.

## Deliberately out of scope

- Persistent queueing, K8s HPA, per-provider token-bucket rate limiting → `InferenceGateway-Scale`
- P95/P99 measurement, load testing, spike injection → `ConvoStream-Bench`
- Authentication, multi-tenant key management, observability backends (OTel, Prometheus) — would be next, but not part of the demo surface
