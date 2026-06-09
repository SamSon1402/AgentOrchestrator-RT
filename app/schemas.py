"""Request/response and tracing models.

All inputs and outputs to the gateway flow through Pydantic — the model is
the contract. ``Trace`` is returned alongside every completion so callers
can attribute latency and cost to the provider that actually served the
request (not the one originally selected, in case of failover).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Strategy(str, Enum):
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    ROUND_ROBIN = "round_robin"


class ProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    strategy: Strategy = Strategy.LATENCY
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # When True the caller must hit /v1/completions/stream; this flag is
    # rejected on the synchronous endpoint to avoid silent contract drift.
    stream: bool = False


class Trace(BaseModel):
    """Per-request attribution. Returned with every completion."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy: Strategy
    provider: ProviderName
    model: str
    total_ms: int
    failed_over: bool = False
    fallback_from: ProviderName | None = None


class CompletionResponse(BaseModel):
    content: str
    trace: Trace


class ProviderHealth(BaseModel):
    name: ProviderName
    model: str
    status: Literal["online", "degraded", "offline"]
    p50_latency_ms: float
    error_rate: float
    circuit_open: bool
    last_check: datetime


class HealthReport(BaseModel):
    providers: list[ProviderHealth]
