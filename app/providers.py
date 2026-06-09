"""LLM provider adapters.

All three adapters present the same async ``complete()`` interface so the
router can use them polymorphically. Each adapter:

* records rolling latency (used by the latency-aware strategy),
* tracks consecutive failures and opens a circuit breaker when a
  threshold is crossed (used to take a provider out of rotation),
* shares a single ``httpx.AsyncClient`` for connection pooling.

The HTTP request body shape is the only thing that differs between
providers — everything else lives in :class:`BaseProvider`.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import httpx

from .schemas import Message, ProviderName

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised when a provider call fails (network, timeout, 4xx/5xx)."""


class CircuitOpenError(ProviderError):
    """Raised when the breaker is open and we refuse the call fast."""


# --------------------------------------------------------------------------- #
# Stats — one instance per provider                                           #
# --------------------------------------------------------------------------- #

@dataclass
class ProviderStats:
    """Rolling stats the router reads on every routing decision.

    Latencies use a bounded deque so memory is O(maxlen) regardless of
    traffic. Sorting on every read is fine at maxlen=64; switch to a
    quantile sketch (e.g. tdigest) if maxlen grows.
    """

    recent_latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0  # epoch seconds; 0 = closed
    total_requests: int = 0
    total_failures: int = 0

    @property
    def p50_latency_ms(self) -> float:
        if not self.recent_latencies_ms:
            return 0.0
        ordered = sorted(self.recent_latencies_ms)
        return ordered[len(ordered) // 2]

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests

    def circuit_is_open(self) -> bool:
        return time.time() < self.circuit_open_until


# --------------------------------------------------------------------------- #
# Base + concrete providers                                                   #
# --------------------------------------------------------------------------- #

class BaseProvider(ABC):
    """Subclasses set ``name``, ``model``, ``cost_per_1k_tokens``,
    ``quality_score`` and implement :meth:`_call_api`.
    """

    name: ProviderName
    model: str
    cost_per_1k_tokens: float  # USD; used by cost strategy
    quality_score: float       # 0..1; used by quality strategy

    def __init__(
        self,
        api_key: str,
        http: httpx.AsyncClient,
        *,
        breaker_threshold: int = 5,
        breaker_reset_s: int = 30,
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._http = http
        self._breaker_threshold = breaker_threshold
        self._breaker_reset_s = breaker_reset_s
        self._timeout_s = timeout_s
        self.stats = ProviderStats()

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> tuple[str, int]:
        """Return ``(content, latency_ms)``. Raises :class:`ProviderError`."""
        if self.stats.circuit_is_open():
            raise CircuitOpenError(f"{self.name.value} circuit open")

        self.stats.total_requests += 1
        start = time.perf_counter()

        try:
            content = await self._call_api(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout or self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — we want to record any failure
            self._record_failure(exc)
            raise ProviderError(f"{self.name.value} call failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        self._record_success(latency_ms)
        return content, latency_ms

    # ---- breaker bookkeeping --------------------------------------------- #

    def _record_success(self, latency_ms: int) -> None:
        self.stats.recent_latencies_ms.append(latency_ms)
        self.stats.consecutive_failures = 0

    def _record_failure(self, exc: Exception) -> None:
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        if self.stats.consecutive_failures >= self._breaker_threshold:
            self.stats.circuit_open_until = time.time() + self._breaker_reset_s
            logger.warning(
                "circuit_open provider=%s reset_in=%ds cause=%s",
                self.name.value, self._breaker_reset_s, exc,
            )

    # ---- to implement ---------------------------------------------------- #

    @abstractmethod
    async def _call_api(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> str:
        ...


# --------------------------------------------------------------------------- #

class OpenAIProvider(BaseProvider):
    name = ProviderName.OPENAI
    model = "gpt-4o-2024-08-06"
    cost_per_1k_tokens = 0.005
    quality_score = 0.95

    async def _call_api(self, messages, *, max_tokens, temperature, timeout) -> str:
        resp = await self._http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicProvider(BaseProvider):
    name = ProviderName.ANTHROPIC
    model = "claude-3-5-sonnet-20241022"
    cost_per_1k_tokens = 0.003
    quality_score = 0.97

    async def _call_api(self, messages, *, max_tokens, temperature, timeout) -> str:
        # Anthropic separates the system prompt from the user/assistant turns.
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        turns = [m.model_dump() for m in messages if m.role != "system"]

        body: dict = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        resp = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


class MistralProvider(BaseProvider):
    name = ProviderName.MISTRAL
    model = "mistral-large-latest"
    cost_per_1k_tokens = 0.002
    quality_score = 0.91

    async def _call_api(self, messages, *, max_tokens, temperature, timeout) -> str:
        resp = await self._http.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": [m.model_dump() for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
