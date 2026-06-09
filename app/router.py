"""Routing strategies and the orchestrator that uses them.

Strategies are simple objects implementing :meth:`RoutingStrategy.select`.
Each one picks a *first choice* from the list of currently healthy
providers. What happens when that choice fails is the orchestrator's
job, not the strategy's — this separation keeps strategies trivial to
unit test and easy to add.
"""
from __future__ import annotations

import itertools
import logging
import time
from abc import ABC, abstractmethod

from .providers import BaseProvider, CircuitOpenError, ProviderError
from .schemas import Message, ProviderName, Strategy, Trace

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

class RoutingStrategy(ABC):
    @abstractmethod
    def select(self, providers: list[BaseProvider]) -> BaseProvider:
        ...


class LatencyStrategy(RoutingStrategy):
    """Pick the provider with the lowest observed p50 latency.

    Providers with no samples yet are treated as last-resort (inf), so
    they don't get picked over a warm provider with real data.
    """

    def select(self, providers: list[BaseProvider]) -> BaseProvider:
        return min(
            providers,
            key=lambda p: p.stats.p50_latency_ms or float("inf"),
        )


class CostStrategy(RoutingStrategy):
    """Pick the cheapest provider per output token."""

    def select(self, providers: list[BaseProvider]) -> BaseProvider:
        return min(providers, key=lambda p: p.cost_per_1k_tokens)


class QualityStrategy(RoutingStrategy):
    """Pick the highest quality_score."""

    def select(self, providers: list[BaseProvider]) -> BaseProvider:
        return max(providers, key=lambda p: p.quality_score)


class RoundRobinStrategy(RoutingStrategy):
    """Cycle through providers in a stable order.

    The order is fixed at strategy-construction time so behaviour is
    deterministic across calls and easy to reason about in tests. If a
    provider is temporarily unhealthy it's already been filtered out
    before ``select`` runs, so we just take the next index modulo the
    healthy list size.
    """

    def __init__(self) -> None:
        self._counter = itertools.count()

    def select(self, providers: list[BaseProvider]) -> BaseProvider:
        idx = next(self._counter) % len(providers)
        return providers[idx]


# Single registry — strategies are stateless except round-robin, which
# keeps a counter. Sharing instances across requests is safe.
STRATEGIES: dict[Strategy, RoutingStrategy] = {
    Strategy.LATENCY: LatencyStrategy(),
    Strategy.COST: CostStrategy(),
    Strategy.QUALITY: QualityStrategy(),
    Strategy.ROUND_ROBIN: RoundRobinStrategy(),
}


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #

class Orchestrator:
    """Selects a provider per request and handles single-hop failover.

    Failover is intentionally shallow: try the chosen provider, and if
    it fails once try one other healthy provider. Retrying further is
    the queue/worker's responsibility (see InferenceGateway-Scale),
    not the router's — chaining N retries here multiplies tail latency.
    """

    def __init__(self, providers: list[BaseProvider]) -> None:
        if not providers:
            raise ValueError("Orchestrator requires at least one provider")
        self._providers = providers

    def providers(self) -> list[BaseProvider]:
        return list(self._providers)

    async def route(
        self,
        *,
        req_id: str,
        messages: list[Message],
        strategy: Strategy,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> tuple[str, Trace]:
        healthy = self._healthy()
        if not healthy:
            raise ProviderError("no healthy providers available")

        chosen = STRATEGIES[strategy].select(healthy)
        logger.info(
            "route req=%s strategy=%s provider=%s model=%s",
            req_id, strategy.value, chosen.name.value, chosen.model,
        )

        failed_over = False
        fallback_from: ProviderName | None = None

        start = time.perf_counter()
        try:
            content, _ = await chosen.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except (ProviderError, CircuitOpenError) as exc:
            logger.warning(
                "failover req=%s primary=%s reason=%s",
                req_id, chosen.name.value, exc,
            )
            fallback = self._pick_fallback(exclude=chosen)
            if fallback is None:
                raise
            fallback_from = chosen.name
            chosen = fallback
            content, _ = await chosen.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            failed_over = True

        total_ms = int((time.perf_counter() - start) * 1000)
        trace = Trace(
            request_id=req_id,
            strategy=strategy,
            provider=chosen.name,
            model=chosen.model,
            total_ms=total_ms,
            failed_over=failed_over,
            fallback_from=fallback_from,
        )
        return content, trace

    # ---- helpers --------------------------------------------------------- #

    def _healthy(self) -> list[BaseProvider]:
        return [p for p in self._providers if not p.stats.circuit_is_open()]

    def _pick_fallback(self, *, exclude: BaseProvider) -> BaseProvider | None:
        for p in self._healthy():
            if p is not exclude:
                return p
        return None
