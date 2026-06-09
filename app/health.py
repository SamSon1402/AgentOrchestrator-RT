"""Background health monitor.

Runs as a long-lived asyncio task started in the FastAPI lifespan
handler. Probes each provider on a fixed interval with a cheap
``ping``-style request so latency stats and circuit state stay fresh
even during low-traffic periods.

The probe is intentionally light: 8-token cap, 5s timeout. It exists
to detect provider drift (e.g. an API key going stale, regional
degradation) before live traffic hits the failure — not to benchmark.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .providers import BaseProvider, ProviderError
from .schemas import HealthReport, Message, ProviderHealth

logger = logging.getLogger(__name__)

_PROBE_MESSAGES = [Message(role="user", content="ping")]
_PROBE_TIMEOUT_S = 5.0
_PROBE_MAX_TOKENS = 8


class HealthMonitor:
    def __init__(self, providers: list[BaseProvider], *, interval_s: int = 10) -> None:
        self._providers = providers
        self._interval_s = interval_s
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="health-monitor")
        logger.info("health_monitor_started interval=%ds providers=%d",
                    self._interval_s, len(self._providers))

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        # Small initial delay so probes don't all fire on the first tick.
        await asyncio.sleep(1)
        while True:
            await asyncio.gather(
                *(self._probe(p) for p in self._providers),
                return_exceptions=True,
            )
            await asyncio.sleep(self._interval_s)

    async def _probe(self, provider: BaseProvider) -> None:
        try:
            await provider.complete(
                _PROBE_MESSAGES,
                max_tokens=_PROBE_MAX_TOKENS,
                timeout=_PROBE_TIMEOUT_S,
            )
        except ProviderError as exc:
            # Failure already recorded inside ``complete``; we just log it.
            logger.debug("probe_failed provider=%s reason=%s",
                         provider.name.value, exc)

    def report(self) -> HealthReport:
        now = datetime.now(timezone.utc)
        items: list[ProviderHealth] = []
        for p in self._providers:
            circuit_open = p.stats.circuit_is_open()
            if circuit_open:
                status: str = "offline"
            elif p.stats.error_rate >= 0.05:
                status = "degraded"
            else:
                status = "online"
            items.append(ProviderHealth(
                name=p.name,
                model=p.model,
                status=status,
                p50_latency_ms=p.stats.p50_latency_ms,
                error_rate=p.stats.error_rate,
                circuit_open=circuit_open,
                last_check=now,
            ))
        return HealthReport(providers=items)
