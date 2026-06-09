"""Application settings, loaded from env vars or a .env file.

Settings is cached so callers can ``from .config import get_settings`` cheaply
inside request handlers without hitting disk on every call.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "agent-orchestrator-rt"
    log_level: str = "INFO"

    # Provider credentials. A None / empty value disables that provider at startup.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    mistral_api_key: str | None = None

    # Routing defaults
    default_strategy: str = "latency"

    # Health monitor
    health_check_interval_s: int = 10

    # Circuit breaker — opens after N consecutive failures, stays open for T seconds.
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: int = 30

    # HTTP client
    request_timeout_s: float = 30.0
    connect_timeout_s: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
