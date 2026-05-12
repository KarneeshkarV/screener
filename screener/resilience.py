"""Retry and circuit-breaker helpers for external data providers."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from threading import Lock
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field


LOG = logging.getLogger(__name__)
T = TypeVar("T")


class RetryConfig(BaseModel):
    attempts: int = Field(default=3, ge=1)
    base_delay: float = Field(default=0.5, ge=0.0)
    max_delay: float = Field(default=8.0, ge=0.0)
    jitter: float = Field(default=0.2, ge=0.0)

    model_config = ConfigDict(frozen=True)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    cooldown_seconds: float = Field(default=60.0, ge=0.0)

    model_config = ConfigDict(frozen=True)


class CircuitOpenError(RuntimeError):
    """Raised when a provider's circuit breaker is open."""


class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.config.cooldown_seconds:
                return
            raise CircuitOpenError(f"{self.name} circuit is open")

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.config.failure_threshold:
                self._opened_at = time.monotonic()


_BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = Lock()


def get_breaker(provider: str) -> CircuitBreaker:
    with _BREAKERS_LOCK:
        breaker = _BREAKERS.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(provider)
            _BREAKERS[provider] = breaker
        return breaker


def _sleep_time(config: RetryConfig, attempt_index: int) -> float:
    raw = cast(float, min(config.max_delay, config.base_delay * (2**attempt_index)))
    if config.jitter <= 0:
        return raw
    return raw + random.uniform(0.0, config.jitter)


def call_with_resilience(
    provider: str,
    operation: str,
    func: Callable[[], T],
    *,
    fallback: T,
    retry: RetryConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call an external provider with retries and a provider-level circuit."""
    config = retry or RetryConfig()
    breaker = get_breaker(provider)
    try:
        breaker.before_call()
    except CircuitOpenError as exc:
        LOG.warning("%s unavailable for %s: %s", provider, operation, exc)
        return fallback

    last_exc: Exception | None = None
    for attempt in range(max(1, config.attempts)):
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 — provider-agnostic retry wrapper; specific types live at the call site
            last_exc = exc
            if attempt < config.attempts - 1:
                sleep(_sleep_time(config, attempt))
            continue
        breaker.record_success()
        return result

    breaker.record_failure()
    LOG.warning(
        "%s failed for %s after %d attempt(s): %s",
        provider,
        operation,
        max(1, config.attempts),
        last_exc,
    )
    return fallback
