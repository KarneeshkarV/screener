"""Retry and circuit-breaker helpers for external data providers."""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from threading import Lock
from typing import Mapping, TypeVar, cast
from urllib.error import HTTPError

from pydantic import BaseModel, ConfigDict, Field


LOG = logging.getLogger(__name__)
T = TypeVar("T")

_SECRET_PARAM_RE = re.compile(r"(?i)\b(apikey|api_key|token|auth)=([^&\s\"']+)")

DEFAULT_PROVIDER_RATES: dict[str, float] = {
    "cboe": 2.0,
    "fred": 2.0,
    "fmp": 8.0,
    "screener-in": 2.0,
    "yfinance": 4.0,
    "tradingview": 2.0,
}
_PROVIDER_RATES = DEFAULT_PROVIDER_RATES.copy()
_PROVIDER_RATES_LOCK = Lock()


def redact_secrets(text: str) -> str:
    """Mask credential-bearing query parameters in log/error text."""
    return _SECRET_PARAM_RE.sub(r"\1=***", text)


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


class ProviderRateLimiter:
    """Thread-safe per-provider minimum-interval rate limiter."""

    def __init__(self) -> None:
        self._next_allowed: dict[str, float] = {}
        self._lock = Lock()

    def wait(
        self,
        provider: str,
        rate: float,
        *,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        if rate <= 0:
            return
        interval = 1.0 / rate
        with self._lock:
            now = clock()
            next_allowed = self._next_allowed.get(provider, now)
            delay = max(0.0, next_allowed - now)
            self._next_allowed[provider] = max(now, next_allowed) + interval
        if delay > 0:
            sleep(delay)


_RATE_LIMITER = ProviderRateLimiter()


def set_provider_rates(rates: Mapping[str, float] | None = None) -> None:
    """Override provider request rates; ``None`` restores the defaults."""
    configured = DEFAULT_PROVIDER_RATES if rates is None else rates
    if any(rate <= 0 for rate in configured.values()):
        raise ValueError("provider rates must be greater than zero")
    with _PROVIDER_RATES_LOCK:
        _PROVIDER_RATES.clear()
        _PROVIDER_RATES.update(configured)


def _provider_rate(provider: str) -> float | None:
    with _PROVIDER_RATES_LOCK:
        return _PROVIDER_RATES.get(provider)


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


def _is_http_429(exc: Exception) -> bool:
    if isinstance(exc, HTTPError) and exc.code == 429:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None and isinstance(exc, HTTPError):
        headers = exc.headers
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return min(30.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def call_with_resilience(
    provider: str,
    operation: str,
    func: Callable[[], T],
    *,
    fallback: T,
    retry: RetryConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rate_limiter: ProviderRateLimiter | None = _RATE_LIMITER,
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
    saw_breaker_failure = False
    for attempt in range(max(1, config.attempts)):
        rate = _provider_rate(provider)
        if rate_limiter is not None and rate is not None:
            rate_limiter.wait(provider, rate, clock=clock, sleep=sleep)
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 — provider-agnostic retry wrapper; specific types live at the call site
            last_exc = exc
            is_429 = _is_http_429(exc)
            if not is_429:
                saw_breaker_failure = True
            if attempt < config.attempts - 1:
                delay = _retry_after(exc) if is_429 else None
                sleep(delay if delay is not None else _sleep_time(config, attempt))
            continue
        breaker.record_success()
        return result

    if saw_breaker_failure:
        breaker.record_failure()
    LOG.warning(
        "%s failed for %s after %d attempt(s): %s",
        provider,
        operation,
        max(1, config.attempts),
        redact_secrets(str(last_exc)),
    )
    return fallback
