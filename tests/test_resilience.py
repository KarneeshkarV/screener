from __future__ import annotations

from screener.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    call_with_resilience,
)


def test_retries_then_returns_success() -> None:
    attempts = 0
    sleeps: list[float] = []

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = call_with_resilience(
        "test-retry-success",
        "flaky",
        flaky,
        fallback="fallback",
        retry=RetryConfig(attempts=3, base_delay=0.01, jitter=0.0),
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [0.01, 0.02]


def test_returns_fallback_after_exhausting_retries() -> None:
    attempts = 0

    def failing() -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("down")

    result = call_with_resilience(
        "test-retry-failure",
        "failing",
        failing,
        fallback="fallback",
        retry=RetryConfig(attempts=2, base_delay=0.01, jitter=0.0),
        sleep=lambda _seconds: None,
    )

    assert result == "fallback"
    assert attempts == 2


def test_circuit_opens_and_closes_after_success() -> None:
    breaker = CircuitBreaker(
        "unit",
        CircuitBreakerConfig(failure_threshold=2, cooldown_seconds=0.0),
    )

    breaker.record_failure()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_success()
    breaker.before_call()
