from __future__ import annotations

import pytest
import requests
from pydantic import ValidationError

from screener.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryConfig,
    call_with_resilience,
    redact_secrets,
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


def test_retry_config_rejects_invalid_attempts() -> None:
    with pytest.raises(ValidationError):
        RetryConfig(attempts=0)


def test_redact_secrets_masks_apikey() -> None:
    url = "https://x/api?from=a&apikey=SECRET123&to=b"
    result = redact_secrets(url)
    assert "apikey=***" in result
    assert "SECRET123" not in result


def test_redact_secrets_case_insensitive_token() -> None:
    text = "request failed: https://x/api?TOKEN=abc"
    result = redact_secrets(text)
    assert "TOKEN=***" in result
    assert "abc" not in result


def test_redact_secrets_no_secrets_unchanged() -> None:
    text = "plain error message with no credentials"
    assert redact_secrets(text) == text


def test_call_with_resilience_redacts_secret_from_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    def always_fail() -> str:
        raise requests.HTTPError("401 Client Error: ... ?apikey=SECRET123")

    with caplog.at_level(logging.WARNING, logger="screener.resilience"):
        result = call_with_resilience(
            "test-redact",
            "secret-op",
            always_fail,
            fallback="fallback",
            retry=RetryConfig(attempts=1, base_delay=0.0, jitter=0.0),
            sleep=lambda _seconds: None,
        )

    assert result == "fallback"
    assert "SECRET123" not in caplog.text
    assert "apikey=***" in caplog.text
