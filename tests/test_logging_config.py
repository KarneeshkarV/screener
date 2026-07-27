"""Tests for the yfinance log suppression used around price downloads."""

from __future__ import annotations

import contextlib
import io
import logging

import pytest

from screener.logging_config import suppressed_yfinance_errors


@pytest.fixture
def yfinance_logger() -> logging.Logger:
    logger = logging.getLogger("yfinance")
    previous = logger.level
    yield logger
    logger.setLevel(previous)


def _stderr_bound_handler() -> logging.Handler:
    """A handler holding its own stream reference, like ``basicConfig`` makes.

    This is what breaks ``redirect_stderr``: the handler captures the stream
    object at construction, so rebinding ``sys.stderr`` afterwards is invisible
    to it.
    """
    return logging.StreamHandler(io.StringIO())


def test_suppresses_records_that_redirect_stderr_cannot_reach(
    yfinance_logger: logging.Logger,
) -> None:
    handler = _stderr_bound_handler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        # Baseline: the redirect misses it, proving why suppression is needed.
        with contextlib.redirect_stderr(io.StringIO()) as redirected:
            yfinance_logger.error("possibly delisted")
        assert redirected.getvalue() == ""
        assert "possibly delisted" in handler.stream.getvalue()

        handler.stream = io.StringIO()
        with suppressed_yfinance_errors():
            yfinance_logger.error("possibly delisted")
        assert handler.stream.getvalue() == ""
    finally:
        root.removeHandler(handler)


def test_restores_the_previous_level(yfinance_logger: logging.Logger) -> None:
    yfinance_logger.setLevel(logging.WARNING)
    with suppressed_yfinance_errors():
        assert yfinance_logger.level == logging.CRITICAL
    assert yfinance_logger.level == logging.WARNING


def test_restores_the_previous_level_on_exception(
    yfinance_logger: logging.Logger,
) -> None:
    yfinance_logger.setLevel(logging.WARNING)
    with pytest.raises(RuntimeError):
        with suppressed_yfinance_errors():
            raise RuntimeError("download blew up")
    assert yfinance_logger.level == logging.WARNING


def test_debug_logging_opts_out(
    yfinance_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Someone diagnosing a fetch must still see why a symbol came back empty.
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    with suppressed_yfinance_errors():
        assert yfinance_logger.level != logging.CRITICAL


def test_does_not_touch_other_loggers(yfinance_logger: logging.Logger) -> None:
    other = logging.getLogger("screener.something")
    with suppressed_yfinance_errors():
        assert other.getEffectiveLevel() != logging.CRITICAL
