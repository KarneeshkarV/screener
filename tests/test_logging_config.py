"""Tests for the yfinance log suppression used around price downloads."""

from __future__ import annotations

import contextlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from screener import logging_config
from screener.logging_config import (
    YFINANCE_DEBUG_HINT,
    configure_logging,
    suppressed_yfinance_errors,
)


@pytest.fixture(autouse=True)
def _already_configured() -> None:
    """Settle the logging configuration before a test installs its handlers.

    ``configure_logging`` calls ``basicConfig(force=True)``, which drops every
    root handler. A test that attaches a capture handler and only then triggers
    first-time configuration would have it torn off mid-test.
    """
    logging_config.get_logger("tests.logging_config")


@pytest.fixture
def unconfigured_logging(monkeypatch: pytest.MonkeyPatch):
    """Run a test against a fresh configuration, then put the old one back.

    ``configure_logging`` calls ``basicConfig(force=True)``, which tears the
    root logger's handlers off, so a test that reconfigures would otherwise
    leave the rest of the suite logging into nowhere.
    """
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setattr(logging_config, "_CONFIGURED_EXPLICITLY", False)
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_explicit_config_overrides_the_implicit_one(unconfigured_logging: None) -> None:
    # Modules bind `log = get_logger(__name__)` at import time, which implicitly
    # configures at the default level long before click parses --log-level. If
    # that provisional setup counted as final, the flag would silently do
    # nothing -- which is exactly the bug this guards.
    configure_logging(_implicit=True)
    assert logging.getLogger().level == logging.INFO

    configure_logging(level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_first_explicit_config_wins(unconfigured_logging: None) -> None:
    # Nested subcommands each call configure_logging; the outermost decides.
    configure_logging(level="DEBUG")
    configure_logging(level="ERROR")
    assert logging.getLogger().level == logging.DEBUG


def test_implicit_config_runs_only_once(unconfigured_logging: None) -> None:
    configure_logging(_implicit=True)
    logging.getLogger().setLevel(logging.ERROR)
    configure_logging(level="DEBUG", _implicit=True)
    assert logging.getLogger().level == logging.ERROR


@pytest.fixture
def yfinance_logger() -> logging.Logger:
    logger = logging.getLogger("yfinance")
    previous = logger.level
    yield logger
    logger.setLevel(previous)


@pytest.fixture
def captured_root() -> logging.Handler:
    """Collect everything reaching the root logger, including the summary.

    The summary is emitted by ``screener.logging_config``'s own logger, which
    propagates to root -- the same path the user's terminal sees.
    """
    handler = logging.StreamHandler(io.StringIO())
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)
    root.setLevel(previous)


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
        # The message itself is gone; only the summary of what was hidden
        # remains, so the run stays quiet without going silent.
        assert "possibly delisted" not in handler.stream.getvalue()
        assert "yfinance.messages_suppressed" in handler.stream.getvalue()
    finally:
        root.removeHandler(handler)


def test_removes_its_filter_afterwards(yfinance_logger: logging.Logger) -> None:
    with suppressed_yfinance_errors():
        assert yfinance_logger.filters
    assert yfinance_logger.filters == []


def test_removes_its_filter_on_exception(yfinance_logger: logging.Logger) -> None:
    with pytest.raises(RuntimeError):
        with suppressed_yfinance_errors():
            raise RuntimeError("download blew up")
    assert yfinance_logger.filters == []


def test_debug_logging_opts_out(
    yfinance_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Someone diagnosing a fetch must still see why a symbol came back empty.
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    with suppressed_yfinance_errors():
        assert yfinance_logger.filters == []


def test_does_not_touch_other_loggers(yfinance_logger: logging.Logger) -> None:
    other = logging.getLogger("screener.something")
    with suppressed_yfinance_errors():
        assert other.filters == []


def test_reports_the_count_and_how_to_see_them(
    yfinance_logger: logging.Logger, captured_root: logging.Handler
) -> None:
    # Silence alone would make a total outage look like a clean run, so the
    # block has to admit what it hid and name the flag that reveals it.
    with suppressed_yfinance_errors():
        for _ in range(3):
            yfinance_logger.error("possibly delisted")
        assert captured_root.stream.getvalue() == ""

    reported = captured_root.stream.getvalue()
    assert "yfinance.messages_suppressed" in reported
    assert "3" in reported
    assert YFINANCE_DEBUG_HINT in reported


def test_stays_quiet_when_nothing_was_suppressed(
    yfinance_logger: logging.Logger, captured_root: logging.Handler
) -> None:
    with suppressed_yfinance_errors():
        pass
    assert captured_root.stream.getvalue() == ""


def test_info_is_hidden_but_not_counted_as_a_problem(
    yfinance_logger: logging.Logger, captured_root: logging.Handler
) -> None:
    # Routine chatter should not trigger a warning that implies something broke.
    with suppressed_yfinance_errors():
        yfinance_logger.info("downloading")
    assert captured_root.stream.getvalue() == ""


def test_critical_still_gets_through(
    yfinance_logger: logging.Logger, captured_root: logging.Handler
) -> None:
    # Reserved for the unrecoverable, which is never routine noise.
    with suppressed_yfinance_errors():
        yfinance_logger.critical("everything is on fire")
    assert "everything is on fire" in captured_root.stream.getvalue()


def test_counts_are_not_lost_across_threads(
    yfinance_logger: logging.Logger, captured_root: logging.Handler
) -> None:
    # Downloads run on a pool, so the tally must survive concurrent batches.
    def emit() -> None:
        for _ in range(200):
            yfinance_logger.error("possibly delisted")

    with suppressed_yfinance_errors():
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: emit(), range(8)))

    assert "1600" in captured_root.stream.getvalue()
