"""Structured-logging bootstrap for the screener CLI.

structlog is wired *alongside* Rich console output. Rich keeps the user-
facing tables on stdout; structlog writes diagnostic events to stderr so
the two streams can be redirected independently. JSON output is selected
by ``SCREENER_LOG_JSON=1`` (env) or ``configure_logging(json=True)``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from typing import Any, Iterator

import structlog


_CONFIGURED = False
_CONFIGURED_EXPLICITLY = False


def configure_logging(
    level: str = "INFO", *, json: bool | None = None, _implicit: bool = False
) -> None:
    """Configure structlog + stdlib logging for the screener.

    The first *explicit* call wins; later ones are no-ops, so nested CLI
    subcommands don't fight over the configuration.

    An *implicit* call -- the auto-configuration in ``get_logger`` -- only
    installs provisional defaults that a later explicit call may still replace.
    That distinction is load-bearing: modules do ``log = get_logger(__name__)``
    at import time, which happens long before click parses ``--log-level``, so
    treating that provisional setup as final would silently pin every run to
    the default level and make the flag do nothing.
    """
    global _CONFIGURED, _CONFIGURED_EXPLICITLY
    if _CONFIGURED_EXPLICITLY or (_CONFIGURED and _implicit):
        return

    use_json = json if json is not None else os.environ.get("SCREENER_LOG_JSON") == "1"

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Safe alongside reconfiguration: get_logger returns a lazy proxy that
        # binds (and only then caches) on first *use*, which comes after the
        # CLI has had its say.
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True
    _CONFIGURED_EXPLICITLY = not _implicit


#: Shown whenever yfinance output is hidden, so the messages are one flag away.
YFINANCE_DEBUG_HINT = "re-run with `screener --log-level DEBUG <command>` to see them"


class _YfinanceSuppressor(logging.Filter):
    """Drops yfinance records and tallies the ones worth reporting.

    A filter rather than a level bump, because a level bump discards the
    records before anything can count them -- ``Logger.isEnabledFor`` short-
    circuits ahead of ``Logger.filter``, so the caller could never tell a quiet
    run from a run that swallowed hundreds of failures.

    ``CRITICAL`` is deliberately let through: it is the level yfinance would
    use for something unrecoverable, which is never routine noise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        # Downloads run across a thread pool, and ``+=`` on an int is three
        # bytecodes, so concurrent batches would lose increments unguarded.
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.CRITICAL:
            return True
        if record.levelno >= logging.WARNING:
            with self._lock:
                self.count += 1
        return False


@contextlib.contextmanager
def suppressed_yfinance_errors() -> Iterator[None]:
    """Silence yfinance's expected "possibly delisted" chatter, but say so.

    yfinance reports empty downloads through ``logging.getLogger("yfinance")``
    rather than by writing to stderr, so ``contextlib.redirect_stderr`` cannot
    reach it: ``redirect_stderr`` only rebinds ``sys.stderr``, while the root
    ``StreamHandler`` that ``configure_logging`` installs holds a reference to
    the *original* stderr object and keeps writing there regardless.

    Most of that output is expected -- an empty pre-listing range is exactly
    the signal the FMP fallback needs -- but a genuine outage looks identical
    from here, so silence alone would hide it. On exit the block reports how
    many messages it hid and how to see them, which keeps a normal run quiet
    without making a broken one indistinguishable from a working one.

    yfinance logs to that one logger, never to children of it, so a single
    filter covers the package. It is attached process-wide for the duration of
    the block, which is what callers want: downloads run across a thread pool,
    and per-thread scoping would race.

    Debug logging opts out entirely, so the individual messages stay reachable
    when someone is actually diagnosing a fetch.
    """
    logger = logging.getLogger("yfinance")
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        yield
        return
    suppressor = _YfinanceSuppressor()
    logger.addFilter(suppressor)
    try:
        yield
    finally:
        logger.removeFilter(suppressor)
        if suppressor.count:
            get_logger(__name__).warning(
                "yfinance.messages_suppressed",
                count=suppressor.count,
                hint=YFINANCE_DEBUG_HINT,
            )


def get_logger(name: str | None = None) -> Any:
    """Return a configured structlog logger.

    Auto-configures with defaults on first call so library code can simply
    ``log = get_logger(__name__)`` without ordering concerns.
    """
    if not _CONFIGURED:
        configure_logging(_implicit=True)
    return structlog.get_logger(name)
