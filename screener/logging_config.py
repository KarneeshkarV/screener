"""Structured-logging bootstrap for the screener CLI.

structlog is wired *alongside* Rich console output. Rich keeps the user-
facing tables on stdout; structlog writes diagnostic events to stderr so
the two streams can be redirected independently. JSON output is selected
by ``SCREENER_LOG_JSON=1`` (env) or ``configure_logging(json=True)``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog + stdlib logging for the screener.

    Idempotent — repeated calls (e.g. from nested CLI subcommands) are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
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
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a configured structlog logger.

    Auto-configures with defaults on first call so library code can simply
    ``log = get_logger(__name__)`` without ordering concerns.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
