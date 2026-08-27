"""Stable public API for embedding the screener in another codebase.

Everything re-exported here is supported for outside callers. Anything else
under ``screener.`` is internal: it may move, change signature, or disappear in
any commit. Import from ``screener`` (or ``screener.api``) and nothing deeper.

Two entry points, in increasing order of control:

``screen(...)``
    One keyword call that mirrors the ``screener screen`` CLI defaults and
    returns a :class:`~screener.screen_workflow.ScreenOutcome`.

``run_screen_workflow(ScreenRequest(...))``
    The underlying workflow, when you want to build the request yourself.

Side effects are opt-in. The default ``persist=False`` returns the result frame
and writes nothing: no ``~/.screener/history.db`` row and no HTML report. Pass
``persist=True`` to get the CLI's full behaviour, which records the run and
renders a report (and so pulls in the ``report`` extra).
"""

from __future__ import annotations

from pathlib import Path

from screener.providers import StaleDataError
from screener.screen_workflow import (
    ScreenMode,
    ScreenOutcome,
    ScreenRequest,
    SignalRow,
    run_screen_workflow,
)


def list_criteria() -> list[str]:
    """Names accepted by ``criteria_names``, e.g. ``["ema", "breakout", ...]``."""
    from screener.criteria import registry

    return sorted(registry.names())


def list_markets() -> list[str]:
    """Names accepted by ``market``, e.g. ``["india", "us"]``."""
    from screener.markets import MARKETS

    return sorted(MARKETS)


def screen(
    *,
    market: str = "us",
    criteria: str | tuple[str, ...] = ("ema",),
    limit: int = 50,
    order_by: str = "setup_score",
    detail: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
    persist: bool = False,
    report_path: Path | str | None = None,
    earnings: bool = False,
    earnings_buffer: int | None = None,
    strict: bool = False,
    timeout: float | None = None,
    retries: int | None = None,
) -> ScreenOutcome:
    """Run one screen and return its outcome.

    Defaults match the ``screener screen`` CLI, except ``persist``, which is
    off so that an embedded call has no side effects. ``outcome.df`` is the
    result frame in both modes; ``outcome.as_of`` is when the scan data was
    fetched from the provider (the original fetch time on a cache hit).

    ``criteria`` accepts a single name for convenience; it is normalised to a
    tuple before it reaches the workflow.

    ``strict=True`` demands fresh-or-error. If the live scan fails, raise
    :class:`StaleDataError` instead of silently serving stale cache. When
    ``refresh=True`` as well and ranking uses a bar-derived ``setup_score``,
    the same refusal applies to the price history behind the ranking: a
    failed bar download that would otherwise merge with on-disk cache
    raises rather than scoring leftover bars. ``strict`` without
    ``refresh`` still only governs the scan snapshot. The default keeps
    the availability-first behaviour of the CLI.

    ``timeout`` caps each TradingView request in seconds (forwarded to
    ``requests.post``; ``None`` blocks indefinitely). ``retries`` overrides
    the retry attempts for the scan - attempts x timeout is the real
    wall-clock budget, so cap both together.

    Raises:
        StaleDataError: ``strict=True`` and no fresh scan (or, with
            ``refresh=True``, no refreshed bars behind a bar-derived ranking)
            could be fetched.
        KeyError: an unknown criterion name, listing the known ones.
        ValueError: ``earnings_buffer`` is negative, or ``report_path`` was
            given without ``persist=True`` (nothing would be written).
    """
    if isinstance(criteria, str):
        criteria = (criteria,)
    if earnings_buffer is not None and earnings_buffer < 0:
        raise ValueError("earnings_buffer must be >= 0.")
    if report_path is not None and not persist:
        raise ValueError(
            "report_path requires persist=True; the no-side-effect path "
            "renders no report."
        )

    request = ScreenRequest(
        market=market,
        criteria_names=tuple(criteria),
        limit=int(limit),
        order_by=order_by,
        # ScreenMode.CSV is the workflow's no-side-effect path: it returns the
        # frame before the history write and the report render.
        output_csv=not persist,
        detail=detail,
        refresh=refresh,
        cache_ttl=cache_ttl,
        report_path=Path(report_path) if report_path is not None else None,
        open_report=False,
        earnings=earnings,
        earnings_buffer=earnings_buffer,
        strict=strict,
        timeout=timeout,
        retries=retries,
    )
    return run_screen_workflow(request)


__all__ = [
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "SignalRow",
    "StaleDataError",
    "list_criteria",
    "list_markets",
    "run_screen_workflow",
    "screen",
]
