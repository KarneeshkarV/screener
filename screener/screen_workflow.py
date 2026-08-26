"""Screen workflow Module behind the Click Adapter.

The workflow calls its collaborators (``scan``, history persistence, report
rendering, earnings enrichment) directly as module-level names. Tests exercise
edge cases by monkeypatching those names — the seams that actually touch the
network, the history DB, or the filesystem — rather than threading a bag of
injected callables through every call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from screener.cache import parse_ttl
from screener.criteria import resolve_criteria
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.history import diff, previous_run, save_run
from screener.scanner import scan
from screener.scoring import (
    DEFAULT_PRICE_ADJUSTMENT,
    OUTPUT_SCORE_COLUMN,
    PriceAdjustment,
    resolve_scorer,
)


def temp_report_path(prefix: str) -> Path:
    """Lazy wrapper so the CSV path skips reporting helpers."""
    from screener.reporting import temp_report_path as _impl

    return _impl(prefix)


def render_screen_report(*args: Any, **kwargs: Any) -> Path:
    """Lazy wrapper: plotly lives behind screen_report, only imported on use."""
    from screener.commands.screen_report import render_screen_report as _impl

    return _impl(*args, **kwargs)


class ScreenMode(str, Enum):
    CSV = "csv"
    RESULTS = "results"


@dataclass(frozen=True)
class ScreenRequest:
    market: str
    criteria_names: tuple[str, ...]
    limit: int
    order_by: str
    output_csv: bool
    detail: bool
    refresh: bool
    cache_ttl: str
    report_path: Path | None
    open_report: bool = False
    earnings: bool = False
    # Drop final result rows whose next earnings date is within N calendar days.
    # ``None`` disables the filter. Unknown earnings dates are always kept.
    earnings_buffer: int | None = None
    # Price adjustment for bar-derived ranking scores. Same spelling as the
    # backtester's ``--price-adjustment``. Snapshot scorers ignore it.
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT
    # Raise StaleDataError instead of serving stale cache when the live scan
    # fails. Off by default so existing callers keep the availability-first
    # behaviour.
    strict: bool = False
    # Per-request socket timeout forwarded to requests.post by
    # tradingview_screener; None keeps the library's blocking default.
    timeout: float | None = None
    # Retry attempts for this scan; attempts x timeout is the real wall-clock
    # budget, so callers cap both together. None keeps the resilience default.
    retries: int | None = None


@dataclass(frozen=True)
class SignalRow:
    """One ranked screen row as plain Python types.

    Exists so a consumer of :meth:`ScreenOutcome.signals` can read results
    without importing pandas: every field is a ``str``/``int``/``float``/None,
    and the row itself is JSON-serializable. ``rank`` is 1-based row order;
    an absent or non-positive close is ``None``, never ``0.0``.
    """

    ticker: str
    rank: int
    score: float | None
    close: float | None


# Column spellings in the scan frame, mirroring how history.py maps the same
# columns into its persisted rows.
_TICKER_COLUMN = "ticker"
_SCORE_COLUMN = "setup_score"
_CLOSE_COLUMN = "close"


@dataclass(frozen=True)
class ScreenOutcome:
    mode: ScreenMode
    market: str
    label: str
    total: int
    df: pd.DataFrame
    # When the scan payload was fetched from the provider - not when this
    # workflow returned. A cache hit carries the original fetch time.
    as_of: datetime
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    first_run: bool = False
    report_path: Path | None = None

    def signals(self) -> list[SignalRow]:
        """The ranked rows as plain objects; no pandas needed to read them.

        Rank follows row order (the frame is already sorted by the workflow).
        Missing score/close columns degrade to ``None`` rather than raising,
        so a consumer gets usable output from any shaped frame.
        """
        rows: list[SignalRow] = []
        for _, record in self.df.iterrows():
            raw_ticker = record.get(_TICKER_COLUMN)
            if raw_ticker is None or pd.isna(raw_ticker):
                continue
            ticker = str(raw_ticker).strip()
            # A row with no ticker has no identity, so it cannot be acted on:
            # a consumer would carry a rank and a price for a name it cannot
            # place an order in. Skip it rather than emit ``ticker=""``, and
            # rank AFTER the skip so ranks stay dense and 1-based.
            if not ticker:
                continue
            rows.append(
                SignalRow(
                    ticker=ticker,
                    rank=len(rows) + 1,
                    score=_plain_float(record.get(_SCORE_COLUMN)),
                    # A zero or negative price is a data hole, not a price.
                    close=_plain_float(record.get(_CLOSE_COLUMN), positive=True),
                )
            )
        return rows


def _plain_float(value: Any, *, positive: bool = False) -> float | None:
    """Coerce a cell to ``float | None``, mirroring history's NULL handling."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    if positive and number <= 0:
        return None
    return number


def run_screen_workflow(request: ScreenRequest) -> ScreenOutcome:
    """Run the full non-Click screen lifecycle and return its outcome."""
    selection = resolve_criteria(request.criteria_names)
    # Only the ``setup_score`` ranking consumes a scorer, and resolving one can
    # refuse a criteria combination whose scores are incomparable. Skip the
    # resolution when the run sorts by a TradingView column, so a refusal fires
    # only for a run that would actually rank by the refused recipe.
    scorer = (
        resolve_scorer(request.criteria_names, strict=False)
        if request.order_by == OUTPUT_SCORE_COLUMN
        else None
    )

    total, df, as_of = scan(
        market=request.market,
        filters=selection.filters,
        limit=request.limit,
        order_by=request.order_by,
        detail=request.detail,
        cache_ttl=parse_ttl(request.cache_ttl, default=900),
        refresh=request.refresh,
        scorer=scorer,
        price_adjustment=request.price_adjustment,
        strict=request.strict,
        timeout=request.timeout,
        retries=request.retries,
    )

    # Earnings enrichment is opt-in and runs only on final result rows.
    if request.earnings or request.earnings_buffer is not None:
        df = enrich_days_to_earnings(df, request.market)
    if request.earnings_buffer is not None:
        df = filter_earnings_buffer(df, request.earnings_buffer)

    if request.output_csv:
        return ScreenOutcome(
            mode=ScreenMode.CSV,
            market=request.market,
            label=selection.label,
            total=total,
            df=df,
            as_of=as_of,
        )

    run_id = save_run(request.market, selection.label, total, df)
    prev = previous_run(request.market, selection.label, run_id)
    if prev is None:
        added: list[str] = []
        removed: list[str] = []
        first_run = True
    else:
        added, removed = diff(df, prev)
        first_run = False

    # Non-CSV always writes an HTML report (temp path when --report omitted).
    # render_screen_report / temp_report_path are lazy wrappers so the CSV path
    # (above) never imports plotly.
    generated_report = request.report_path
    if generated_report is None:
        generated_report = temp_report_path("screen")

    render_screen_report(
        df,
        total,
        request.market,
        selection.label,
        generated_report,
        added=added,
        removed=removed,
        first_run=first_run,
        detail=request.detail,
        refresh=request.refresh,
        cache_ttl=request.cache_ttl,
        order_by=request.order_by,
    )

    return ScreenOutcome(
        mode=ScreenMode.RESULTS,
        market=request.market,
        label=selection.label,
        total=total,
        df=df,
        as_of=as_of,
        added=tuple(added),
        removed=tuple(removed),
        first_run=first_run,
        report_path=generated_report,
    )


__all__ = [
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "SignalRow",
    "run_screen_workflow",
]
