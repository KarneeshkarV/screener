"""Screen workflow Module behind the Click Adapter.

The workflow calls its collaborators (``scan``, history persistence, report
rendering, earnings enrichment) directly as module-level names. Tests exercise
edge cases by monkeypatching those names — the seams that actually touch the
network, the history DB, or the filesystem — rather than threading a bag of
injected callables through every call site.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from screener.cache import parse_ttl
from screener.criteria import resolve_criteria
from screener.screen_candidates import (
    DEFAULT_INTERVAL,
    ScreenStrategy,
    UnscreenableStrategyError,
    prefilter_filters,
    resolve_screen_strategy,
    resolve_screen_gates,
    resolve_universe_tickers,
    screen_candidates,
    screen_label,
    settings_fingerprint,
)
from screener.enrich import enrich_days_to_earnings, filter_earnings_buffer
from screener.history import diff, previous_run, save_run
from screener.scanner import scan
from screener.scoring import (
    DEFAULT_PRICE_ADJUSTMENT,
    OUTPUT_SCORE_COLUMN,
    PriceAdjustment,
    resolve_scorer,
)


LOG = logging.getLogger(__name__)

#: Upper bound on the prefilter scan, which exists only to stop an unbounded
#: vendor request - not to rank. It is well above the size of either market's
#: listed universe, so in practice the scan returns every name the prefilter
#: matched and the bar rule sees the whole field. When it does not,
#: :func:`_run_bar_screen` warns rather than quietly screening a short field.
_PREFILTER_CANDIDATE_CAP = 5000


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
    # Named universe or universe file for the exact path (D9). ``None`` keeps
    # the TradingView prefilter, which is the default. Only a criterion that
    # aliases a strategy has a bar rule to run, so this is refused otherwise.
    universe: str | None = None
    # Raise StaleDataError instead of serving stale cache when the live scan
    # fails. When ranking by a bar-derived setup_score, the same flag is
    # forwarded to the price fetcher, so a failed bar refresh also raises
    # instead of scoring leftover parquet. That extra refusal only fires
    # when refresh is also True; strict without refresh still only governs
    # the scan snapshot. Off by default so existing callers keep the
    # availability-first behaviour.
    strict: bool = False
    # Per-request socket timeout forwarded to requests.post by
    # tradingview_screener; None keeps the library's blocking default.
    timeout: float | None = None
    # Retry attempts for this scan; attempts x timeout is the real wall-clock
    # budget, so callers cap both together. None keeps the resilience default.
    retries: int | None = None
    # Hand the report back unrendered, as ``ScreenOutcome.render_report``, so
    # the caller decides when it runs. Rendering imports plotly and lays out
    # the whole page, which is about 0.4s the caller is otherwise blocked on
    # before it can show a result it already has. The CLI sets this and renders
    # after printing; every other caller keeps the report written for it.
    defer_report: bool = False
    # The candidate gates the user actually typed, as
    # ``resolve_strategy_profile`` overrides. Empty means "whatever the
    # strategy declares", which is what the rolling backtest would have used.
    # Built by ``screener.gate_options.gate_overrides`` so a flag cannot mean
    # one thing here and another on ``backtest-rolling``.
    gate_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Bar interval for the exact path. Only ``1d`` can honour the dated inputs
    # (fundamentals, earnings), which ``screen_candidates`` enforces.
    interval: str = DEFAULT_INTERVAL
    # Cap on the field before bars are fetched; 0 means no cap. Run-scoped,
    # like the universe itself, so no strategy profile carries it.
    max_universe: int = 0


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
    # When the payload behind this result was fetched - not when this workflow
    # returned. A cache hit carries the original fetch time. With --universe
    # there is no vendor payload, so it is when the bars were read.
    as_of: datetime
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    first_run: bool = False
    report_path: Path | None = None
    # Set only for a ``defer_report`` request: writes the HTML report to
    # :attr:`report_path` and returns it. ``None`` means the report is already
    # written, or that this is a CSV outcome, which has none.
    render_report: Callable[[], Path] | None = None

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


def _run_bar_screen(
    request: ScreenRequest,
    strategy: ScreenStrategy,
) -> tuple[int, pd.DataFrame, datetime]:
    """Screen by ``strategy``'s entry rule instead of by TradingView filters.

    Two modes, differing only in where the field comes from. With
    ``--universe`` the names come from ``screener.universes`` and no vendor
    field is consulted at all. Without it the TradingView prefilter narrows the
    field first, which is only sound because a prefilter may not drop a name
    the bar rule would have kept - the property
    ``tests/correctness`` pins. The cap on that scan can break the property on
    a field wider than it, so the two counts are compared and a short scan is
    warned about instead of being reported as a whole one.
    """
    warnings: list[str] = []
    signal_date = date.today()

    if request.universe:
        tickers = resolve_universe_tickers(request.universe, request.market)
        scanned = None
        total = len(tickers)
        # No vendor payload in this mode, so the freshness the outcome reports
        # is the local bar read, which happens now.
        fetched_at = datetime.now()
    else:
        total, scanned, fetched_at = scan(
            market=request.market,
            filters=prefilter_filters(strategy),
            # NOT ``request.limit``. The scan here is a field cut, not the
            # result: cutting it to the top ``-n`` names by raw volume would
            # drop names the bar rule keeps, which is precisely what a
            # prefilter may never do (D21). ``--limit`` applies to the
            # candidates the rule returns, and is applied there.
            limit=_PREFILTER_CANDIDATE_CAP,
            order_by="volume",
            detail=request.detail,
            cache_ttl=parse_ttl(request.cache_ttl, default=900),
            refresh=request.refresh,
            scorer=None,
            price_adjustment=request.price_adjustment,
            strict=request.strict,
            timeout=request.timeout,
            retries=request.retries,
        )
        tickers = [str(t) for t in scanned.get("ticker", pd.Series(dtype=str))]
        # Against the cap, not against ``len(tickers)``: the scan payload is
        # deduped (dual listings collapse) before it reaches here, so a row
        # count below ``total`` is the ordinary answer on any field carrying
        # one. The vendor returns ``min(total, cap)`` rows, so truncation
        # happened exactly when the match count exceeded the cap.
        if total > _PREFILTER_CANDIDATE_CAP:
            # The cap exists to bound the request, but a truncated scan is
            # still a prefilter that dropped names the bar rule never saw -
            # the one thing a prefilter may not do (D21). The scan orders by
            # volume, so what went missing is the low-volume tail, and nothing
            # downstream can recover it. Say so rather than report a partial
            # field as the whole one.
            warnings.append(
                f"prefilter scan returned {len(tickers)} of {total} matching "
                f"names (cap {_PREFILTER_CANDIDATE_CAP}, ordered by volume), "
                "so the low-volume tail was never evaluated against the bar "
                "rule and this result may be missing candidates. Re-run with "
                "--universe to screen the exact field."
            )

    df = screen_candidates(
        strategy,
        market=request.market,
        tickers=tickers,
        as_of=signal_date,
        scanned=scanned,
        limit=request.limit,
        order_by=request.order_by,
        refresh=request.refresh,
        price_adjustment=request.price_adjustment,
        strict=request.strict,
        gates=resolve_screen_gates(
            strategy, market=request.market, overrides=request.gate_overrides
        ),
        interval=request.interval,
        max_universe=request.max_universe,
        warnings=warnings,
    )
    for warning in warnings:
        LOG.warning("%s", warning)
    return total, df, fetched_at


#: Request fields that only the bar path can honour, mapped to the flag that
#: sets each. A snapshot screen ranks TradingView columns and never builds a
#: candidate panel, so there is nothing there for a gate to gate.
_BAR_PATH_ONLY: tuple[tuple[str, str], ...] = (
    ("gate_overrides", "gate flags"),
    ("interval", "--interval"),
    ("max_universe", "--max-universe"),
)


def _refuse_bar_path_options(request: ScreenRequest) -> None:
    """Refuse bar-path options on a criteria set that has no bar rule.

    Accepting them quietly would be the worse failure: the user would type
    ``--min-price`` and get a result that ignored it, with nothing to say so.
    """
    empty = ScreenRequest(
        market=request.market,
        criteria_names=request.criteria_names,
        limit=request.limit,
        order_by=request.order_by,
        output_csv=request.output_csv,
        detail=request.detail,
        refresh=request.refresh,
        cache_ttl=request.cache_ttl,
        report_path=request.report_path,
    )
    given = [
        flag
        for name, flag in _BAR_PATH_ONLY
        if getattr(request, name) != getattr(empty, name)
    ]
    if given:
        raise UnscreenableStrategyError(
            f"{', '.join(given)} need a criterion that names a strategy, because "
            f"only a strategy carries a bar rule to gate; {request.criteria_names} "
            "names TradingView filters only."
        )


def run_screen_workflow(request: ScreenRequest) -> ScreenOutcome:
    """Run the full non-Click screen lifecycle and return its outcome."""
    strategy = resolve_screen_strategy(request.criteria_names)
    if strategy is None:
        _refuse_bar_path_options(request)
    if request.universe and strategy is None:
        raise UnscreenableStrategyError(
            f"--universe needs a criterion that names a strategy, because only "
            f"a strategy carries a bar rule to evaluate; {request.criteria_names} "
            "names TradingView filters only, which have nothing to run against "
            "a local universe."
        )
    if strategy is not None:
        label = screen_label(
            request.criteria_names,
            strategy=strategy,
            universe=request.universe,
            # The gates are part of the question, so they are part of the
            # label: history must not diff a screen run with --min-price 50
            # against one run without it.
            fingerprint=settings_fingerprint(
                resolve_screen_gates(
                    strategy,
                    market=request.market,
                    overrides=request.gate_overrides,
                ),
                price_adjustment=request.price_adjustment,
                interval=request.interval,
            ),
        )
        return _finish_screen(request, label, *_run_bar_screen(request, strategy))

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

    return _finish_screen(request, selection.label, total, df, as_of)


def _finish_screen(
    request: ScreenRequest,
    label: str,
    total: int,
    df: pd.DataFrame,
    as_of: datetime,
) -> ScreenOutcome:
    """Everything after the candidate set is decided: enrich, persist, report.

    Shared by both paths on purpose. The two paths differ in how a name is
    selected and in nothing else, so the earnings filter, the history diff and
    the report must not be written twice.
    """
    # Earnings enrichment is opt-in and runs only on final result rows.
    if request.earnings or request.earnings_buffer is not None:
        df = enrich_days_to_earnings(df, request.market)
    if request.earnings_buffer is not None:
        df = filter_earnings_buffer(df, request.earnings_buffer)

    if request.output_csv:
        return ScreenOutcome(
            mode=ScreenMode.CSV,
            market=request.market,
            label=label,
            total=total,
            df=df,
            as_of=as_of,
        )

    run_id = save_run(request.market, label, total, df)
    prev = previous_run(request.market, label, run_id)
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

    def render() -> Path:
        return render_screen_report(
            df,
            total,
            request.market,
            label,
            generated_report,
            added=added,
            removed=removed,
            first_run=first_run,
            detail=request.detail,
            refresh=request.refresh,
            cache_ttl=request.cache_ttl,
            order_by=request.order_by,
        )

    if not request.defer_report:
        render()

    return ScreenOutcome(
        mode=ScreenMode.RESULTS,
        market=request.market,
        label=label,
        total=total,
        df=df,
        as_of=as_of,
        added=tuple(added),
        removed=tuple(removed),
        first_run=first_run,
        report_path=generated_report,
        render_report=render if request.defer_report else None,
    )


__all__ = [
    "ScreenMode",
    "ScreenOutcome",
    "ScreenRequest",
    "SignalRow",
    "run_screen_workflow",
]
