"""Screen adapter for the shared price-only score layer.

Computes a :class:`screener.factors.PriceScoreSpec` from cached OHLCV bars for
the tickers a scan returned and takes the value at the last bar. The backtest
counterpart is :mod:`screener.strategies.factor_adapter`; both call
:func:`screener.factors.score_bars`, so today's screen number is literally the
last point of the series the backtester ranks on.

**Two columns, one ranking.** ``setup_score`` is the *within-scan percentile*
of that raw value, 0-100, and the recipe's ``aux_column`` (``mom_12_1`` for
12-1 momentum) carries the raw value itself, which is the number the backtest
ranks on. Percentile is a monotone transform, so both columns sort the scanned
names into the same order and select the same names; only the units differ.
The units matter because ``setup_score`` is the column the screen displays and
persists to ``run_rows.setup_score``, where downstream consumers threshold it -
``execution-trade`` vetoes a signal below its ``min_score``. Every other
scorer writes a 0-100 composite there, so writing a raw return in roughly -0.5
to +2 instead would silently re-scale a calibrated threshold, and two-decimal
display would collapse tightly clustered returns onto one printed score. The
percentile is taken over the scanned cross-section *after* the ineligible rows
are dropped, so a name nobody can trade never moves an eligible name's score.

Two deliberate properties:

* **Bars are fetched only for rows the TradingView filters already returned.**
  The adapter runs inside ``scanner.shape_scan_results``, after the scan, so
  the field is already cut to the scan's fetch limit rather than the whole
  market. For a bar-derived scorer that ceiling is ``max(limit * 5, 200)``,
  set in ``scanner.build_scanner_plan``. The extra rows exist so the
  eligibility floor, price-fetch outages, and NSE/BSE dedupe still leave
  ``limit`` names. A default screen downloads at most 200 tickers of daily
  bars. The fetcher's on-disk parquet cache is reused as-is. ``refresh``
  asks that cache to update; a failed download still merges leftover
  parquet so the ranking stays available. When the caller asked for
  ``strict`` and ``refresh`` together, that merge is refused:
  :class:`~screener.providers.StaleDataError` is raised instead of ranking
  on bars that were not actually refreshed. ``strict`` without ``refresh``
  does not change this path; it governs the TradingView snapshot.
* **Ineligible names are dropped, not "ranked last".** A name without enough
  history has no score, and a name whose raw value fails the recipe's own
  ``eligible_above`` floor is not a candidate; neither is filled with 0 and
  quietly sorted to the bottom where it is still selectable. The floor is the
  recipe's declaration, not this adapter's, so the screen's candidate set is
  the same rule the backtest strategy's entry expression gates on. A score can
  go missing three ways, and they are *not* the same failure, so each is
  counted apart. Too few bars means the name is ineligible. If the newest bar
  predates ``as_of`` by more than :data:`STALE_BAR_ALLOWANCE_DAYS`, the name
  is a dead listing, not a current value. An empty frame means the price
  provider failed. That last one is logged loudly, because a whole scan of
  empty frames renders as a bare "0 results" that reads like "nothing matched
  your filters".

**Adjustment must match the backtest.** ``score_bars`` reads ``close``, so the
number this adapter reports is only comparable to the backtest's number when
both sides adjust closes the same way. The backtester derives
``auto_adjust=(price_adjustment == "full")`` from its ``--price-adjustment``
flag, so under ``splits_only`` or ``none`` its closes keep dividends and its
``momentum_12_1`` is a different number from a dividend-adjusted screen's.
The screen offers the same ``--price-adjustment`` choice and passes it through
``scan``, ``apply_score``, and this adapter. In ``splits_only`` mode, this
adapter applies the same split-only panel transformation as the backtest. It
defaults to :data:`DEFAULT_PRICE_ADJUSTMENT` (``"full"``, the backtester's own
default), so a caller who says nothing keeps the old behaviour instead of
inheriting an adjustment silently from ``build_price_fetcher``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from screener.factors import PriceScoreSpec, eligible_mask, score_bars
from screener.scoring.components import percentile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.data import PriceFetcher

LOG = logging.getLogger(__name__)

# Trading days are ~5/7 of calendar days; pad generously so a lookback of N
# sessions is actually covered, plus a month of holidays//listing slack.
_CALENDAR_DAYS_PER_SESSION = 1.6
_CALENDAR_SLACK_DAYS = 45

TICKER_COLUMN = "ticker"

#: ``setup_score`` is reported on the same 0-100 scale as every snapshot
#: composite, so a threshold calibrated against one scorer keeps its meaning
#: against another. ``components.percentile`` returns [0, 1]; this scales it.
PERCENTILE_SCALE = 100.0

#: Same spelling as ``BacktestConfig.price_adjustment`` so the two sides of the
#: shared score layer are configured with one vocabulary.
PriceAdjustment = Literal["full", "splits_only", "none"]

#: Default when the caller names no adjustment. Fully adjusted closes
#: (``auto_adjust=True``), the backtester's own default. The screen's
#: ``--price-adjustment`` flag overrides it per run.
DEFAULT_PRICE_ADJUSTMENT: PriceAdjustment = "full"

_PRICE_ADJUSTMENTS: frozenset[str] = frozenset(("full", "splits_only", "none"))

#: Largest gap allowed between a symbol's newest bar and ``as_of``. A holiday
#: week plus a weekend still fits. A dead listing does not.
STALE_BAR_ALLOWANCE_DAYS = 10


def _fetch_start(as_of: date, lookback: int) -> date:
    span = int(lookback * _CALENDAR_DAYS_PER_SESSION) + _CALENDAR_SLACK_DAYS
    return (pd.Timestamp(as_of) - pd.Timedelta(days=span)).date()


def _stale_last_bar(series: pd.Series, as_of: date) -> bool:
    """True when the newest bar predates ``as_of`` by more than the allowance.

    An empty index, or one that is not a ``DatetimeIndex``, is never stale.
    Those frames score the way they always did.
    """
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return False
    try:
        gap_days = (pd.Timestamp(as_of) - index[-1]).days
    except TypeError:  # tz-aware index vs naive as_of. Leave the value alone.
        return False
    return gap_days > STALE_BAR_ALLOWANCE_DAYS


def _last_value(series: pd.Series, as_of: date) -> float:
    """Value at the newest bar, or NaN when that bar is too old to trade on.

    A renamed or suspended symbol can still return a long frame months after
    its coverage stopped. Ranking that frame would treat a dead listing as
    current. A stale newest bar becomes NaN, ineligible, the same as missing
    history.
    """
    if series.empty or _stale_last_bar(series, as_of):
        return float("nan")
    return float(series.iloc[-1])


def _resolve_price_adjustment(price_adjustment: str) -> bool:
    """Return ``auto_adjust`` for a ``price_adjustment`` spelling.

    Mirrors ``screener.backtester.cli_common.build_backtest_fetcher`` exactly:
    only ``"full"`` back-propagates dividends into the closes that
    ``score_bars`` reads.
    """
    if price_adjustment not in _PRICE_ADJUSTMENTS:
        raise ValueError(
            f"unknown price_adjustment {price_adjustment!r}; expected one of "
            f"{sorted(_PRICE_ADJUSTMENTS)} (the same spellings the backtester's "
            "--price-adjustment accepts, so both sides of the shared score "
            "layer adjust closes identically)"
        )
    return price_adjustment == "full"


def _log_missing_price_data(
    spec: PriceScoreSpec,
    *,
    total: int,
    no_price_data: int,
    short_history: int,
    stale_price_data: int = 0,
) -> None:
    """Report each reason a bar score went missing as its own failure.

    ``short_history`` and ``stale_price_data`` are intended and quiet. The
    fetch worked and the name is ineligible. ``short_history`` means too few
    bars. ``stale_price_data`` means history is long enough, but the newest
    bar predates the scan date by more than :data:`STALE_BAR_ALLOWANCE_DAYS`.
    That is what a renamed or suspended listing looks like. ``no_price_data``
    means the fetch returned nothing. Provider outage, rate limit, network
    failure, or a symbol yfinance does not carry. Unscored rows get dropped,
    so that case reaches the user only as a smaller result count.
    """
    if short_history:
        LOG.info(
            "%d/%d scanned tickers have too little history for scorer %r "
            "(needs %d sessions); dropped as ineligible.",
            short_history,
            total,
            spec.name,
            spec.required_lookback,
        )
    if stale_price_data:
        LOG.info(
            "%d/%d scanned tickers have a stale last bar for scorer %r. The "
            "fetch worked and history is long enough, but the newest bar "
            "predates the scan date by more than %d calendar days, so price "
            "coverage likely stopped. Dropped as untradeable instead of "
            "ranking on a months-old value.",
            stale_price_data,
            total,
            spec.name,
            STALE_BAR_ALLOWANCE_DAYS,
        )
    if not no_price_data:
        return
    message = (
        "%d/%d scanned tickers returned no price bars at all for scorer %r: "
        "the price provider failed, was rate limited, or does not list them. "
        "They are dropped unscored, so the result count understates the scan."
    )
    args = (no_price_data, total, spec.name)
    if no_price_data == total:
        LOG.error(
            "No price data for ANY scanned ticker. " + message + " Treat this "
            "as a price-provider outage, not as an empty screen.",
            *args,
        )
    else:
        LOG.warning(message, *args)


def _log_floor_drops(
    spec: PriceScoreSpec, *, total: int, below_floor: int, survivors: int
) -> None:
    """Report rows that carry a value but fail the recipe's ``eligible_above`` floor.

    A value at or below the floor is an eligibility drop, same as short
    history, so it logs at INFO. If the floor removes every remaining
    candidate, that is worse. The screen then renders as "N matches, showing
    0" over an empty table. That is the ambiguity this module's docstring
    exists to prevent, so that case logs at WARNING and names the rule that
    emptied the table.
    """
    if spec.eligible_above is None or not below_floor:
        return
    detail = (
        "%d/%d scanned tickers have a %s value at or below the recipe's "
        "eligible_above floor (%g). Dropped as ineligible."
    )
    args = (below_floor, total, spec.name, spec.eligible_above)
    if survivors == 0:
        LOG.warning(
            "No name passed the recipe's floor. " + detail + " This is not "
            '"nothing matched your filters". The scan matched, and every '
            "matched name failed the recipe's own eligibility floor.",
            *args,
        )
    else:
        LOG.info(detail, *args)


def bar_scores_for_tickers(
    tickers: Iterable[str],
    spec: PriceScoreSpec,
    *,
    market: str,
    as_of: date | None = None,
    fetcher: "PriceFetcher | None" = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    strict: bool = False,
) -> dict[str, float]:
    """Return ``{tv_ticker: score at the last bar}``; missing history -> NaN.

    ``price_adjustment`` uses the backtester's spelling and applies the same
    fetch and split-only transformations as the backtest.

    ``strict`` is forwarded to :func:`~screener.backtester.data.build_price_fetcher`
    along with ``refresh``. Together they refuse leftover bar cache after a
    failed download. ``strict`` without ``refresh`` is a no-op on this path.
    An injected ``fetcher`` is used as-is and does not see these flags.
    """
    from screener.backtester.data import build_price_fetcher, tv_to_yf

    auto_adjust = _resolve_price_adjustment(price_adjustment)
    symbols = [str(t) for t in tickers if isinstance(t, str) or pd.notna(t)]
    if not symbols:
        return {}
    resolved_as_of = as_of or date.today()
    yf_by_tv = {tv: tv_to_yf(tv, market) for tv in symbols}
    active = fetcher or build_price_fetcher(
        refresh=refresh, auto_adjust=auto_adjust, strict=strict
    )
    panel = active.fetch(
        list(dict.fromkeys(yf_by_tv.values())),
        _fetch_start(resolved_as_of, spec.required_lookback),
        resolved_as_of,
    )
    if price_adjustment == "splits_only":
        from screener.backtester.data import (
            apply_splits_only_adjustment,
            warn_unadjustable_fmp_frames,
        )

        warn_unadjustable_fmp_frames(panel)
        panel = apply_splits_only_adjustment(panel)
    scores: dict[str, float] = {}
    no_price_data = 0
    short_history = 0
    stale_price_data = 0
    for tv, yf_symbol in yf_by_tv.items():
        bars = panel.get(yf_symbol)
        if bars is None or bars.empty or "close" not in bars.columns:
            # Empty/absent frame: the fetch failed, this is not an eligibility
            # verdict about the name.
            no_price_data += 1
            scores[tv] = float("nan")
            continue
        scored_series = score_bars(spec, bars)
        score = _last_value(scored_series, resolved_as_of)
        if score != score:  # NaN despite real bars. Split stale from short history.
            if _stale_last_bar(scored_series, resolved_as_of):
                stale_price_data += 1
            else:
                short_history += 1
        scores[tv] = score
    _log_missing_price_data(
        spec,
        total=len(scores),
        no_price_data=no_price_data,
        short_history=short_history,
        stale_price_data=stale_price_data,
    )
    return scores


def apply_bar_score(
    df: pd.DataFrame,
    spec: PriceScoreSpec,
    *,
    market: str,
    output_column: str,
    as_of: date | None = None,
    fetcher: "PriceFetcher | None" = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    strict: bool = False,
) -> pd.DataFrame:
    """Write the ranked ``output_column`` plus the raw ``aux_column``, keeping candidates.

    ``output_column`` gets the cross-sectional percentile of ``spec`` over the
    surviving rows, 0-100; ``spec.aux_column`` gets the raw recipe value, which
    is the number the backtest ranks on. Ranking is identical either way, so
    the scaling changes the reported units and nothing about the selection.

    Dropping (rather than ``fillna(0)``) is the unified layer's eligibility
    policy: a name with too little history, or one below the recipe's
    ``eligible_above`` floor, is not a candidate rather than the worst-ranked
    candidate. Both drops happen before the percentile, so an ineligible name
    never shifts an eligible one's score. A row dropped because the price fetch
    failed is logged by :func:`bar_scores_for_tickers` instead, since that is
    an outage rather than an eligibility verdict.

    ``price_adjustment`` must match the backtest's ``--price-adjustment`` for
    the raw value to be the same number; see the module docstring.

    ``strict`` is forwarded with ``refresh`` to the price fetcher; see
    :func:`bar_scores_for_tickers`.
    """
    if df.empty:
        empty = df.assign(**{output_column: pd.Series(dtype=float)})
        if spec.aux_column:
            empty = empty.assign(**{spec.aux_column: pd.Series(dtype=float)})
        return empty
    if TICKER_COLUMN not in df.columns:
        raise KeyError(
            f"bar-derived scorer {spec.name!r} needs a {TICKER_COLUMN!r} column "
            "to resolve price history for the scanned rows"
        )
    scores = bar_scores_for_tickers(
        df[TICKER_COLUMN].tolist(),
        spec,
        market=market,
        as_of=as_of,
        fetcher=fetcher,
        refresh=refresh,
        price_adjustment=price_adjustment,
        strict=strict,
    )
    mapped: Any = df[TICKER_COLUMN].map(scores)
    raw = pd.to_numeric(mapped, errors="coerce").astype(float)
    scored = df if not spec.aux_column else df.assign(**{spec.aux_column: raw})
    candidates = eligible_mask(spec, raw)
    # A value that fails the floor is the third way a name leaves the ranking.
    # Count those apart. A scan that drops every row must not look like a
    # silent empty table.
    _log_floor_drops(
        spec,
        total=len(raw),
        below_floor=int((raw.notna() & ~candidates).sum()),
        survivors=int(candidates.sum()),
    )
    scored = scored[candidates]
    # Percentile over the candidates only, on the same [0, 1] rank with
    # average tie-breaking every snapshot recipe uses, scaled to 0-100. A lone
    # survivor therefore scores 100, exactly as a one-row snapshot scan does.
    ranked = PERCENTILE_SCALE * percentile(raw[candidates])
    return scored.assign(**{output_column: ranked})


__all__ = [
    "DEFAULT_PRICE_ADJUSTMENT",
    "PERCENTILE_SCALE",
    "TICKER_COLUMN",
    "PriceAdjustment",
    "apply_bar_score",
    "bar_scores_for_tickers",
]
