"""Screen adapter for the shared price-only score layer.

Computes a :class:`screener.factors.PriceScoreSpec` from cached OHLCV bars for
the tickers a scan returned, takes the value at the last bar, and writes it as
``setup_score``. The backtest counterpart is
:mod:`screener.strategies.factor_adapter`; both call
:func:`screener.factors.score_bars`, so today's screen score is literally the
last point of the series the backtester ranks on.

Two deliberate properties:

* **Bars are fetched only for rows the TradingView filters already returned.**
  The adapter runs inside ``scanner.shape_scan_results``, after the scan, so
  the field is already cut to the scan's fetch limit rather than the whole
  market. The fetcher's on-disk parquet cache is reused as-is.
* **NaN means ineligible, not "rank last".** A name without enough history has
  no score and is dropped from the result, instead of being filled with 0 and
  quietly sorted to the bottom where it is still selectable.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from screener.factors import PriceScoreSpec, score_bars

if TYPE_CHECKING:  # pragma: no cover - typing only
    from screener.backtester.data import PriceFetcher

# Trading days are ~5/7 of calendar days; pad generously so a lookback of N
# sessions is actually covered, plus a month of holidays//listing slack.
_CALENDAR_DAYS_PER_SESSION = 1.6
_CALENDAR_SLACK_DAYS = 45

TICKER_COLUMN = "ticker"


def _fetch_start(as_of: date, lookback: int) -> date:
    span = int(lookback * _CALENDAR_DAYS_PER_SESSION) + _CALENDAR_SLACK_DAYS
    return (pd.Timestamp(as_of) - pd.Timedelta(days=span)).date()


def _last_value(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.iloc[-1])


def bar_scores_for_tickers(
    tickers: Iterable[str],
    spec: PriceScoreSpec,
    *,
    market: str,
    as_of: date | None = None,
    fetcher: "PriceFetcher | None" = None,
    refresh: bool = False,
) -> dict[str, float]:
    """Return ``{tv_ticker: score at the last bar}``; missing history -> NaN."""
    from screener.backtester.data import build_price_fetcher, tv_to_yf

    symbols = [str(t) for t in tickers if isinstance(t, str) or pd.notna(t)]
    if not symbols:
        return {}
    resolved_as_of = as_of or date.today()
    yf_by_tv = {tv: tv_to_yf(tv, market) for tv in symbols}
    active = fetcher or build_price_fetcher(refresh=refresh)
    panel = active.fetch(
        list(dict.fromkeys(yf_by_tv.values())),
        _fetch_start(resolved_as_of, spec.required_lookback),
        resolved_as_of,
    )
    scores: dict[str, float] = {}
    for tv, yf_symbol in yf_by_tv.items():
        bars = panel.get(yf_symbol)
        if bars is None or bars.empty or "close" not in bars.columns:
            scores[tv] = float("nan")
            continue
        scores[tv] = _last_value(score_bars(spec, bars))
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
) -> pd.DataFrame:
    """Write ``output_column`` from ``spec`` and drop rows with no score.

    Dropping (rather than ``fillna(0)``) is the unified layer's NaN policy: a
    name with too little history is ineligible, not the worst-ranked name.
    """
    if df.empty:
        return df.assign(**{output_column: pd.Series(dtype=float)})
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
    )
    mapped: Any = df[TICKER_COLUMN].map(scores)
    scored = df.assign(
        **{output_column: pd.to_numeric(mapped, errors="coerce").astype(float)}
    )
    return scored[scored[output_column].notna()]


__all__ = ["TICKER_COLUMN", "apply_bar_score", "bar_scores_for_tickers"]
