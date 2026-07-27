"""Local bar-store-backed scanner: the offline twin of the TradingView path.

The default ``screener screen`` runs TradingView server-side filters. This
module evaluates the *same* :mod:`screener.criteria` registry — the identical
``col(...) > ...`` filter expressions — against our own minute bars, so that
intraday screens are reproducible offline and inside the backtester. Bars are
loaded from the interval-partitioned store (:func:`bar_store.load_bars`), the
canonical 1m series is downsampled locally
(:func:`price_frames.resample_intraday_bars`), and a small interpreter maps each
TradingView ``FilterOperationDict`` onto locally computed features.

Field conventions (documented so the backtester can reproduce them exactly):

* ``close`` — last bar close on the requested interval.
* ``volume`` — the current (last) session's cumulative volume.
* ``change`` — percent move of the last close vs. the previous session's close
  (the day's move so far); ``NaN`` when only one session is stored.
* ``EMA5/EMA20/EMA100/EMA200`` — standard EMAs (``span=n``, ``adjust=False``) of
  the interval close series, last value. Intraday intervals therefore carry
  interval-native EMAs, which is the point of an intraday screen.
* ``RSI`` — Wilder RSI(14) via :func:`indicators.frames.wilder_rsi` (the same
  primitive the backtester's event scanners use).
* ``average_volume_10d_calc`` — mean total session volume over the up-to-10
  sessions *before* the current one; ``relative_volume_10d_calc`` is the current
  session volume divided by it.
* ``price_52_week_high`` — highest high in the trailing 52 weeks ending at the
  frame's last timestamp (not the entire archive).

Criteria that reference fields the bar store cannot compute (fundamentals such
as ``price_earnings_ttm``) raise :class:`LocalScanUnsupported` so the CLI can
tell the user the criterion is TradingView-only.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from screener.backtester.bar_store import load_bars, stored_symbols
from screener.backtester.price_frames import resample_intraday_bars
from screener.backtester.sessions import session_dates
from screener.indicators.frames import wilder_rsi
from screener.markets import get_market
from screener.scanner import (
    DEFAULT_COLUMNS,
    DETAIL_COLUMNS,
    SETUP_SCORE_COLUMNS,
    shape_scan_results,
)


class LocalScanUnsupported(ValueError):
    """A criterion references a field the local bar store cannot compute."""


_EMA_SPANS = (5, 20, 100, 200)

# Fundamentals the TradingView scanner selects but the bar store cannot derive;
# emitted as NaN so ``--detail`` output keeps the same columns as the TV path.
_FUNDAMENTAL_FIELDS = (
    "price_earnings_ttm",
    "return_on_equity",
    "dividend_yield_recent",
    "debt_to_equity",
    "market_cap_basic",
)


def compute_features(frame: pd.DataFrame, market_tz: str) -> dict[str, float]:
    """Compute the last-bar TradingView-equivalent features for one series."""
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce")
    volume = pd.to_numeric(
        frame.get("volume", pd.Series(index=frame.index)), errors="coerce"
    )

    index = pd.DatetimeIndex(frame.index)
    labels = session_dates(index, market_tz)
    volume_by_session = volume.groupby(labels).sum()
    close_by_session = close.groupby(labels).last()

    current_volume = float(volume_by_session.iloc[-1])
    prior_volume = volume_by_session.iloc[:-1].tail(10)
    average_volume = float(prior_volume.mean()) if len(prior_volume) else float("nan")
    relative_volume = (
        current_volume / average_volume
        if average_volume and not np.isnan(average_volume)
        else float("nan")
    )

    last_close = float(close.iloc[-1])
    if len(close_by_session) >= 2:
        prev_close = float(close_by_session.iloc[-2])
        change = (last_close / prev_close - 1.0) * 100.0 if prev_close else float("nan")
    else:
        change = float("nan")

    # Trailing 52-week high ending at the frame's last bar — not the entire
    # archive (a multi-year store would otherwise poison the feature forever).
    last_ts = index[-1]
    high_52w = high.loc[high.index >= last_ts - pd.Timedelta(weeks=52)]
    price_52_week_high = float(high_52w.max()) if len(high_52w) else float("nan")

    features: dict[str, float] = {
        "close": last_close,
        "volume": current_volume,
        "change": change,
        "relative_volume_10d_calc": relative_volume,
        "average_volume_10d_calc": average_volume,
        "RSI": float(wilder_rsi(close, 14).iloc[-1]),
        "price_52_week_high": price_52_week_high,
    }
    for span in _EMA_SPANS:
        features[f"EMA{span}"] = float(
            close.ewm(span=span, adjust=False).mean().iloc[-1]
        )
    return features


def _resolve(value: Any, features: dict[str, float]) -> float:
    """Resolve a filter operand: a field name into its feature, else a literal."""
    if isinstance(value, str):
        if value not in features:
            raise LocalScanUnsupported(value)
        return features[value]
    return float(value)


def _passes_filter(features: dict[str, float], filt: dict[str, Any]) -> bool:
    """Evaluate one TradingView ``FilterOperationDict`` against local features.

    A ``NaN`` on either side fails the comparison, mirroring how the server-side
    filter excludes rows with missing values.
    """
    operation = filt["operation"]
    left = _resolve(filt["left"], features)
    right = filt["right"]
    if np.isnan(left):
        return False

    if operation in {"above%", "below%"}:
        column, pct = right
        threshold = _resolve(column, features) * float(pct)
        if np.isnan(threshold):
            return False
        return left > threshold if operation == "above%" else left < threshold
    if operation in {"in_range%", "not_in_range%", "in_range", "not_in_range"}:
        column, low, *rest = right
        if operation.endswith("%"):
            base = _resolve(column, features)
            lower, upper = base * float(low), base * float(rest[0])
        else:
            lower, upper = _resolve(column, features), _resolve(low, features)
        if np.isnan(lower) or np.isnan(upper):
            return False
        inside = lower <= left <= upper
        return inside if operation in {"in_range%", "in_range"} else not inside

    threshold = _resolve(right, features)
    if np.isnan(threshold):
        return False
    return {
        "greater": left > threshold,
        "egreater": left >= threshold,
        "less": left < threshold,
        "eless": left <= threshold,
        "equal": left == threshold,
        "nequal": left != threshold,
    }[operation]


def passes_all(features: dict[str, float], filters: list[Any]) -> bool:
    """Return True when every filter (AND-joined, like TradingView) passes."""
    return all(_passes_filter(features, filt) for filt in filters)


def _output_columns(order_by: str, detail: bool) -> list[str]:
    """Mirror ``scanner.build_scanner_plan`` column selection for local rows."""
    columns = list(DEFAULT_COLUMNS)
    if detail:
        columns.extend(column for column in DETAIL_COLUMNS if column not in columns)
    if order_by == "setup_score":
        columns.extend(
            column for column in SETUP_SCORE_COLUMNS if column not in columns
        )
    return columns


def _row(symbol: str, features: dict[str, float], columns: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"name": symbol, "description": ""}
    for column in columns:
        if column in ("name", "description"):
            continue
        if column in _FUNDAMENTAL_FIELDS:
            row[column] = float("nan")
        else:
            row[column] = features.get(column, float("nan"))
    return row


def local_scan(
    *,
    market: str,
    filters: list[Any],
    interval: str = "5m",
    symbols: Optional[list[str]] = None,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
    root: Optional[Any] = None,
) -> tuple[int, pd.DataFrame]:
    """Evaluate ``filters`` over the local bar store, shaped like ``scanner.scan``.

    Loads the canonical 1m series for every stored symbol (or ``symbols`` when
    given), downsamples to ``interval`` when coarser, evaluates the criteria, and
    returns ``(total_matches, shaped_frame)`` with the same column contract as the
    TradingView path so history persistence and reporting are unchanged.
    """
    market_tz = get_market(market).timezone
    resolved = stored_symbols(market, "1m", root=root) if symbols is None else symbols
    columns = _output_columns(order_by, detail)

    rows: list[dict[str, Any]] = []
    for symbol in resolved:
        frame = load_bars(symbol, market=market, interval="1m", root=root)
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        if interval != "1m":
            frame = resample_intraday_bars(frame, interval, market_tz)
        if frame.empty:
            continue
        features = compute_features(frame, market_tz)
        if not passes_all(features, filters):
            continue
        rows.append(_row(symbol, features, columns))

    frame_out = pd.DataFrame(rows, columns=columns)
    total = len(frame_out)
    shaped = shape_scan_results(
        frame_out, limit=limit, order_by=order_by, detail=detail
    )
    return total, shaped


__all__ = [
    "LocalScanUnsupported",
    "compute_features",
    "local_scan",
    "passes_all",
]
