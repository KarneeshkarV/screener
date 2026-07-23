"""Price-frame schema, normalization, and range operations."""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
CORPORATE_ACTION_COLUMNS = ["dividend", "split_factor", "stock_splits"]

# Intraday intervals a stored 1m series can be downsampled into locally, with
# the bucket width in minutes. Buckets are anchored at each session's first
# bar (see ``resample_intraday_bars``), so the result matches how providers
# label their native bars (US 1h: 09:30/10:30/... ET; India 30m: 09:15/09:45
# IST) — a plain clock-aligned resample would land on :00/:30 boundaries
# instead.
RESAMPLE_MINUTES_FROM_1M = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}


def naive_normalized_index(index: pd.Index, interval: str = "1d") -> pd.DatetimeIndex:
    """Return the canonical tz-naive daily or UTC-intraday index."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    if index.tz is not None:
        if interval == "1d":
            index = index.tz_localize(None)
        else:
            index = index.tz_convert("UTC").tz_localize(None)
    return index.normalize() if interval == "1d" else index


def empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OHLCV_COLUMNS,
        index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
    )


def normalize_price_frame(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Normalize one provider frame to the backtester's canonical schema."""
    if df is None or df.empty:
        return empty_ohlcv_frame()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)
    rename = {column: column.lower().replace(" ", "_") for column in df.columns}
    df = df.rename(columns=rename)
    out = df[[column for column in OHLCV_COLUMNS if column in df.columns]].copy()
    if "adj_close" in df.columns:
        out["adj_close"] = df["adj_close"]
    if "dividends" in df.columns:
        out["dividend"] = df["dividends"].fillna(0.0).astype(float)
    elif "dividend" in df.columns:
        out["dividend"] = df["dividend"].fillna(0.0).astype(float)
    if "stock_splits" in df.columns:
        splits = df["stock_splits"].fillna(0.0).astype(float)
        factor = splits.replace(0.0, 1.0)[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        out["split_factor"] = factor.astype(float)
        out["stock_splits"] = splits
    out.index = naive_normalized_index(out.index, interval)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    price_columns = [column for column in OHLCV_COLUMNS if column in out.columns]
    return out.dropna(subset=price_columns) if price_columns else out


def apply_splits_only_adjustment(
    bars_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Back-adjust OHLC, volume, and dividends using explicit split factors."""
    adjusted_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker, frame in bars_by_ticker.items():
        if frame is None or frame.empty or "split_factor" not in frame.columns:
            adjusted_by_ticker[ticker] = frame
            continue
        factor = frame["split_factor"].astype(float)
        if bool((factor == 1.0).all()):
            adjusted_by_ticker[ticker] = frame
            continue
        adjusted = frame.copy()
        for column in ("open", "high", "low", "close", "dividend"):
            if column in adjusted.columns:
                adjusted[column] = adjusted[column].astype(float) / factor
        if "volume" in adjusted.columns:
            adjusted["volume"] = adjusted["volume"].astype(float) * factor
        adjusted_by_ticker[ticker] = adjusted
    return adjusted_by_ticker


def warn_unadjustable_fmp_frames(
    bars_by_ticker: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Warn when FMP frames lack the split factors needed for adjustment."""
    unadjusted = [
        ticker
        for ticker, frame in bars_by_ticker.items()
        if frame is not None and not frame.empty and "split_factor" not in frame.columns
    ]
    if unadjusted:
        from screener.logging_config import get_logger

        get_logger(__name__).warning(
            "fmp_unadjusted_in_splits_only",
            reason=(
                "FMP frames lack a Stock Splits column; splits cannot be "
                "reliably recovered from adj_close (splits+dividends are "
                "conflated), so these tickers are left split-unadjusted"
            ),
            tickers=unadjusted[:20],
            count=len(unadjusted),
        )
    return bars_by_ticker


def merge_price_frames(
    existing: Optional[pd.DataFrame], new: pd.DataFrame, interval: str = "1d"
) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], axis=0)
    if merged.empty:
        return merged
    merged.index = naive_normalized_index(merged.index, interval)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def inclusive_fetch_bounds(
    start: date, end: date, interval: str = "1d"
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if interval != "1d" and end_timestamp == end_timestamp.normalize():
        end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return start_timestamp, end_timestamp


def frame_has_range(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1d",
) -> bool:
    del interval  # Range tolerance is identical for daily and intraday frames.
    if frame is None or frame.empty:
        return False
    in_range = frame.loc[(frame.index >= start) & (frame.index <= end)]
    return (
        not in_range.empty
        and in_range.index.min() <= start + pd.Timedelta(days=3)
        and in_range.index.max() >= end - pd.Timedelta(days=3)
    )


def resample_intraday_bars(
    frame: pd.DataFrame, interval: str, market_tz: str
) -> pd.DataFrame:
    """Downsample 1m bars to a coarser intraday interval, anchored per session.

    Buckets are anchored at the first bar of each exchange session (session
    labels come from :func:`screener.backtester.sessions.session_dates` on the
    naive-UTC index), so US 1h bars land on 09:30/10:30/... ET and India 30m
    bars on 09:15/09:45 IST — matching how providers label native bars.
    Half-day sessions simply produce fewer buckets. Aggregation is the standard
    OHLCV roll-up (open=first, high=max, low=min, close=last, volume=sum);
    ``adj_close`` takes last and ``dividend``/``stock_splits`` take sum when
    present.

    When a session's first stored bar is late (partial archive), that session's
    buckets anchor at the first *stored* bar instead of the true session open —
    labels can drift by less than one bucket for that session only.
    """
    minutes = RESAMPLE_MINUTES_FROM_1M.get(interval)
    if minutes is None:
        raise ValueError(
            f"cannot resample to interval {interval!r}; expected one of "
            f"{sorted(RESAMPLE_MINUTES_FROM_1M)}"
        )
    if frame is None or frame.empty:
        return empty_ohlcv_frame()

    from screener.backtester.sessions import session_dates

    index = pd.DatetimeIndex(frame.index)
    ts_ns = index.to_numpy().view("i8")
    labels = session_dates(index, market_tz)
    n = len(index)
    session_change = np.empty(n, dtype=bool)
    session_change[0] = True
    session_change[1:] = labels[1:] != labels[:-1]
    session_id = np.cumsum(session_change) - 1
    session_first_ns = ts_ns[session_change][session_id]
    bucket = (ts_ns - session_first_ns) // pd.Timedelta(minutes=minutes).value
    # Unique per (session, bucket); non-decreasing because the index is sorted.
    key = session_id.astype(np.int64) * (1 << 24) + bucket
    first_in_bucket = np.empty(n, dtype=bool)
    first_in_bucket[0] = True
    first_in_bucket[1:] = key[1:] != key[:-1]

    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "adj_close": "last",
        "dividend": "sum",
        "stock_splits": "sum",
        "split_factor": "last",
    }
    agg = {col: fn for col, fn in aggregations.items() if col in frame.columns}
    out = frame.groupby(key, sort=False).agg(agg)
    out.index = pd.DatetimeIndex(index[first_in_bucket])
    return out


def split_yfinance_download(
    raw: pd.DataFrame, tickers: list[str], interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Split yfinance's single- or multi-ticker response into normalized frames."""
    if raw is None or raw.empty:
        return {ticker: empty_ohlcv_frame() for ticker in tickers}
    if not isinstance(raw.columns, pd.MultiIndex):
        ticker = tickers[0] if tickers else ""
        return {ticker: normalize_price_frame(raw, interval)}

    frames: dict[str, pd.DataFrame] = {}
    level_values = [
        set(raw.columns.get_level_values(level)) for level in range(raw.columns.nlevels)
    ]
    for ticker in tickers:
        frame = pd.DataFrame()
        for level, values in enumerate(level_values):
            if ticker not in values:
                continue
            selected = raw.xs(ticker, level=level, axis=1, drop_level=True)
            frame = selected.to_frame() if isinstance(selected, pd.Series) else selected
            break
        frames[ticker] = normalize_price_frame(frame, interval)
    return frames


__all__ = [
    "CORPORATE_ACTION_COLUMNS",
    "OHLCV_COLUMNS",
    "RESAMPLE_MINUTES_FROM_1M",
    "apply_splits_only_adjustment",
    "empty_ohlcv_frame",
    "frame_has_range",
    "inclusive_fetch_bounds",
    "merge_price_frames",
    "naive_normalized_index",
    "normalize_price_frame",
    "resample_intraday_bars",
    "split_yfinance_download",
    "warn_unadjustable_fmp_frames",
]
