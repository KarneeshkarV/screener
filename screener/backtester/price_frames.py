"""Price-frame schema, normalization, and range operations."""

from __future__ import annotations

from datetime import date

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
CORPORATE_ACTION_COLUMNS = ["dividend", "split_factor", "stock_splits"]


def naive_normalized_index(index: pd.Index, interval: str = "1d") -> pd.DatetimeIndex:
    """Return the canonical tz-naive daily or UTC-intraday index.

    The result never carries a ``freq``. Under pandas 3 ``normalize`` *infers*
    one from the values, so an index that had none acquired ``BusinessDay`` or
    ``Day`` purely from the shape of the data it happened to hold. No price
    source supplies that, nothing in the codebase reads it, and it is not
    stored in the parquet cache - so leaving it on made a frame's identity
    depend on which path it was read through.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    if index.tz is not None:
        if interval == "1d":
            index = index.tz_localize(None)
        else:
            index = index.tz_convert("UTC").tz_localize(None)
    if interval == "1d":
        index = index.normalize()
    return drop_index_freq(index)


def drop_index_freq(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``index`` without its inferred frequency, allocating only if set."""
    if index.freq is None:
        return index
    return pd.DatetimeIndex(index.to_numpy(copy=False), name=index.name)


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
    existing: pd.DataFrame | None, new: pd.DataFrame, interval: str = "1d"
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


def _snapped(index: pd.DatetimeIndex, bound: pd.Timestamp, side: str) -> pd.Timestamp:
    """``bound`` in the index's own datetime resolution, without moving any row.

    ``searchsorted`` refuses a bound it cannot convert losslessly, and an
    intraday end bound carries a nanosecond offset while a cache entry may be
    stored in microseconds or seconds. A boolean mask has no such trouble, so
    the snap has to keep the same rows: ``index >= bound`` is unchanged by
    rounding the lower bound *up*, and ``index <= bound`` by rounding the upper
    bound *down*, because the discarded sub-unit part cannot match any row.
    """
    if bound.unit == index.unit:
        return bound
    snapped = bound.ceil(index.unit) if side == "left" else bound.floor(index.unit)
    return snapped.as_unit(index.unit)


def _range_bounds(
    index: pd.Index, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[int, int] | None:
    """Row positions of ``[start, end]`` in a sorted datetime index, or None.

    None means the caller has to fall back to a boolean mask: the index is not
    a sorted ``DatetimeIndex``, so a position pair cannot describe the answer.
    A price frame is sorted by construction - every path that builds one ends
    in ``sort_index`` - so the fallback is for frames from somewhere else.
    """
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing:
        return None
    return (
        int(index.searchsorted(_snapped(index, start, "left"), side="left")),
        int(index.searchsorted(_snapped(index, end, "right"), side="right")),
    )


def range_slice(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Rows of ``frame`` from ``start`` to ``end``, both bounds inclusive.

    The obvious ``frame.loc[(index >= start) & (index <= end)]`` builds two
    full-length boolean arrays and then gathers the rows one by one. On a
    sorted index the same rows are a contiguous block, so two binary searches
    and a slice give the identical frame for a fraction of the work - and a
    warm fetch takes this slice once per symbol.
    """
    bounds = _range_bounds(frame.index, start, end)
    if bounds is None:
        return frame.loc[(frame.index >= start) & (frame.index <= end)]
    first, last = bounds
    return frame.iloc[first:last]


def frame_has_range(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1d",
) -> bool:
    del interval  # Range tolerance is identical for daily and intraday frames.
    if frame is None or frame.empty:
        return False
    bounds = _range_bounds(frame.index, start, end)
    if bounds is None:
        in_range = frame.loc[(frame.index >= start) & (frame.index <= end)]
        if in_range.empty:
            return False
        first_bar = in_range.index.min()
        last_bar = in_range.index.max()
    else:
        first, last = bounds
        if last <= first:
            return False
        # Sorted, so the first and last rows of the block are its min and max.
        first_bar = frame.index[first]
        last_bar = frame.index[last - 1]
    return bool(
        first_bar <= start + pd.Timedelta(days=3)
        and last_bar >= end - pd.Timedelta(days=3)
    )


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
    "apply_splits_only_adjustment",
    "empty_ohlcv_frame",
    "frame_has_range",
    "inclusive_fetch_bounds",
    "merge_price_frames",
    "naive_normalized_index",
    "normalize_price_frame",
    "range_slice",
    "split_yfinance_download",
    "warn_unadjustable_fmp_frames",
]
