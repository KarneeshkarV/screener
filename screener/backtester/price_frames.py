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


#: How far a cached bar's close may differ from the vendor's freshly served
#: close for the same session before the cache is judged to be on a different
#: price scale. The two are the same number when nothing happened, so the
#: tolerance only has to absorb float round-trips through parquet and the
#: vendor's own rounding, not a real price move.
RESCALE_TOLERANCE = 0.01

#: Overlapping sessions needed before a scale verdict is trusted. One shared
#: bar can differ for reasons that are not a rescale (a late vendor
#: correction to a single session), so a verdict rests on a handful.
RESCALE_MIN_OVERLAP = 2


#: Single-session price ratio that marks a *possible* stitched split seam in
#: a stored frame. A real session rarely quadruples or quarters, and the
#: splits that corrupt a cache are far larger than that (1:10 upward), so this
#: is a cheap, generous trigger for re-verification - never a verdict. The
#: verdict is :func:`overlap_rescaled`, which asks the vendor.
SEAM_RATIO = 4.0


def has_price_seam(frame: pd.DataFrame | None) -> bool:
    """True when ``frame`` contains a session-over-session jump of >= :data:`SEAM_RATIO`.

    This exists for entries written *before* split detection was added. Their
    seam is already stitched into the stored bars, and a tail refresh cannot
    reveal it: the fresh week agrees with the cached week, because both sit on
    the post-split scale. Only the older bars are on the abandoned one.

    So the seam is looked for in the stored series itself. A jump this large
    is usually a split, sometimes a real move on a microcap, and this function
    does not try to tell them apart - it only marks the entry for a full
    re-download, after which the bars are the vendor's own and whatever jump
    remains is real.
    """
    if frame is None or frame.empty or "close" not in frame.columns:
        return False
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    close = close[close > 0]
    if len(close) < 2:
        return False
    ratio = close / close.shift()
    ratio = ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if ratio.empty:
        return False
    return bool((ratio >= SEAM_RATIO).any() or (ratio <= 1.0 / SEAM_RATIO).any())


def overlap_rescaled(
    existing: pd.DataFrame | None, new: pd.DataFrame, interval: str = "1d"
) -> bool:
    """True when ``existing`` and ``new`` price the same sessions differently.

    This is the split-detection test. With ``auto_adjust=True`` the vendor
    back-propagates every split into the *whole* history it serves, so after a
    split its series sits on a new scale while a cache entry written before it
    still holds the old one. :func:`merge_price_frames` keeps the last row per
    date, which stitches the two scales together into one series with a fake
    jump at the seam - and that jump is read downstream as a real return.
    A 1:125 reverse split turns a -96% loser into a +3164% momentum winner.

    Comparing the closes of the sessions *both* frames carry is what separates
    a rescale from ordinary new bars: those sessions are the same trading days,
    so the same vendor under the same adjustment must report the same number
    for them. A ratio that is not ~1.0 means the vendor has re-based its
    history and the cached bars are stale in value, not merely in range.

    The test is a ratio, not a difference, because that is the shape a split
    takes: every overlapping bar moves by one common factor. It is deliberately
    symmetric about 1.0 so a forward split (factor < 1) is caught exactly as a
    reverse split (factor > 1) is.

    Returns False when either frame is missing or the overlap is too small to
    judge; the caller then keeps its ordinary merge.
    """
    if existing is None or existing.empty or new is None or new.empty:
        return False
    if "close" not in existing.columns or "close" not in new.columns:
        return False
    old_close = pd.to_numeric(existing["close"], errors="coerce").astype(float)
    new_close = pd.to_numeric(new["close"], errors="coerce").astype(float)
    old_close.index = naive_normalized_index(old_close.index, interval)
    new_close.index = naive_normalized_index(new_close.index, interval)
    old_close = old_close[~old_close.index.duplicated(keep="last")]
    new_close = new_close[~new_close.index.duplicated(keep="last")]
    shared = old_close.index.intersection(new_close.index)
    if len(shared) < RESCALE_MIN_OVERLAP:
        return False
    old_shared = old_close.reindex(shared)
    new_shared = new_close.reindex(shared)
    usable = old_shared.notna() & new_shared.notna() & (old_shared > 0)
    if int(usable.sum()) < RESCALE_MIN_OVERLAP:
        return False
    ratio = (new_shared[usable] / old_shared[usable]).abs()
    return bool((ratio - 1.0).abs().max() > RESCALE_TOLERANCE)


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
    "has_price_seam",
    "overlap_rescaled",
    "inclusive_fetch_bounds",
    "merge_price_frames",
    "naive_normalized_index",
    "normalize_price_frame",
    "range_slice",
    "split_yfinance_download",
    "warn_unadjustable_fmp_frames",
]
