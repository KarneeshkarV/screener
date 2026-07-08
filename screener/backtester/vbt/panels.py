"""OHLCV panel builders for vectorbt sweeps."""

from __future__ import annotations

import pandas as pd

from screener.backtester.data import _naive_normalized_index


def _build_column_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for sym in yf_symbols:
        frame = price_panel.get(sym)
        if frame is None or frame.empty or column not in frame.columns:
            continue
        col = frame[column].astype(float)
        col.index = _naive_normalized_index(col.index)
        trimmed = col.loc[(col.index >= start) & (col.index <= end)]
        if trimmed.empty:
            continue
        series[sym] = trimmed
    if not series:
        raise ValueError(f"No usable {column} prices for the requested window.")
    panel = pd.DataFrame(series).sort_index()
    panel = panel.ffill()
    panel = panel.dropna(axis=1, how="any")
    return panel.dropna(how="all")


def build_close_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="close", start=start, end=end
    )


def build_open_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="open", start=start, end=end
    )


def build_high_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="high", start=start, end=end
    )


def build_low_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="low", start=start, end=end
    )


def build_volume_panel(
    price_panel: dict[str, pd.DataFrame],
    yf_symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return _build_column_panel(
        price_panel, yf_symbols, column="volume", start=start, end=end
    )
