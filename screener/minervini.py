"""Mark Minervini Trend Template screening helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher, build_price_fetcher
from screener.cache import parse_ttl
from screener.commands.rs_breakout import load_universe
from screener.symbols import tv_to_yf


MINERVINI_ENTRY_EXPR = (
    "close > sma(close, 150) and "
    "close > sma(close, 200) and "
    "sma(close, 150) > sma(close, 200) and "
    "sma200_up_1m > 0 and "
    "sma(close, 50) > sma(close, 150) and "
    "sma(close, 50) > sma(close, 200) and "
    "close > sma(close, 50) and "
    "close >= lowest(low, 252) * 1.3 and "
    "close >= highest(high, 252) * 0.75 and "
    "rs_rank >= 70"
)

MINERVINI_EXIT_EXPR = "crossunder(close, sma(close, 50))"


@dataclass(frozen=True)
class MinerviniRow:
    symbol: str
    close: float
    sma50: float
    sma150: float
    sma200: float
    pct_above_low_52w: float
    pct_below_high_52w: float
    rs_rank: float
    as_of: date


def required_history_bars() -> int:
    """Bars needed for 52-week range, 200-day SMA, and 1-month SMA trend."""
    return 253


def add_rs_rank_column(
    bars_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add 12-month relative-strength percentile, 0-100, across the universe."""
    returns: dict[str, pd.Series] = {}
    for symbol, bars in bars_by_symbol.items():
        if bars is None or bars.empty or "close" not in bars:
            continue
        close = bars["close"].astype(float)
        returns[symbol] = close / close.shift(252) - 1.0
    if not returns:
        return bars_by_symbol

    rs_frame = pd.DataFrame(returns)
    ranks = rs_frame.rank(axis=1, pct=True) * 100.0
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in bars_by_symbol.items():
        frame = bars.copy()
        if "close" in frame:
            sma200 = frame["close"].astype(float).rolling(200, min_periods=200).mean()
            frame["sma200_up_1m"] = (sma200 > sma200.shift(21)).astype(float)
        if symbol in ranks:
            frame["rs_rank"] = ranks[symbol].reindex(frame.index)
        else:
            frame["rs_rank"] = pd.NA
        out[symbol] = frame
    return out


def prepare_backtest_frames(
    bars_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return add_rs_rank_column(bars_by_symbol)


def evaluate_symbol(
    symbol: str, bars: pd.DataFrame, as_of: date
) -> MinerviniRow | None:
    if bars is None or bars.empty:
        return None
    history = bars.loc[bars.index <= pd.Timestamp(as_of)].copy()
    if len(history) < required_history_bars():
        return None

    close = history["close"].astype(float)
    high = history["high"].astype(float)
    low = history["low"].astype(float)
    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=150).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    low_52w = low.rolling(252, min_periods=252).min()
    high_52w = high.rolling(252, min_periods=252).max()

    latest = history.index[-1]
    values = {
        "close": close.loc[latest],
        "sma50": sma50.loc[latest],
        "sma150": sma150.loc[latest],
        "sma200": sma200.loc[latest],
        "sma200_prev_month": sma200.shift(21).loc[latest],
        "low_52w": low_52w.loc[latest],
        "high_52w": high_52w.loc[latest],
        "rs_rank": history["rs_rank"].loc[latest]
        if "rs_rank" in history
        else float("nan"),
    }
    if any(pd.isna(value) for value in values.values()):
        return None

    price = float(values["close"])
    year_low = float(values["low_52w"])
    year_high = float(values["high_52w"])
    checks = [
        price > float(values["sma150"]),
        price > float(values["sma200"]),
        float(values["sma150"]) > float(values["sma200"]),
        float(values["sma200"]) > float(values["sma200_prev_month"]),
        float(values["sma50"]) > float(values["sma150"]),
        float(values["sma50"]) > float(values["sma200"]),
        price > float(values["sma50"]),
        price >= year_low * 1.3,
        price >= year_high * 0.75,
        float(values["rs_rank"]) >= 70.0,
    ]
    if not all(checks):
        return None

    return MinerviniRow(
        symbol=symbol,
        close=price,
        sma50=float(values["sma50"]),
        sma150=float(values["sma150"]),
        sma200=float(values["sma200"]),
        pct_above_low_52w=(price / year_low - 1.0) * 100.0,
        pct_below_high_52w=(price / year_high - 1.0) * 100.0,
        rs_rank=float(values["rs_rank"]),
        as_of=latest.date(),
    )


def scan_minervini(
    market: str,
    *,
    as_of: date,
    limit: int,
    cache_ttl: str,
    refresh: bool,
    fetcher: PriceFetcher | None = None,
) -> list[MinerviniRow]:
    universe = load_universe(
        market,
        universe_limit=max(int(limit) * 20, 500),
        cache_ttl=parse_ttl(cache_ttl, default=900),
        refresh=refresh,
    )
    yf_by_tv = {symbol: tv_to_yf(symbol, market) for symbol in universe}
    fetcher = fetcher or build_price_fetcher(refresh=refresh)
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=430)).date()
    panel = fetcher.fetch(yf_by_tv.values(), start, as_of)
    bars_by_symbol = {
        symbol: panel.get(yf_symbol, pd.DataFrame())
        for symbol, yf_symbol in yf_by_tv.items()
    }
    prepared = add_rs_rank_column(bars_by_symbol)
    rows = [
        row
        for symbol, bars in prepared.items()
        if (row := evaluate_symbol(symbol, bars, as_of)) is not None
    ]
    return sorted(rows, key=lambda row: row.rs_rank, reverse=True)[: int(limit)]


def render_rows(rows: list[MinerviniRow], console: Console, market: str) -> None:
    table = Table(title=f"{market.upper()} Mark Minervini Screen")
    table.add_column("Symbol")
    table.add_column("Close", justify="right")
    table.add_column("SMA50", justify="right")
    table.add_column("SMA150", justify="right")
    table.add_column("SMA200", justify="right")
    table.add_column("% Above 52w Low", justify="right")
    table.add_column("% From 52w High", justify="right")
    table.add_column("RS Rank", justify="right")
    table.add_column("As Of")
    for row in rows:
        table.add_row(
            row.symbol,
            f"{row.close:.2f}",
            f"{row.sma50:.2f}",
            f"{row.sma150:.2f}",
            f"{row.sma200:.2f}",
            f"{row.pct_above_low_52w:.1f}%",
            f"{row.pct_below_high_52w:.1f}%",
            f"{row.rs_rank:.0f}",
            row.as_of.isoformat(),
        )
    console.print(table)
