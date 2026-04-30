"""India daily relative-strength breakout scanner.

The scan is intentionally local/OHLCV-based because the required filters
depend on stock-vs-index history, SuperTrend state, previous completed weekly
high, and NSE delivery bhavcopy data.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher, tv_to_yf
from screener.unusual_volume.delivery import load_delivery_panel


DEFAULT_BENCHMARK = "^NSEI"
RS_WINDOW = 55
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
VOLUME_WINDOW = 20
VOLUME_MULTIPLIER = 1.5


@dataclass(frozen=True)
class RsBreakoutRow:
    symbol: str
    date: date
    close: float
    rs_55: float
    supertrend: float
    previous_week_high: Optional[float]
    volume: float
    avg_volume_20d: float
    volume_ratio: float
    delivery_pct: Optional[float]
    previous_delivery_pct: Optional[float]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["date"] = self.date.isoformat()
        return data


@dataclass(frozen=True)
class RsBreakoutResult:
    as_of: date
    benchmark: str
    full: list[RsBreakoutRow]
    relaxed: list[RsBreakoutRow]


def normalize_bars(bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return sorted OHLCV bars up to as_of with a DatetimeIndex."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            return pd.DataFrame()
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]).values))
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    df = df[df.index <= pd.Timestamp(as_of).normalize()]
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df[list(needed)].astype(float)


def relative_strength_55(stock_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [stock_close.astype(float), benchmark_close.astype(float)],
        axis=1,
        join="inner",
    ).dropna()
    aligned.columns = ["stock", "benchmark"]
    stock_ret = aligned["stock"] / aligned["stock"].shift(RS_WINDOW)
    bench_ret = aligned["benchmark"] / aligned["benchmark"].shift(RS_WINDOW)
    rs = ((stock_ret / bench_ret) - 1.0) * 100.0
    rs.name = "rs_55"
    return rs


def supertrend(
    bars: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> pd.Series:
    """Compute SuperTrend with Wilder/RMA ATR."""
    if bars.empty:
        return pd.Series(dtype=float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(np.nan, index=bars.index, dtype=float)
    final_lower = pd.Series(np.nan, index=bars.index, dtype=float)
    st = pd.Series(np.nan, index=bars.index, dtype=float)

    for i in range(len(bars)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            st.iloc[i] = final_lower.iloc[i] if close.iloc[i] >= hl2.iloc[i] else final_upper.iloc[i]
            continue

        final_upper.iloc[i] = (
            basic_upper.iloc[i]
            if basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            basic_lower.iloc[i]
            if basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
            else final_lower.iloc[i - 1]
        )

        prev_st = st.iloc[i - 1]
        if prev_st == final_upper.iloc[i - 1]:
            st.iloc[i] = final_lower.iloc[i] if close.iloc[i] > final_upper.iloc[i] else final_upper.iloc[i]
        else:
            st.iloc[i] = final_upper.iloc[i] if close.iloc[i] < final_lower.iloc[i] else final_lower.iloc[i]
    st.name = "supertrend"
    return st


def previous_completed_week_high(bars: pd.DataFrame, as_of: date) -> Optional[float]:
    """High of the last fully completed Monday-Friday week before as_of."""
    if bars.empty:
        return None
    as_ts = pd.Timestamp(as_of).normalize()
    this_monday = as_ts - pd.Timedelta(days=as_ts.weekday())
    prev_monday = this_monday - pd.Timedelta(days=7)
    prev_friday = this_monday - pd.Timedelta(days=3)
    week = bars[(bars.index >= prev_monday) & (bars.index <= prev_friday)]
    if week.empty:
        return None
    return float(week["high"].max())


def delivery_lookup(panel: pd.DataFrame) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """Return symbol -> (latest DELIV_PER, previous DELIV_PER)."""
    if panel is None or panel.empty:
        return {}
    out: dict[str, tuple[Optional[float], Optional[float]]] = {}
    df = panel.copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper()
    df = df.sort_values(["SYMBOL", "date"])
    for sym, group in df.groupby("SYMBOL"):
        pct = pd.to_numeric(group["DELIV_PER"], errors="coerce").dropna()
        if pct.empty:
            continue
        latest = float(pct.iloc[-1])
        prev = float(pct.iloc[-2]) if len(pct) >= 2 else None
        out[sym] = (latest, prev)
    return out


def evaluate_symbol(
    symbol: str,
    bars: pd.DataFrame,
    benchmark_close: pd.Series,
    as_of: date,
    delivery: tuple[Optional[float], Optional[float]] | None = None,
) -> Optional[tuple[RsBreakoutRow, bool, bool]]:
    """Return row plus price/delivery pass booleans when base filters pass."""
    df = normalize_bars(bars, as_of)
    if len(df) < max(RS_WINDOW + 1, VOLUME_WINDOW + 1, SUPERTREND_PERIOD + 1):
        return None

    rs = relative_strength_55(df["close"], benchmark_close)
    st = supertrend(df)
    vol_avg = df["volume"].rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean().shift(1)
    prev_week_high = previous_completed_week_high(df, df.index[-1].date())

    last_idx = df.index[-1]
    if last_idx not in rs.index or pd.isna(rs.loc[last_idx]) or pd.isna(st.loc[last_idx]):
        return None
    avg20 = float(vol_avg.loc[last_idx]) if not pd.isna(vol_avg.loc[last_idx]) else float("nan")
    if not math.isfinite(avg20) or avg20 <= 0:
        return None

    close = float(df.loc[last_idx, "close"])
    volume = float(df.loc[last_idx, "volume"])
    rs_55 = float(rs.loc[last_idx])
    supertrend_value = float(st.loc[last_idx])
    volume_ratio = volume / avg20
    delivery_pct, previous_delivery_pct = delivery or (None, None)

    base_pass = rs_55 > 0 and close > supertrend_value and volume_ratio >= VOLUME_MULTIPLIER
    if not base_pass:
        return None

    price_pass = prev_week_high is not None and close > prev_week_high
    delivery_pass = (
        delivery_pct is not None
        and previous_delivery_pct is not None
        and delivery_pct > previous_delivery_pct
    )
    row = RsBreakoutRow(
        symbol=symbol,
        date=last_idx.date(),
        close=close,
        rs_55=round(rs_55, 4),
        supertrend=round(supertrend_value, 4),
        previous_week_high=None if prev_week_high is None else round(prev_week_high, 4),
        volume=volume,
        avg_volume_20d=round(avg20, 4),
        volume_ratio=round(volume_ratio, 4),
        delivery_pct=None if delivery_pct is None else round(delivery_pct, 4),
        previous_delivery_pct=None
        if previous_delivery_pct is None
        else round(previous_delivery_pct, 4),
    )
    return row, price_pass, delivery_pass


def scan_rs_breakouts(
    bars_by_symbol: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    as_of: date,
    delivery_panel: Optional[pd.DataFrame] = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK,
) -> RsBreakoutResult:
    benchmark = normalize_bars(benchmark_bars, as_of)
    if benchmark.empty:
        raise ValueError("Benchmark OHLCV data is empty.")
    lookup = delivery_lookup(delivery_panel if delivery_panel is not None else pd.DataFrame())
    full: list[RsBreakoutRow] = []
    relaxed: list[RsBreakoutRow] = []
    for symbol, bars in bars_by_symbol.items():
        bare = india_symbol(symbol)
        evaluated = evaluate_symbol(
            bare,
            bars,
            benchmark["close"],
            as_of,
            delivery=lookup.get(bare),
        )
        if evaluated is None:
            continue
        row, price_pass, delivery_pass = evaluated
        relaxed.append(row)
        if price_pass and delivery_pass:
            full.append(row)
    return RsBreakoutResult(
        as_of=as_of,
        benchmark=benchmark_symbol,
        full=sort_rows(full),
        relaxed=sort_rows(relaxed),
    )


def fetch_price_data(
    tickers: Iterable[str],
    market: str,
    as_of: date,
    fetcher: PriceFetcher,
    benchmark: str = DEFAULT_BENCHMARK,
    history_days: int = 220,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start = as_of - timedelta(days=history_days)
    end = as_of + timedelta(days=1)
    ticker_list = list(tickers)
    yf_map = {t: tv_to_yf(t, market) for t in ticker_list}
    requested = list(yf_map.values()) + [benchmark]
    fetched = fetcher.fetch(requested, start, end)
    bars_by_symbol = {
        tv_sym: fetched.get(yf_sym, pd.DataFrame()) for tv_sym, yf_sym in yf_map.items()
    }
    benchmark_bars = fetched.get(benchmark, pd.DataFrame())
    return bars_by_symbol, benchmark_bars


def load_india_delivery_for_scan(symbols: Iterable[str], as_of: date) -> pd.DataFrame:
    return load_delivery_panel([india_symbol(s) for s in symbols], as_of, history_days=14)


def india_symbol(symbol: str) -> str:
    if ":" in symbol:
        return symbol.split(":", 1)[1].upper()
    return symbol.replace(".NS", "").replace(".BO", "").upper()


def sort_rows(rows: Iterable[RsBreakoutRow]) -> list[RsBreakoutRow]:
    return sorted(rows, key=lambda r: (r.volume_ratio, r.rs_55), reverse=True)


def render_result(result: RsBreakoutResult, console: Console, limit: int = 50) -> None:
    console.print(
        f"[bold]India RS Breakout Screen[/bold] [dim]as of {result.as_of} "
        f"vs {result.benchmark}[/dim]"
    )
    _render_bucket("Full", result.full[:limit], console)
    _render_bucket("Relaxed (without price breakout and delivery increase)", result.relaxed[:limit], console)


def _render_bucket(title: str, rows: list[RsBreakoutRow], console: Console) -> None:
    table = Table(title=f"{title} - {len(rows)} match(es)", show_header=True, header_style="bold")
    for name, justify in [
        ("Ticker", "left"),
        ("Close", "right"),
        ("RS55", "right"),
        ("ST", "right"),
        ("PrevWkHigh", "right"),
        ("VolRatio", "right"),
        ("Deliv%", "right"),
        ("PrevDeliv%", "right"),
    ]:
        table.add_column(name, justify=justify)
    for row in rows:
        table.add_row(
            row.symbol,
            _fmt_float(row.close),
            _fmt_float(row.rs_55),
            _fmt_float(row.supertrend),
            _fmt_float(row.previous_week_high),
            _fmt_float(row.volume_ratio),
            _fmt_float(row.delivery_pct),
            _fmt_float(row.previous_delivery_pct),
        )
    console.print(table)


def write_json(result: RsBreakoutResult, path: Path) -> None:
    payload = {
        "as_of": result.as_of.isoformat(),
        "benchmark": result.benchmark,
        "full": [r.to_dict() for r in result.full],
        "relaxed": [r.to_dict() for r in result.relaxed],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_markdown(result: RsBreakoutResult, path: Path) -> None:
    lines = [
        f"# India RS Breakout Screen ({result.as_of})",
        "",
        f"**Benchmark:** {result.benchmark}",
        "",
    ]
    for title, rows in [
        ("Full", result.full),
        ("Relaxed (without price breakout and delivery increase)", result.relaxed),
    ]:
        lines.extend(
            [
                f"## {title} ({len(rows)})",
                "",
                "| # | Ticker | Close | RS55 | SuperTrend | Prev Week High | Vol Ratio | Deliv% | Prev Deliv% |",
                "|---|--------|------:|-----:|-----------:|---------------:|----------:|-------:|------------:|",
            ]
        )
        for i, row in enumerate(rows, 1):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        f"**{row.symbol}**",
                        _fmt_float(row.close),
                        _fmt_float(row.rs_55),
                        _fmt_float(row.supertrend),
                        _fmt_float(row.previous_week_high),
                        _fmt_float(row.volume_ratio),
                        _fmt_float(row.delivery_pct),
                        _fmt_float(row.previous_delivery_pct),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def _fmt_float(value: Optional[float], ndp: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{ndp}f}"
