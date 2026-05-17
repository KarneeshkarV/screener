"""India daily relative-strength breakout scanner.

The scan is intentionally local/OHLCV-based because the required filters
depend on stock-vs-index history, SuperTrend state, previous completed weekly
high, and NSE delivery bhavcopy data.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, field_validator
from rich.console import Console
from rich.table import Table

from screener.backtester.data import PriceFetcher, tv_to_yf
from screener.unusual_volume.delivery import load_delivery_panel


logger = logging.getLogger(__name__)
DEFAULT_BENCHMARK = "^NSEI"
DEFAULT_BENCHMARKS = {"india": "^NSEI", "us": "SPY"}
RS_WINDOW = 55
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
VOLUME_WINDOW = 20
VOLUME_MULTIPLIER = 1.5


class RsBreakoutRow(BaseModel):
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

    model_config = ConfigDict(frozen=True)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("symbol must not be empty")
        return normalized

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class RsBreakoutResult(BaseModel):
    as_of: date
    benchmark: str
    full: list[RsBreakoutRow]
    relaxed: list[RsBreakoutRow]

    model_config = ConfigDict(frozen=True)

    @field_validator("benchmark")
    @classmethod
    def _normalize_benchmark(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("benchmark must not be empty")
        return normalized


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


def relative_strength_55(
    stock_close: pd.Series, benchmark_close: pd.Series
) -> pd.Series:
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
            st.iloc[i] = (
                final_lower.iloc[i]
                if close.iloc[i] >= hl2.iloc[i]
                else final_upper.iloc[i]
            )
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
            st.iloc[i] = (
                final_lower.iloc[i]
                if close.iloc[i] > final_upper.iloc[i]
                else final_upper.iloc[i]
            )
        else:
            st.iloc[i] = (
                final_upper.iloc[i]
                if close.iloc[i] < final_lower.iloc[i]
                else final_lower.iloc[i]
            )
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


def delivery_lookup(
    panel: pd.DataFrame,
) -> dict[str, tuple[Optional[float], Optional[float]]]:
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
    vol_avg = (
        df["volume"].rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean().shift(1)
    )
    prev_week_high = previous_completed_week_high(df, df.index[-1].date())

    last_idx = df.index[-1]
    if (
        last_idx not in rs.index
        or pd.isna(rs.loc[last_idx])
        or pd.isna(st.loc[last_idx])
    ):
        return None
    avg20 = (
        float(vol_avg.loc[last_idx])
        if not pd.isna(vol_avg.loc[last_idx])
        else float("nan")
    )
    if not math.isfinite(avg20) or avg20 <= 0:
        return None

    close = float(df.loc[last_idx, "close"])
    volume = float(df.loc[last_idx, "volume"])
    rs_55 = float(rs.loc[last_idx])
    supertrend_value = float(st.loc[last_idx])
    volume_ratio = volume / avg20
    delivery_pct, previous_delivery_pct = delivery or (None, None)

    base_pass = (
        rs_55 > 0 and close > supertrend_value and volume_ratio >= VOLUME_MULTIPLIER
    )
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
    require_delivery: bool = True,
) -> RsBreakoutResult:
    benchmark = normalize_bars(benchmark_bars, as_of)
    if benchmark.empty:
        raise ValueError("Benchmark OHLCV data is empty.")
    lookup = delivery_lookup(
        delivery_panel if delivery_panel is not None else pd.DataFrame()
    )
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
        if price_pass and (delivery_pass or not require_delivery):
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
    max_workers: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start = as_of - timedelta(days=history_days)
    end = as_of + timedelta(days=1)
    ticker_list = list(tickers)
    yf_map = {t: tv_to_yf(t, market) for t in ticker_list}
    benchmark_bars = fetcher.fetch([benchmark], start, end).get(
        benchmark, pd.DataFrame()
    )
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    def _fetch_one(tv_sym: str, yf_sym: str) -> tuple[str, pd.DataFrame]:
        try:
            data = fetcher.fetch([yf_sym], start, end)
        except (
            requests.RequestException,
            ConnectionError,
            TimeoutError,
            KeyError,
            ValueError,
        ):
            return tv_sym, pd.DataFrame()
        return tv_sym, data.get(yf_sym, pd.DataFrame())

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [
            pool.submit(_fetch_one, tv_sym, yf_sym) for tv_sym, yf_sym in yf_map.items()
        ]
        for fut in as_completed(futures):
            tv_sym, frame = fut.result()
            bars_by_symbol[tv_sym] = frame
    return bars_by_symbol, benchmark_bars


def load_india_delivery_for_scan(symbols: Iterable[str], as_of: date) -> pd.DataFrame:
    return load_delivery_panel(
        [india_symbol(s) for s in symbols], as_of, history_days=14
    )


def india_symbol(symbol: str) -> str:
    if ":" in symbol:
        return symbol.split(":", 1)[1].upper()
    return symbol.replace(".NS", "").replace(".BO", "").upper()


def sort_rows(rows: Iterable[RsBreakoutRow]) -> list[RsBreakoutRow]:
    return sorted(rows, key=lambda r: (r.volume_ratio, r.rs_55), reverse=True)


def required_history_bars() -> int:
    return max(RS_WINDOW + 1, VOLUME_WINDOW + 1, SUPERTREND_PERIOD + 1)


def previous_completed_week_high_series(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    week_key = bars.index.to_period("W-FRI")
    weekly_high = bars["high"].astype(float).groupby(week_key).max()
    prev_week_high = week_key.map(weekly_high.shift(1))
    return pd.Series(
        prev_week_high, index=bars.index, dtype=float, name="previous_week_high"
    )


def _delivery_series_for_symbol(
    panel: Optional[pd.DataFrame],
    symbol: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    cols = (
        "delivery_pct",
        "previous_delivery_pct",
        "delivery_pct_last",
        "delivery_trend",
        "delivery_spike",
    )
    empty = pd.DataFrame({c: pd.Series(np.nan, index=index, dtype=float) for c in cols})
    if panel is None or panel.empty:
        return empty
    sym = india_symbol(symbol)
    rows = panel[panel["SYMBOL"].astype(str).str.upper() == sym].copy()
    if rows.empty:
        return empty
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = (
        rows.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    delivery_pct = pd.to_numeric(rows["DELIV_PER"], errors="coerce")
    sma20 = delivery_pct.rolling(20, min_periods=5).mean()
    std20 = delivery_pct.rolling(20, min_periods=5).std(ddof=0)
    trend = delivery_pct / sma20.replace(0.0, np.nan)
    spike = (delivery_pct - sma20) / std20.replace(0.0, np.nan)
    series = pd.DataFrame(
        {
            "delivery_pct": delivery_pct.to_numpy(dtype=float),
            "previous_delivery_pct": delivery_pct.shift(1).to_numpy(dtype=float),
            "delivery_pct_last": delivery_pct.to_numpy(dtype=float),
            "delivery_trend": trend.to_numpy(dtype=float),
            "delivery_spike": spike.to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(rows["date"]),
    )
    return series.reindex(index)


def build_signal_frame(
    bars: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    delivery_panel: Optional[pd.DataFrame] = None,
    symbol: str = "",
    require_delivery: bool = False,
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy().sort_index()
    rs = relative_strength_55(df["close"], benchmark_close)
    st = supertrend(df)
    avg_volume = (
        df["volume"]
        .astype(float)
        .rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW)
        .mean()
        .shift(1)
    )
    prev_week_high = previous_completed_week_high_series(df)
    delivery = _delivery_series_for_symbol(delivery_panel, symbol, df.index)
    out = df.copy()
    out["rs_55"] = rs.reindex(df.index)
    out["supertrend_value"] = st.reindex(df.index)
    out["avg_volume_20d"] = avg_volume
    out["volume_ratio"] = df["volume"].astype(float) / avg_volume
    out["previous_week_high"] = prev_week_high
    out["delivery_pct"] = delivery["delivery_pct"]
    out["previous_delivery_pct"] = delivery["previous_delivery_pct"]
    out["delivery_pct_last"] = delivery["delivery_pct_last"]
    out["delivery_trend"] = delivery["delivery_trend"]
    out["delivery_spike"] = delivery["delivery_spike"]
    base_pass = (
        (out["rs_55"] > 0)
        & (out["close"].astype(float) > out["supertrend_value"])
        & (out["volume_ratio"] >= VOLUME_MULTIPLIER)
    )
    price_pass = out["previous_week_high"].notna() & (
        out["close"].astype(float) > out["previous_week_high"]
    )
    delivery_pass = (
        out["delivery_pct"].notna()
        & out["previous_delivery_pct"].notna()
        & (out["delivery_pct"] > out["previous_delivery_pct"])
    )
    out["rs_breakout_entry"] = (
        base_pass & price_pass & (delivery_pass if require_delivery else True)
    ).astype(float)
    return out


def prepare_backtest_frames(
    bars_by_symbol: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    *,
    market: str,
    delivery_panel: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    benchmark = benchmark_bars.copy()
    if benchmark is None or benchmark.empty:
        return {symbol: bars.copy() for symbol, bars in bars_by_symbol.items()}
    benchmark = benchmark.sort_index()
    benchmark_close = benchmark["close"].astype(float)
    require_delivery = market == "india"
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, bars in bars_by_symbol.items():
        prepared[symbol] = build_signal_frame(
            bars,
            benchmark_close,
            delivery_panel=delivery_panel,
            symbol=symbol,
            require_delivery=require_delivery,
        )
    if market == "india":
        _join_microstructure_panels(prepared)
    return prepared


def _join_microstructure_panels(prepared: dict[str, pd.DataFrame]) -> None:
    """Left-join accumulated option-chain / FII-DII snapshot panels as feature
    columns. These live-only sources have no historical backfill, so columns
    are NaN for dates before the daily snapshot accumulation began —
    strategies referencing them simply don't trigger on those bars. Read-only,
    keeps backtests offline/deterministic.
    """
    from screener.cache import panel_path, read_frame
    from screener.unusual_volume.fii_dii import fii_dii_metric_series

    oc = read_frame(panel_path("option_chain"))
    fd = read_frame(panel_path("fii_dii"))
    oc_by_sym: dict[str, pd.DataFrame] = {}
    if oc is not None and not oc.empty:
        oc = oc.copy()
        oc["as_of"] = pd.to_datetime(oc["as_of"], errors="coerce").dt.normalize()
        for sym, grp in oc.groupby(oc["SYMBOL"].astype(str).str.upper()):
            oc_by_sym[sym] = grp.set_index("as_of").sort_index()
    if fd is not None and not fd.empty:
        fd = fd.copy()
        fd = fii_dii_metric_series(fd)
    for symbol, frame in prepared.items():
        if frame is None or frame.empty:
            continue
        sym = india_symbol(symbol)
        target_index = pd.DatetimeIndex(
            pd.to_datetime(pd.Index(frame.index), errors="coerce")
        )
        if target_index.tz is not None:
            target_index = target_index.tz_localize(None)
        target_index = target_index.normalize()
        g = oc_by_sym.get(sym)
        # One-bar lag: the FII/DII provisional figure (and the option-chain
        # snapshot) is only published after market close, so a same-day
        # intraday/open decision must not see today's value. Shift the
        # reindexed series by one trading bar so each bar only sees the prior
        # day's accumulated snapshot. Cold-start bars stay NaN (shift fills
        # the leading bar with NaN, matching the missing-history contract).
        for col in ("call_put_oi_ratio", "pcr"):
            if g is not None and col in g.columns:
                joined = g[col].reindex(target_index).shift(1)
                frame[col] = pd.Series(joined.to_numpy(dtype=float), index=frame.index)
                if g[col].notna().any() and frame[col].notna().sum() == 0:
                    logger.debug(
                        "option-chain panel for %s joined zero non-NaN %s rows",
                        sym,
                        col,
                    )
            else:
                frame[col] = np.nan
        for col in ("fii_5d_net", "dii_5d_net", "fii_trend"):
            if fd is not None and not fd.empty and col in fd.columns:
                joined = fd[col].reindex(target_index).shift(1)
                frame[col] = pd.Series(joined.to_numpy(dtype=float), index=frame.index)
                if fd[col].notna().any() and frame[col].notna().sum() == 0:
                    logger.debug(
                        "FII/DII panel for %s joined zero non-NaN %s rows",
                        sym,
                        col,
                    )
            else:
                frame[col] = np.nan


def render_result(
    result: RsBreakoutResult,
    console: Console,
    limit: int = 50,
    market: str = "india",
) -> None:
    console.print(
        f"[bold]{market.upper()} RS Breakout Screen[/bold] [dim]as of {result.as_of} "
        f"vs {result.benchmark}[/dim]"
    )
    _render_bucket("Full", result.full[:limit], console)
    _render_bucket(
        "Relaxed (without price breakout and delivery increase)",
        result.relaxed[:limit],
        console,
    )


def _render_bucket(title: str, rows: list[RsBreakoutRow], console: Console) -> None:
    table = Table(
        title=f"{title} - {len(rows)} match(es)", show_header=True, header_style="bold"
    )
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
    payload = result.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_markdown(result: RsBreakoutResult, path: Path, market: str = "india") -> None:
    lines = [
        f"# {market.upper()} RS Breakout Screen ({result.as_of})",
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
