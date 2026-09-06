"""India daily relative-strength breakout scanner.

The scan is intentionally local/OHLCV-based because the required filters
depend on stock-vs-index history, SuperTrend state, previous completed weekly
high, and NSE delivery bhavcopy data.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, field_validator
from rich.console import Console, JustifyMethod
from rich.table import Table

from screener.backtester.data import PriceFetcher, tv_to_yf
from screener.format import fmt_float as _fmt_float
from screener.indicators.frames import wilder_atr
from screener.markets import get_market
from screener.parallel import parallel_map
from screener.relative_strength import RS_RATIO_WINDOW, relative_strength_ratio
from screener.reporting import dump_json_file, markdown_row
from screener.symbols import normalize_symbol, tv_to_nse
from screener.unusual_volume.delivery import load_delivery_panel

logger = logging.getLogger(__name__)
RS_WINDOW = RS_RATIO_WINDOW
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
    previous_week_high: float | None
    volume: float
    avg_volume_20d: float
    volume_ratio: float
    delivery_pct: float | None
    previous_delivery_pct: float | None

    model_config = ConfigDict(frozen=True)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

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


def supertrend(
    bars: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> pd.Series:
    """Compute SuperTrend with Wilder/RMA ATR.

    The band recursion reads its own previous value, so it cannot be
    vectorised along time. What it can avoid is paying pandas scalar-access
    overhead for every read and write: the loop touches roughly a dozen
    positions per bar, and ``Series.iloc`` costs about two orders of magnitude
    more per touch than a float64 array does. Over a 503-name S&P 500 panel
    that was 5.9 million ``__getitem__`` calls and 97% of a two-year
    ``backtest-rolling --strategy rs_breakout``. Running the identical
    recurrence over numpy arrays and building one Series at the end took the
    whole command from 47.2s to 13.0s with a byte-identical trade ledger.

    ``wilder_atr`` still sees the pandas Series it always did, so the ATR the
    recursion rests on is unchanged. Every branch below is a transcription of
    the pandas form, NaN comparisons included: ``nan == nan`` and
    ``nan < x`` are False either way, so the seeding and carry-forward
    decisions land on the same bars.

    Do not swap this for ``screener.indicators.plugins.supertrend``. That one
    answers a different question (direction, not the band), seeds without
    emitting a value, and compares the close against the *previous* bands
    where this compares against the current ones. Its ``_supertrend_dir_panel``
    is not the panel form of this function for the same three reasons, so
    routing ``prepare_backtest_frames`` through it would change the indicator,
    not just its cost.

    What is left on the table is the loop *count*: ``prepare_backtest_frames``
    calls this once per symbol, so a 503-name panel still runs 503 Python
    loops. Collapsing those needs a panel form of *this* recurrence, over a
    ``(bars, symbols)`` block with a per-row validity mask, which is a new
    implementation rather than a reuse and has to be proved
    trade-for-trade before it can land.
    """
    if bars.empty:
        return pd.Series(dtype=float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close_s = bars["close"].astype(float)
    atr = wilder_atr(high, low, close_s, period, min_periods=period).to_numpy(
        dtype=float
    )
    hl2 = (high.to_numpy(dtype=float) + low.to_numpy(dtype=float)) / 2.0
    close = close_s.to_numpy(dtype=float)
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(bars)
    final_upper = np.full(n, np.nan, dtype=float)
    final_lower = np.full(n, np.nan, dtype=float)
    st_values = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            st_values[i] = final_lower[i] if close[i] >= hl2[i] else final_upper[i]
            continue

        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        if st_values[i - 1] == final_upper[i - 1]:
            st_values[i] = (
                final_lower[i] if close[i] > final_upper[i] else final_upper[i]
            )
        else:
            st_values[i] = (
                final_upper[i] if close[i] < final_lower[i] else final_lower[i]
            )
    st = pd.Series(st_values, index=bars.index, dtype=float)
    st.name = "supertrend"
    return st


def previous_completed_week_high(bars: pd.DataFrame, as_of: date) -> float | None:
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
) -> dict[str, tuple[float | None, float | None]]:
    """Return symbol -> (latest DELIV_PER, previous DELIV_PER)."""
    if panel is None or panel.empty:
        return {}
    out: dict[str, tuple[float | None, float | None]] = {}
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


def rs_breakout_signals(frame: pd.DataFrame, *, require_delivery: bool) -> pd.DataFrame:
    """Evaluate the RS-breakout entry rule over a prepared signal frame.

    This is the one place the rule is written. Both consumers go through it:
    :func:`build_signal_frame` evaluates it across the whole history for the
    backtest plugin, and :func:`evaluate_symbol` evaluates it over the one-bar
    frame it assembles for the live scan, so the two paths cannot drift apart.

    ``frame`` must carry ``close``, ``rs_55``, ``supertrend_value``,
    ``volume_ratio``, ``previous_week_high``, ``delivery_pct`` and
    ``previous_delivery_pct``. The returned frame carries the three component
    verdicts plus the composed entry flag.
    """
    close = frame["close"].astype(float)
    base_pass = (
        (frame["rs_55"] > 0)
        & (close > frame["supertrend_value"])
        & (frame["volume_ratio"] >= VOLUME_MULTIPLIER)
    )
    price_pass = frame["previous_week_high"].notna() & (
        close > frame["previous_week_high"]
    )
    delivery_pass = (
        frame["delivery_pct"].notna()
        & frame["previous_delivery_pct"].notna()
        & (frame["delivery_pct"] > frame["previous_delivery_pct"])
    )
    return pd.DataFrame(
        {
            "base_pass": base_pass,
            "price_pass": price_pass,
            "delivery_pass": delivery_pass,
            "rs_breakout_entry": base_pass
            & price_pass
            & (delivery_pass if require_delivery else True),
        },
        index=frame.index,
    )


def _one_bar_signal_frame(
    index: pd.DatetimeIndex, **values: float | None
) -> pd.DataFrame:
    """Assemble a one-row frame in the shape :func:`rs_breakout_signals` reads."""
    return pd.DataFrame(
        {
            name: pd.Series(
                [float("nan") if value is None else float(value)],
                index=index,
                dtype=float,
            )
            for name, value in values.items()
        },
        index=index,
    )


def evaluate_symbol(
    symbol: str,
    bars: pd.DataFrame,
    benchmark_close: pd.Series,
    as_of: date,
    delivery: tuple[float | None, float | None] | None = None,
) -> tuple[RsBreakoutRow, bool, bool] | None:
    """Return row plus price/delivery pass booleans when base filters pass."""
    df = normalize_bars(bars, as_of)
    if len(df) < max(RS_WINDOW + 1, VOLUME_WINDOW + 1, SUPERTREND_PERIOD + 1):
        return None

    rs = relative_strength_ratio(df["close"], benchmark_close)
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

    # The live scan's verdict is the vectorized rule applied to one bar, so it
    # cannot disagree with the backtest plugin's. ``require_delivery`` only
    # composes the entry flag, which the caller derives itself from the three
    # component verdicts below.
    verdict = rs_breakout_signals(
        _one_bar_signal_frame(
            cast(pd.DatetimeIndex, df.index[-1:]),
            close=close,
            rs_55=rs_55,
            supertrend_value=supertrend_value,
            volume_ratio=volume_ratio,
            previous_week_high=prev_week_high,
            delivery_pct=delivery_pct,
            previous_delivery_pct=previous_delivery_pct,
        ),
        require_delivery=False,
    ).iloc[-1]
    if not bool(verdict["base_pass"]):
        return None

    price_pass = bool(verdict["price_pass"])
    delivery_pass = bool(verdict["delivery_pass"])
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
    delivery_panel: pd.DataFrame | None = None,
    benchmark_symbol: str | None = None,
    require_delivery: bool = True,
) -> RsBreakoutResult:
    resolved_benchmark = benchmark_symbol or get_market("india").benchmark
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
        benchmark=resolved_benchmark,
        full=sort_rows(full),
        relaxed=sort_rows(relaxed),
    )


def fetch_price_data(
    tickers: Iterable[str],
    market: str,
    as_of: date,
    fetcher: PriceFetcher,
    benchmark: str | None = None,
    history_days: int = 220,
    max_workers: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    resolved_benchmark = benchmark or get_market("india").benchmark
    start = as_of - timedelta(days=history_days)
    end = as_of + timedelta(days=1)
    ticker_list = list(tickers)
    yf_map = {t: tv_to_yf(t, market) for t in ticker_list}
    benchmark_bars = fetcher.fetch([resolved_benchmark], start, end).get(
        resolved_benchmark, pd.DataFrame()
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

    for tv_sym, frame in parallel_map(
        lambda item: _fetch_one(item[0], item[1]),
        yf_map.items(),
        max_workers=max(1, int(max_workers)),
    ):
        bars_by_symbol[tv_sym] = frame
    return bars_by_symbol, benchmark_bars


def load_india_delivery_for_scan(symbols: Iterable[str], as_of: date) -> pd.DataFrame:
    return load_delivery_panel(
        [india_symbol(s) for s in symbols], as_of, history_days=14
    )


def india_symbol(symbol: str) -> str:
    return tv_to_nse(symbol, strip_suffix=True)


def sort_rows(rows: Iterable[RsBreakoutRow]) -> list[RsBreakoutRow]:
    return sorted(rows, key=lambda r: (r.volume_ratio, r.rs_55), reverse=True)


def required_history_bars() -> int:
    return max(RS_WINDOW + 1, VOLUME_WINDOW + 1, SUPERTREND_PERIOD + 1)


def previous_completed_week_high_series(bars: pd.DataFrame) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    week_key = cast(pd.DatetimeIndex, bars.index).to_period("W-FRI")
    weekly_high = bars["high"].astype(float).groupby(week_key).max()
    # Index.map accepts a Series as a label->value mapping at runtime; the
    # stub only lists Mapping/Callable, so cast the Series argument.
    prev_week_high = week_key.map(cast("Mapping[Any, Any]", weekly_high.shift(1)))
    return pd.Series(
        prev_week_high, index=bars.index, dtype=float, name="previous_week_high"
    )


def _delivery_series_for_symbol(
    panel: pd.DataFrame | None,
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
    delivery_panel: pd.DataFrame | None = None,
    symbol: str = "",
    require_delivery: bool = False,
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy().sort_index()
    rs = relative_strength_ratio(df["close"], benchmark_close)
    st = supertrend(df)
    avg_volume = (
        df["volume"]
        .astype(float)
        .rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW)
        .mean()
        .shift(1)
    )
    prev_week_high = previous_completed_week_high_series(df)
    delivery = _delivery_series_for_symbol(
        delivery_panel, symbol, cast(pd.DatetimeIndex, df.index)
    )
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
    signals = rs_breakout_signals(out, require_delivery=require_delivery)
    out["rs_breakout_entry"] = signals["rs_breakout_entry"].astype(float)
    return out


def prepare_backtest_frames(
    bars_by_symbol: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    *,
    market: str,
    delivery_panel: pd.DataFrame | None = None,
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
    columns: list[tuple[str, JustifyMethod]] = [
        ("Ticker", "left"),
        ("Close", "right"),
        ("RS55", "right"),
        ("ST", "right"),
        ("PrevWkHigh", "right"),
        ("VolRatio", "right"),
        ("Deliv%", "right"),
        ("PrevDeliv%", "right"),
    ]
    for name, justify in columns:
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
    dump_json_file(payload, path)


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
                markdown_row(
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
            )
        lines.append("")
    path.write_text("\n".join(lines))
