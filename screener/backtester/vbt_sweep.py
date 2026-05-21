"""Fast vectorbt parameter sweeps for strategy exploration.

This module is for **fast exploration**, not validation. Results may diverge
from ``backtest-rolling`` because the following are **not** modeled here:

- slot allocation / position sizing rules
- partial exits
- dividends
- custom slippage models (``slippage=0.0`` hard-coded)
- commissions / fees (``fees=0.0`` hard-coded)
- trailing stops, stop-loss, take-profit (custom engine supports all three)

Entries/exits are shifted by one bar and filled at the next bar's **open** to
match the custom engine's MOO (market-on-open) semantics — so there is no
same-bar look-ahead, but residual differences vs MOC-style exits still apply.

Always validate promising parameter combinations with ``backtest-rolling``
before drawing conclusions.

Indicator menu
--------------
Solo indicators:
    sma, ema  — moving-average crossover (fast vs slow)
    breakout  — Donchian breakout on close
    bbands    — EMA-based Bollinger Band upper-band breakout
    supertrend — ATR-based trend direction flip (numba kernel)
    keltner   — Keltner channel upper-band breakout
    rsi       — Wilder RSI(14) crosses above a threshold
    macd      — MACD line crosses signal line (default 12/26/9)
    vol_breakout — Donchian breakout AND volume > vol-MA
    obv_trend — OBV crosses its own EMA

Combinations (entry = primary signal AND secondary filter):
    sma_rsi      — SMA cross + RSI(14) > 50 filter
    breakout_rsi — Donchian breakout + RSI(14) > 50 filter

The ``sma_cross`` strategy in :func:`sma_crossover_signals` is a hand-coded
vectorbt signal generator; it is **not** the same as the ``ma_cross`` plugin
in ``screener/strategies/plugins/ma_cross.py``. The two are intentionally
decoupled so this module can stay vectorbt-native for speed; do not assume
parameter results transfer between them.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, Literal, cast

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.backtester.cli_common import DEFAULT_BENCHMARK
from screener.backtester.data import PriceFetcher, build_price_fetcher, tv_to_yf
from screener.universes import load_current_universe

DISCLAIMER = (
    "[yellow]Exploration only — approximations; validate with backtest-rolling.\n"
    "Not modeled: slot allocation, partial exits, dividends, custom slippage "
    "(slippage=0), fees (fees=0), trailing stops, stop-loss, take-profit.\n"
    "Fills: next-bar open (MOO match); residual MOC differences may apply.[/yellow]"
)

MetricName = Literal["sharpe", "total_return", "calmar"]
StrategyName = Literal["sma_cross"]
IndicatorName = Literal[
    "sma",
    "ema",
    "breakout",
    "bbands",
    "supertrend",
    "keltner",
    "rsi",
    "macd",
    "vol_breakout",
    "obv_trend",
    "sma_rsi",
    "breakout_rsi",
]
VALID_INDICATORS: tuple[str, ...] = (
    "sma",
    "ema",
    "breakout",
    "bbands",
    "supertrend",
    "keltner",
    "rsi",
    "macd",
    "vol_breakout",
    "obv_trend",
    "sma_rsi",
    "breakout_rsi",
)

# Fixed-config defaults (kept as module constants so the viewer can document them).
DEFAULT_BREAKOUT_WINDOWS: tuple[int, ...] = (20, 55, 100)
DEFAULT_BBANDS_WINDOWS: tuple[int, ...] = (20, 50)
DEFAULT_BBANDS_STD: float = 2.0
DEFAULT_SUPERTREND_PERIODS: tuple[int, ...] = (7, 10, 14)
DEFAULT_SUPERTREND_MULT: float = 3.0
DEFAULT_KELTNER_WINDOWS: tuple[int, ...] = (20, 50)
DEFAULT_KELTNER_MULT: float = 2.0
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_RSI_THRESHOLDS: tuple[int, ...] = (50, 55, 60)
DEFAULT_VOL_MA_WINDOW: int = 20
DEFAULT_VOL_MULTIPLIER: float = 1.0
DEFAULT_MACD_FAST: int = 12
DEFAULT_MACD_SLOW: int = 26
DEFAULT_MACD_SIGNAL: int = 9
DEFAULT_OBV_EMA_WINDOWS: tuple[int, ...] = (20, 50)

# Indicators that require a separate volume panel for signal generation.
VOLUME_INDICATORS: frozenset[str] = frozenset({"vol_breakout", "obv_trend"})
# Indicators that require high/low panels.
HL_INDICATORS: frozenset[str] = frozenset({"supertrend", "keltner"})

StrategyBuilder = Callable[
    [pd.DataFrame, int, int, int, Any],
    tuple[pd.DataFrame, pd.DataFrame],
]

INITIAL_CAPITAL_DEFAULT = 100_000.0


def _require_vectorbt() -> Any:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise click.ClickException(
            "vectorbt is not installed. Install optional deps with: "
            "uv sync --extra vectorbt"
        ) from exc
    return vbt


def parse_int_list(raw: str, *, name: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError(f"--{name} requires at least one integer.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.UsageError(f"--{name} expects comma-separated integers.") from exc


def parse_indicator_list(raw: str) -> list[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.UsageError("--indicator requires at least one value.")
    if parts == ["all"]:
        return list(VALID_INDICATORS)
    bad = [p for p in parts if p not in VALID_INDICATORS]
    if bad:
        raise click.UsageError(
            f"--indicator values must be in {VALID_INDICATORS} (or 'all'); got {bad!r}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _sma(close: pd.DataFrame, window: int, vbt: Any) -> pd.DataFrame:
    ma_out = vbt.MA.run(close, window=window).ma
    if isinstance(ma_out.columns, pd.MultiIndex):
        return ma_out.xs(window, axis=1, level="ma_window")
    return ma_out


def _fixed_hold_exits_np(arr: np.ndarray, hold: int) -> np.ndarray:
    """Numpy-only equivalent of :func:`_fixed_hold_exits` for use inside the
    vectorized signal builder. ``arr`` is the entries mask as a bool ndarray of
    shape ``(n_days, n_tickers)``. Returns an exits mask of the same shape.
    """
    out = np.zeros_like(arr, dtype=bool)
    if hold <= 0 or not arr.any():
        return out
    entry_idx = np.argwhere(arr)
    exit_rows = entry_idx[:, 0] + hold
    valid = exit_rows < arr.shape[0]
    if valid.any():
        out[exit_rows[valid], entry_idx[valid, 1]] = True
    return out


def _fixed_hold_exits(entries: pd.DataFrame, hold: int) -> pd.DataFrame:
    arr = entries.to_numpy(dtype=bool)
    out = _fixed_hold_exits_np(arr, hold)
    return pd.DataFrame(out, index=entries.index, columns=entries.columns)


def _ma_for_window(ma_panel: pd.DataFrame, window: int) -> pd.DataFrame:
    if isinstance(ma_panel.columns, pd.MultiIndex):
        return ma_panel.xs(window, axis=1, level="ma_window")
    return ma_panel


def sma_crossover_signals(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    hold: int,
    vbt: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SMA crossover: enter when close crosses above slow SMA while close > fast SMA."""
    sma_fast = _sma(close, fast, vbt)
    sma_slow = _sma(close, slow, vbt)
    entries = close.vbt.crossed_above(sma_slow) & (close > sma_fast)
    cross_exits = close.vbt.crossed_below(sma_slow)
    if hold > 0:
        exits = cross_exits | _fixed_hold_exits(entries, hold)
    else:
        exits = cross_exits
    return entries.fillna(False), exits.fillna(False)


STRATEGY_BUILDERS: dict[StrategyName, StrategyBuilder] = {
    "sma_cross": sma_crossover_signals,
}


def iter_param_combos(
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
) -> list[tuple[int, int, int]]:
    combos: list[tuple[int, int, int]] = []
    for fast, slow, hold in itertools.product(fast_values, slow_values, hold_values):
        if slow <= fast:
            continue
        combos.append((fast, slow, hold))
    return combos


def iter_indicator_combos(
    indicators: list[str],
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
    *,
    breakout_windows: list[int] | None = None,
    bbands_windows: list[int] | None = None,
    supertrend_periods: list[int] | None = None,
    keltner_windows: list[int] | None = None,
    rsi_thresholds: list[int] | None = None,
    obv_ema_windows: list[int] | None = None,
) -> list[tuple[str, int, int, int]]:
    """Build (indicator, fast_or_window, slow, hold) combos for each indicator.

    Conventions for the (fast, slow) pair per indicator:
    - sma / ema: (fast_window, slow_window). slow > fast required.
    - breakout / bbands / supertrend / keltner / vol_breakout / obv_trend /
      breakout_rsi: (window_or_period, 0). slow is the sentinel ``0``.
    - rsi: (threshold, 0). period is fixed at ``DEFAULT_RSI_PERIOD``.
    - macd: (macd_fast, macd_slow). signal fixed at ``DEFAULT_MACD_SIGNAL``.
    - sma_rsi: (sma_fast, sma_slow). RSI filter uses defaults.
    """
    breakout_windows = list(breakout_windows or DEFAULT_BREAKOUT_WINDOWS)
    bbands_windows = list(bbands_windows or DEFAULT_BBANDS_WINDOWS)
    supertrend_periods = list(supertrend_periods or DEFAULT_SUPERTREND_PERIODS)
    keltner_windows = list(keltner_windows or DEFAULT_KELTNER_WINDOWS)
    rsi_thresholds = list(rsi_thresholds or DEFAULT_RSI_THRESHOLDS)
    obv_ema_windows = list(obv_ema_windows or DEFAULT_OBV_EMA_WINDOWS)

    def _pair_grid(name: str) -> list[tuple[str, int, int, int]]:
        out: list[tuple[str, int, int, int]] = []
        for fast, slow, hold in itertools.product(
            fast_values, slow_values, hold_values
        ):
            if slow <= fast:
                continue
            out.append((name, int(fast), int(slow), int(hold)))
        return out

    def _window_grid(name: str, windows: list[int]) -> list[tuple[str, int, int, int]]:
        return [
            (name, int(w), 0, int(hold))
            for w, hold in itertools.product(windows, hold_values)
        ]

    combos: list[tuple[str, int, int, int]] = []
    for ind in indicators:
        if ind in ("sma", "ema", "sma_rsi"):
            combos.extend(_pair_grid(ind))
        elif ind == "macd":
            combos.extend(
                ("macd", DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, int(hold))
                for hold in hold_values
            )
        elif ind == "breakout":
            combos.extend(_window_grid("breakout", breakout_windows))
        elif ind == "bbands":
            combos.extend(_window_grid("bbands", bbands_windows))
        elif ind == "supertrend":
            combos.extend(_window_grid("supertrend", supertrend_periods))
        elif ind == "keltner":
            combos.extend(_window_grid("keltner", keltner_windows))
        elif ind == "vol_breakout":
            combos.extend(_window_grid("vol_breakout", breakout_windows))
        elif ind == "breakout_rsi":
            combos.extend(_window_grid("breakout_rsi", breakout_windows))
        elif ind == "obv_trend":
            combos.extend(_window_grid("obv_trend", obv_ema_windows))
        elif ind == "rsi":
            combos.extend(
                ("rsi", int(thresh), 0, int(hold))
                for thresh, hold in itertools.product(rsi_thresholds, hold_values)
            )
        else:
            raise ValueError(f"Unknown indicator: {ind!r}")
    return combos


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
        col.index = pd.to_datetime(col.index).tz_localize(None).normalize()
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


def _scalar_metric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (pd.Series, pd.DataFrame)):
        flat = value.to_numpy().ravel()
        if flat.size == 0:
            return float("nan")
        return float(flat[0])
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.ravel()[0])


def run_combo_backtest(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    hold: int,
    *,
    vbt: Any,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
) -> dict[str, float | int]:
    entries, exits = STRATEGY_BUILDERS["sma_cross"](close, fast, slow, hold, vbt)
    # Match custom engine MOO semantics: signals on bar t fill at bar t+1 open.
    # Shift entries/exits by 1 bar and price the fill at the next bar's open.
    entries_shifted = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_shifted = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    fill_price = open_ if open_ is not None else close
    pf = vbt.Portfolio.from_signals(
        close,
        entries_shifted,
        exits_shifted,
        price=fill_price,
        init_cash=float(initial_capital),
        fees=0.0,
        slippage=0.0,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    )
    win_rate = _scalar_metric(pf.trades.win_rate())
    trade_count = pf.trades.count()
    if isinstance(trade_count, (pd.Series, pd.DataFrame)):
        trade_count = int(trade_count.to_numpy().sum())
    else:
        trade_count = int(trade_count)
    return {
        "fast": fast,
        "slow": slow,
        "hold": hold,
        "sharpe": _scalar_metric(pf.sharpe_ratio()),
        "total_return": _scalar_metric(pf.total_return()),
        "calmar": _scalar_metric(pf.calmar_ratio()),
        "max_drawdown": _scalar_metric(pf.max_drawdown()),
        "win_rate": win_rate,
        "trades": trade_count,
    }


# ---------------------------------------------------------------------------
# Indicator helpers (per-column vectorized intermediates)
# ---------------------------------------------------------------------------


def _rsi_wilder(close: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder's RSI on a 2D close panel. Returns ndarray of shape ``close``."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.to_numpy(dtype=float)


def _atr_wilder(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int
) -> np.ndarray:
    """Wilder ATR on 2D panels (close-shifted true range, EWM smoothed)."""
    prev_close = close.shift(1)
    tr1 = (high - low).to_numpy(dtype=float)
    tr2 = (high - prev_close).abs().to_numpy(dtype=float)
    tr3 = (low - prev_close).abs().to_numpy(dtype=float)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_df = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr_df.ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(dtype=float)


def _obv(close: pd.DataFrame, volume: pd.DataFrame) -> np.ndarray:
    """On-Balance Volume cumulative sum."""
    diff = close.diff().to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)
    sign = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    flow = sign * vol
    flow[~np.isfinite(flow)] = 0.0
    return np.cumsum(flow, axis=0)


def _supertrend_signals_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column Supertrend direction state machine.

    Pure numpy implementation (no numba dep) — runs once per (period, multiplier)
    pair across the entire panel. Inner loop is O(n_days * n_tickers) which is
    fine at this scale.
    """
    n, m = close.shape
    entries = np.zeros((n, m), dtype=bool)
    exits = np.zeros((n, m), dtype=bool)

    # ATR (Wilder) computed inline to avoid pandas roundtrips.
    tr = np.zeros((n, m), dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = np.abs(high[i] - close[i - 1])
        c = np.abs(low[i] - close[i - 1])
        tr[i] = np.maximum(a, np.maximum(b, c))
    atr = np.zeros((n, m), dtype=float)
    if period <= n:
        atr[period - 1] = tr[:period].mean(axis=0)
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    final_upper = np.zeros((n, m), dtype=float)
    final_lower = np.zeros((n, m), dtype=float)
    direction = np.zeros((n, m), dtype=np.int8)
    if period < n:
        final_upper[period] = upper[period]
        final_lower[period] = lower[period]
        direction[period] = np.where(close[period] > final_upper[period], 1, -1)
        for i in range(period + 1, n):
            keep_upper = (upper[i] >= final_upper[i - 1]) & (
                close[i - 1] <= final_upper[i - 1]
            )
            final_upper[i] = np.where(keep_upper, final_upper[i - 1], upper[i])
            keep_lower = (lower[i] <= final_lower[i - 1]) & (
                close[i - 1] >= final_lower[i - 1]
            )
            final_lower[i] = np.where(keep_lower, final_lower[i - 1], lower[i])
            prev_dir = direction[i - 1]
            new_dir = np.where(
                prev_dir == 1,
                np.where(close[i] < final_lower[i], -1, 1),
                np.where(close[i] > final_upper[i], 1, -1),
            ).astype(np.int8)
            direction[i] = new_dir
            entries[i] = (prev_dir == -1) & (new_dir == 1)
            exits[i] = (prev_dir == 1) & (new_dir == -1)
    return entries, exits


# ---------------------------------------------------------------------------
# Vectorized signal panel builder
# ---------------------------------------------------------------------------


def _build_indicator_signal_panels(
    close: pd.DataFrame,
    combos: list[tuple[str, int, int, int]],
    *,
    vbt: Any,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute entry/exit masks for combos across all supported indicators.

    All indicators feed into a single (entries, exits) panel keyed by
    ``(indicator, fast, slow, hold, ticker)``; they will be handed to a
    single ``Portfolio.from_signals`` call so the import/setup cost amortises.

    The SMA path is byte-identical to the historical implementation:
    ``vbt.MA.run`` + the ``crossed_above_nb`` Numba kernel. This keeps the
    frozen-fixture regression at ``atol=1e-9``.
    """
    from vectorbt.generic.nb import crossed_above_nb

    tickers = list(close.columns)
    close_arr = np.ascontiguousarray(close.to_numpy(dtype=float))
    n_days, n_tickers = close_arr.shape

    needs_high_low = any(c[0] in HL_INDICATORS for c in combos)
    needs_volume = any(c[0] in VOLUME_INDICATORS for c in combos)
    if needs_high_low and (high is None or low is None):
        raise ValueError("supertrend / keltner indicators require high & low panels.")
    if needs_volume and volume is None:
        raise ValueError("vol_breakout / obv_trend indicators require a volume panel.")

    # ---- SMA intermediates (preserves frozen regression bit-for-bit) ----
    sma_fast_set = sorted({c[1] for c in combos if c[0] in ("sma", "sma_rsi")})
    sma_slow_set = sorted({c[2] for c in combos if c[0] in ("sma", "sma_rsi")})
    sma_above_fast_by_w: dict[int, np.ndarray] = {}
    sma_crossed_above_by_w: dict[int, np.ndarray] = {}
    sma_crossed_below_by_w: dict[int, np.ndarray] = {}
    if sma_fast_set:
        sma_fast_panel = vbt.MA.run(close, window=sma_fast_set).ma
        for w in sma_fast_set:
            fast_arr = _ma_for_window(sma_fast_panel, w).to_numpy(dtype=float)
            sma_above_fast_by_w[w] = close_arr > fast_arr
    if sma_slow_set:
        sma_slow_panel = vbt.MA.run(close, window=sma_slow_set).ma
        for w in sma_slow_set:
            slow_arr = np.ascontiguousarray(
                _ma_for_window(sma_slow_panel, w).to_numpy(dtype=float)
            )
            sma_crossed_above_by_w[w] = crossed_above_nb(close_arr, slow_arr)
            sma_crossed_below_by_w[w] = crossed_above_nb(slow_arr, close_arr)

    # ---- EMA intermediates ----
    ema_fast_set = sorted({c[1] for c in combos if c[0] == "ema"})
    ema_slow_set = sorted({c[2] for c in combos if c[0] == "ema"})
    ema_all_windows = sorted(set(ema_fast_set) | set(ema_slow_set))
    ema_by_w: dict[int, np.ndarray] = {}
    for w in ema_all_windows:
        ema_by_w[w] = np.ascontiguousarray(
            close.ewm(span=w, adjust=False).mean().to_numpy(dtype=float)
        )
    ema_above_fast_by_w = {w: close_arr > ema_by_w[w] for w in ema_fast_set}
    ema_crossed_above_by_w = {
        w: crossed_above_nb(close_arr, ema_by_w[w]) for w in ema_slow_set
    }
    ema_crossed_below_by_w = {
        w: crossed_above_nb(ema_by_w[w], close_arr) for w in ema_slow_set
    }

    # ---- Breakout / vol_breakout / breakout_rsi (Donchian on close) ----
    breakout_w_set = sorted(
        {c[1] for c in combos if c[0] in ("breakout", "vol_breakout", "breakout_rsi")}
    )
    breakout_entries_by_w: dict[int, np.ndarray] = {}
    for w in breakout_w_set:
        rmax = close.rolling(w).max().shift(1).to_numpy(dtype=float)
        breakout_entries_by_w[w] = crossed_above_nb(
            close_arr, np.ascontiguousarray(rmax)
        )

    # ---- BBands (EMA middle + n_std * rolling std) ----
    bbands_w_set = sorted({c[1] for c in combos if c[0] == "bbands"})
    bbands_entries_by_w: dict[int, np.ndarray] = {}
    bbands_exits_by_w: dict[int, np.ndarray] = {}
    for w in bbands_w_set:
        middle = close.ewm(span=w, adjust=False).mean()
        std = close.rolling(w).std()
        upper = (middle + DEFAULT_BBANDS_STD * std).to_numpy(dtype=float)
        middle_arr = middle.to_numpy(dtype=float)
        bbands_entries_by_w[w] = crossed_above_nb(
            close_arr, np.ascontiguousarray(upper)
        )
        # Exit: close crosses back below middle.
        bbands_exits_by_w[w] = crossed_above_nb(
            np.ascontiguousarray(middle_arr), close_arr
        )

    # ---- Supertrend ----
    supertrend_p_set = sorted({c[1] for c in combos if c[0] == "supertrend"})
    supertrend_signals_by_p: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if supertrend_p_set:
        assert high is not None and low is not None
        high_arr = high.to_numpy(dtype=float)
        low_arr = low.to_numpy(dtype=float)
        for p in supertrend_p_set:
            supertrend_signals_by_p[p] = _supertrend_signals_np(
                close_arr, high_arr, low_arr, p, DEFAULT_SUPERTREND_MULT
            )

    # ---- Keltner channels (EMA middle ± mult * ATR) ----
    keltner_w_set = sorted({c[1] for c in combos if c[0] == "keltner"})
    keltner_entries_by_w: dict[int, np.ndarray] = {}
    keltner_exits_by_w: dict[int, np.ndarray] = {}
    if keltner_w_set:
        assert high is not None and low is not None
        for w in keltner_w_set:
            middle = close.ewm(span=w, adjust=False).mean()
            atr = _atr_wilder(high, low, close, w)
            upper = middle.to_numpy(dtype=float) + DEFAULT_KELTNER_MULT * atr
            lower_band = middle.to_numpy(dtype=float) - DEFAULT_KELTNER_MULT * atr
            keltner_entries_by_w[w] = crossed_above_nb(
                close_arr, np.ascontiguousarray(upper)
            )
            keltner_exits_by_w[w] = crossed_above_nb(
                np.ascontiguousarray(lower_band), close_arr
            )

    # ---- RSI (Wilder), shared across rsi indicator + *_rsi combos ----
    rsi_needed = any(c[0] in ("rsi", "sma_rsi", "breakout_rsi") for c in combos)
    rsi_arr: np.ndarray | None = (
        _rsi_wilder(close, DEFAULT_RSI_PERIOD) if rsi_needed else None
    )
    rsi_threshold_signals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rsi_filter_above_50: np.ndarray | None = None
    if rsi_arr is not None:
        rsi_filter_above_50 = (rsi_arr > 50).astype(bool)
        for thresh in sorted({c[1] for c in combos if c[0] == "rsi"}):
            thresh_arr = np.full_like(rsi_arr, float(thresh))
            entries_t = crossed_above_nb(
                np.ascontiguousarray(rsi_arr), np.ascontiguousarray(thresh_arr)
            )
            exits_t = crossed_above_nb(
                np.ascontiguousarray(thresh_arr), np.ascontiguousarray(rsi_arr)
            )
            rsi_threshold_signals[thresh] = (entries_t, exits_t)

    # ---- MACD ----
    macd_needed = any(c[0] == "macd" for c in combos)
    macd_entries: np.ndarray | None = None
    macd_exits: np.ndarray | None = None
    if macd_needed:
        ema_f = close.ewm(span=DEFAULT_MACD_FAST, adjust=False).mean()
        ema_s = close.ewm(span=DEFAULT_MACD_SLOW, adjust=False).mean()
        macd_line = (ema_f - ema_s).to_numpy(dtype=float)
        signal_line = (
            pd.DataFrame(macd_line, index=close.index, columns=close.columns)
            .ewm(span=DEFAULT_MACD_SIGNAL, adjust=False)
            .mean()
            .to_numpy(dtype=float)
        )
        macd_entries = crossed_above_nb(
            np.ascontiguousarray(macd_line), np.ascontiguousarray(signal_line)
        )
        macd_exits = crossed_above_nb(
            np.ascontiguousarray(signal_line), np.ascontiguousarray(macd_line)
        )

    # ---- Volume MA for vol_breakout ----
    vol_above_ma: np.ndarray | None = None
    if any(c[0] == "vol_breakout" for c in combos):
        assert volume is not None
        vol_arr = volume.to_numpy(dtype=float)
        vol_ma = volume.rolling(DEFAULT_VOL_MA_WINDOW).mean().to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            vol_above_ma = vol_arr > (DEFAULT_VOL_MULTIPLIER * vol_ma)
        vol_above_ma[~np.isfinite(vol_ma)] = False

    # ---- OBV trend (OBV crosses its own EMA) ----
    obv_w_set = sorted({c[1] for c in combos if c[0] == "obv_trend"})
    obv_entries_by_w: dict[int, np.ndarray] = {}
    obv_exits_by_w: dict[int, np.ndarray] = {}
    if obv_w_set:
        assert volume is not None
        obv = _obv(close, volume)
        for w in obv_w_set:
            obv_df = pd.DataFrame(obv, index=close.index, columns=close.columns)
            obv_ema = obv_df.ewm(span=w, adjust=False).mean().to_numpy(dtype=float)
            obv_entries_by_w[w] = crossed_above_nb(
                np.ascontiguousarray(obv), np.ascontiguousarray(obv_ema)
            )
            obv_exits_by_w[w] = crossed_above_nb(
                np.ascontiguousarray(obv_ema), np.ascontiguousarray(obv)
            )

    # ---- Assemble per-combo entry/exit panels ----
    n_combos = len(combos)
    entries_panel = np.empty((n_days, n_combos * n_tickers), dtype=bool)
    exits_panel = np.empty((n_days, n_combos * n_tickers), dtype=bool)

    for k, (ind, fast_or_w, slow, hold) in enumerate(combos):
        if ind == "sma":
            entries_k = sma_crossed_above_by_w[slow] & sma_above_fast_by_w[fast_or_w]
            exits_k = sma_crossed_below_by_w[slow]
        elif ind == "ema":
            entries_k = ema_crossed_above_by_w[slow] & ema_above_fast_by_w[fast_or_w]
            exits_k = ema_crossed_below_by_w[slow]
        elif ind == "breakout":
            entries_k = breakout_entries_by_w[fast_or_w]
            exits_k = np.zeros_like(entries_k, dtype=bool)
        elif ind == "bbands":
            entries_k = bbands_entries_by_w[fast_or_w]
            exits_k = bbands_exits_by_w[fast_or_w]
        elif ind == "supertrend":
            entries_k, exits_k = supertrend_signals_by_p[fast_or_w]
        elif ind == "keltner":
            entries_k = keltner_entries_by_w[fast_or_w]
            exits_k = keltner_exits_by_w[fast_or_w]
        elif ind == "rsi":
            entries_k, exits_k = rsi_threshold_signals[fast_or_w]
        elif ind == "macd":
            assert macd_entries is not None and macd_exits is not None
            entries_k = macd_entries
            exits_k = macd_exits
        elif ind == "vol_breakout":
            assert vol_above_ma is not None
            entries_k = breakout_entries_by_w[fast_or_w] & vol_above_ma
            exits_k = np.zeros_like(entries_k, dtype=bool)
        elif ind == "obv_trend":
            entries_k = obv_entries_by_w[fast_or_w]
            exits_k = obv_exits_by_w[fast_or_w]
        elif ind == "sma_rsi":
            assert rsi_filter_above_50 is not None
            entries_k = (
                sma_crossed_above_by_w[slow]
                & sma_above_fast_by_w[fast_or_w]
                & rsi_filter_above_50
            )
            exits_k = sma_crossed_below_by_w[slow]
        elif ind == "breakout_rsi":
            assert rsi_filter_above_50 is not None
            entries_k = breakout_entries_by_w[fast_or_w] & rsi_filter_above_50
            exits_k = np.zeros_like(entries_k, dtype=bool)
        else:
            raise ValueError(f"Unknown indicator: {ind!r}")
        if hold > 0:
            exits_k = exits_k | _fixed_hold_exits_np(entries_k, hold)
        start = k * n_tickers
        stop = start + n_tickers
        entries_panel[:, start:stop] = entries_k
        exits_panel[:, start:stop] = exits_k

    col_index = pd.MultiIndex.from_tuples(
        [
            (ind, fast_or_w, slow, hold, t)
            for ind, fast_or_w, slow, hold in combos
            for t in tickers
        ],
        names=["indicator", "fast", "slow", "hold", "ticker"],
    )
    entries_df = pd.DataFrame(entries_panel, index=close.index, columns=col_index)
    exits_df = pd.DataFrame(exits_panel, index=close.index, columns=col_index)
    return entries_df, exits_df


def _combo_metric(
    series: pd.Series | float, ind: str, fast: int, slow: int, hold: int
) -> float:
    if isinstance(series, pd.Series):
        return _scalar_metric(series.loc[(ind, fast, slow, hold)])
    return _scalar_metric(series)


SOLO_INDICATORS_NAN_SLOW: frozenset[str] = frozenset(
    {
        "breakout",
        "bbands",
        "supertrend",
        "keltner",
        "rsi",
        "vol_breakout",
        "obv_trend",
        "breakout_rsi",
    }
)


def _portfolio_chunk_metrics(
    close: pd.DataFrame,
    fill_price: pd.DataFrame,
    entries_chunk: pd.DataFrame,
    exits_chunk: pd.DataFrame,
    *,
    vbt: Any,
    initial_capital: float,
) -> tuple[Any, ...]:
    """Run one ``Portfolio.from_signals`` call over a slice of combo columns.

    Materialising the tiled close/price for every combo at once peaks at
    ~``n_days * n_tickers * n_combos * 8`` bytes — which OOMs on SP500 ×
    multi-year × many-indicator runs. Chunking by combo keeps the peak bounded.
    """
    n_chunk_combos = entries_chunk.columns.get_level_values(0).nunique() * (
        entries_chunk.shape[1] // close.shape[1]
    )
    n_combos = entries_chunk.shape[1] // close.shape[1]
    tiled_close = np.tile(close.to_numpy(), (1, n_combos))
    close_broadcast = pd.DataFrame(
        tiled_close, index=close.index, columns=entries_chunk.columns, copy=False
    )
    if fill_price is close:
        price_broadcast = close_broadcast
    else:
        tiled_price = np.tile(fill_price.to_numpy(), (1, n_combos))
        price_broadcast = pd.DataFrame(
            tiled_price,
            index=fill_price.index,
            columns=entries_chunk.columns,
            copy=False,
        )
    del n_chunk_combos  # only used to make linter happy on unused branches
    pf = vbt.Portfolio.from_signals(
        close_broadcast,
        entries_chunk,
        exits_chunk,
        price=price_broadcast,
        init_cash=float(initial_capital),
        fees=0.0,
        slippage=0.0,
        group_by=["indicator", "fast", "slow", "hold"],
        cash_sharing=True,
        freq="1D",
    )
    return (
        pf.sharpe_ratio(),
        pf.total_return(),
        pf.calmar_ratio(),
        pf.max_drawdown(),
        pf.trades.win_rate(),
        pf.trades.count(),
    )


def run_parameter_sweep(
    close: pd.DataFrame,
    *,
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
    indicators: list[str] | None = None,
    breakout_windows: list[int] | None = None,
    bbands_windows: list[int] | None = None,
    supertrend_periods: list[int] | None = None,
    keltner_windows: list[int] | None = None,
    rsi_thresholds: list[int] | None = None,
    obv_ema_windows: list[int] | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    chunk_size: int | None = None,
) -> pd.DataFrame:
    vbt = _require_vectorbt()
    if indicators is None:
        indicators = ["sma"]
    combos = iter_indicator_combos(
        indicators,
        fast_values,
        slow_values,
        hold_values,
        breakout_windows=breakout_windows,
        bbands_windows=bbands_windows,
        supertrend_periods=supertrend_periods,
        keltner_windows=keltner_windows,
        rsi_thresholds=rsi_thresholds,
        obv_ema_windows=obv_ema_windows,
    )
    if not combos:
        raise ValueError("No valid parameter combinations to evaluate.")

    entries, exits = _build_indicator_signal_panels(
        close, combos, vbt=vbt, high=high, low=low, volume=volume
    )
    entries_shifted = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_shifted = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    fill_price = open_ if open_ is not None else close

    n_tickers = close.shape[1]
    n_combos = len(combos)
    # Bound peak memory: tiled close+price for one chunk costs
    # ``2 * n_days * n_tickers * chunk * 8`` bytes. Aim for <1 GiB per
    # chunk and keep at least one chunk even at very high ticker counts.
    if chunk_size is None:
        n_days = close.shape[0]
        bytes_per_combo = n_days * n_tickers * 8 * 2  # close + price tile
        target_bytes = 1 * 1024 * 1024 * 1024  # ~1 GiB
        chunk_size = max(1, min(n_combos, target_bytes // max(1, bytes_per_combo)))

    sharpe_parts: list[pd.Series] = []
    total_return_parts: list[pd.Series] = []
    calmar_parts: list[pd.Series] = []
    max_drawdown_parts: list[pd.Series] = []
    win_rate_parts: list[pd.Series] = []
    trade_count_parts: list[pd.Series] = []

    for start in range(0, n_combos, chunk_size):
        stop = min(start + chunk_size, n_combos)
        col_start = start * n_tickers
        col_stop = stop * n_tickers
        entries_chunk = entries_shifted.iloc[:, col_start:col_stop]
        exits_chunk = exits_shifted.iloc[:, col_start:col_stop]
        sharpe_c, total_c, calmar_c, dd_c, win_c, trades_c = _portfolio_chunk_metrics(
            close,
            fill_price,
            entries_chunk,
            exits_chunk,
            vbt=vbt,
            initial_capital=initial_capital,
        )
        sharpe_parts.append(sharpe_c)
        total_return_parts.append(total_c)
        calmar_parts.append(calmar_c)
        max_drawdown_parts.append(dd_c)
        win_rate_parts.append(win_c)
        trade_count_parts.append(trades_c)

    def _concat(parts: list[Any]) -> pd.Series | float:
        cleaned = [p for p in parts if isinstance(p, pd.Series) and not p.empty]
        if not cleaned:
            return parts[0] if parts else float("nan")
        return pd.concat(cleaned)

    sharpe = _concat(sharpe_parts)
    total_return = _concat(total_return_parts)
    calmar = _concat(calmar_parts)
    max_drawdown = _concat(max_drawdown_parts)
    win_rate = _concat(win_rate_parts)
    trade_count = _concat(trade_count_parts)

    rows: list[dict[str, Any]] = []
    for ind, fast_or_w, slow, hold in combos:
        trades_val = _combo_metric(trade_count, ind, fast_or_w, slow, hold)
        row: dict[str, Any] = {
            "indicator": ind,
            "fast": fast_or_w,
            "slow": (float("nan") if ind in SOLO_INDICATORS_NAN_SLOW else slow),
            "hold": hold,
            "sharpe": _combo_metric(sharpe, ind, fast_or_w, slow, hold),
            "total_return": _combo_metric(total_return, ind, fast_or_w, slow, hold),
            "calmar": _combo_metric(calmar, ind, fast_or_w, slow, hold),
            "max_drawdown": _combo_metric(max_drawdown, ind, fast_or_w, slow, hold),
            "win_rate": _combo_metric(win_rate, ind, fast_or_w, slow, hold),
            "trades": int(trades_val) if np.isfinite(trades_val) else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def rank_results(df: pd.DataFrame, metric: MetricName) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    sort_key = df[metric].replace([np.inf, -np.inf], np.nan)
    return (
        df.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=False, kind="stable", na_position="last")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )


def _fmt_int_or_dash(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return str(int(f))


def print_results_table(
    df: pd.DataFrame,
    *,
    top_n: int,
    metric: MetricName,
    console: Console | None = None,
) -> None:
    out = console or Console()
    out.print(DISCLAIMER)
    table = Table(title=f"Top {top_n} by {metric}")
    has_indicator = "indicator" in df.columns
    columns = (["indicator"] if has_indicator else []) + [
        "fast",
        "slow",
        "hold",
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    ]
    for col in columns:
        table.add_column(col)
    for _, row in df.head(top_n).iterrows():
        cells: list[str] = []
        if has_indicator:
            cells.append(str(row["indicator"]))
        cells.extend(
            [
                _fmt_int_or_dash(row["fast"]),
                _fmt_int_or_dash(row["slow"]),
                _fmt_int_or_dash(row["hold"]),
                f"{row['sharpe']:.3f}" if np.isfinite(row["sharpe"]) else "n/a",
                f"{row['total_return'] * 100:+.2f}%",
                f"{row['calmar']:.3f}" if np.isfinite(row["calmar"]) else "n/a",
                f"{row['max_drawdown'] * 100:+.2f}%",
                f"{row['win_rate'] * 100:.1f}%"
                if np.isfinite(row["win_rate"])
                else "n/a",
                _fmt_int_or_dash(row["trades"]),
            ]
        )
        table.add_row(*cells)
    out.print(table)


@click.command(name="vbt-sweep")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    help="Market to backtest.",
)
@click.option(
    "--start", "start_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--end", "end_arg", type=click.DateTime(formats=["%Y-%m-%d"]), default=None
)
@click.option(
    "--years",
    type=int,
    default=2,
    show_default=True,
    help="Trailing calendar years when --start is omitted.",
)
@click.option(
    "--universe",
    type=click.Choice(["sp500", "nifty50"]),
    default=None,
    help="Current index universe. Defaults to sp500 for US and nifty50 for India.",
)
@click.option(
    "--no-universe-cache",
    is_flag=True,
    default=False,
    help="Force live constituent refresh instead of today's cache.",
)
@click.option("--tickers", default=None, help="Comma-separated ticker list.")
@click.option(
    "--universe-file", default=None, help="Path to newline-separated ticker file."
)
@click.option(
    "--indicator",
    "indicator_arg",
    default="sma",
    show_default=True,
    help=(
        "Comma-separated subset of "
        f"{','.join(VALID_INDICATORS)} (or 'all'). EMA/SMA/MACD/sma_rsi use "
        "--fast/--slow/--hold; window-based indicators use their own --*-windows."
    ),
)
@click.option(
    "--fast",
    default="10,20,50",
    show_default=True,
    help="Comma-separated fast SMA/EMA windows.",
)
@click.option(
    "--slow",
    default="50,100,200",
    show_default=True,
    help="Comma-separated slow SMA/EMA windows (must be > fast).",
)
@click.option(
    "--hold",
    default="0",
    show_default=True,
    help="Comma-separated fixed hold lengths in bars; 0 disables fixed hold.",
)
@click.option(
    "--breakout-windows",
    "breakout_windows_arg",
    default=",".join(str(w) for w in DEFAULT_BREAKOUT_WINDOWS),
    show_default=True,
    help="Donchian breakout windows (used by breakout, vol_breakout, breakout_rsi).",
)
@click.option(
    "--bbands-windows",
    "bbands_windows_arg",
    default=",".join(str(w) for w in DEFAULT_BBANDS_WINDOWS),
    show_default=True,
    help=f"EMA Bollinger band windows (std fixed at {DEFAULT_BBANDS_STD}).",
)
@click.option(
    "--supertrend-periods",
    "supertrend_periods_arg",
    default=",".join(str(p) for p in DEFAULT_SUPERTREND_PERIODS),
    show_default=True,
    help=f"Supertrend ATR periods (multiplier fixed at {DEFAULT_SUPERTREND_MULT}).",
)
@click.option(
    "--keltner-windows",
    "keltner_windows_arg",
    default=",".join(str(w) for w in DEFAULT_KELTNER_WINDOWS),
    show_default=True,
    help=f"Keltner channel windows (multiplier fixed at {DEFAULT_KELTNER_MULT}).",
)
@click.option(
    "--rsi-thresholds",
    "rsi_thresholds_arg",
    default=",".join(str(t) for t in DEFAULT_RSI_THRESHOLDS),
    show_default=True,
    help=f"RSI entry thresholds (period fixed at {DEFAULT_RSI_PERIOD}).",
)
@click.option(
    "--obv-ema-windows",
    "obv_ema_windows_arg",
    default=",".join(str(w) for w in DEFAULT_OBV_EMA_WINDOWS),
    show_default=True,
    help="OBV smoothing EMA windows for obv_trend.",
)
@click.option(
    "--top", type=int, default=10, show_default=True, help="Print top N rows."
)
@click.option(
    "--csv", "output_csv", is_flag=True, help="Emit full results as CSV on stdout."
)
@click.option(
    "--metric",
    type=click.Choice(["sharpe", "total_return", "calmar"]),
    default="sharpe",
    show_default=True,
    help="Metric used to rank combinations.",
)
def vbt_sweep(
    market: str,
    start_arg: datetime | None,
    end_arg: datetime | None,
    years: int,
    universe: str | None,
    no_universe_cache: bool,
    tickers: str | None,
    universe_file: str | None,
    indicator_arg: str,
    fast: str,
    slow: str,
    hold: str,
    breakout_windows_arg: str,
    bbands_windows_arg: str,
    supertrend_periods_arg: str,
    keltner_windows_arg: str,
    rsi_thresholds_arg: str,
    obv_ema_windows_arg: str,
    top: int,
    output_csv: bool,
    metric: str,
) -> None:
    """Fast vectorbt grid search for exploration (not validation).

    Approximate results — slot allocation, partial exits, dividends, and custom
    slippage are not modeled. Validate winners with backtest-rolling.
    """
    indicators = parse_indicator_list(indicator_arg)
    fast_values = parse_int_list(fast, name="fast")
    slow_values = parse_int_list(slow, name="slow")
    hold_values = parse_int_list(hold, name="hold")
    breakout_windows = parse_int_list(breakout_windows_arg, name="breakout-windows")
    bbands_windows = parse_int_list(bbands_windows_arg, name="bbands-windows")
    supertrend_periods = parse_int_list(
        supertrend_periods_arg, name="supertrend-periods"
    )
    keltner_windows = parse_int_list(keltner_windows_arg, name="keltner-windows")
    rsi_thresholds = parse_int_list(rsi_thresholds_arg, name="rsi-thresholds")
    obv_ema_windows = parse_int_list(obv_ema_windows_arg, name="obv-ema-windows")

    end_date = (
        end_arg.date() if isinstance(end_arg, datetime) else (end_arg or date.today())
    )
    start_date = (
        start_arg.date()
        if isinstance(start_arg, datetime)
        else (start_arg or (end_date - timedelta(days=365 * int(years))))
    )
    if end_date < start_date:
        raise click.UsageError("--end must be on or after --start.")

    bench = DEFAULT_BENCHMARK.get(market, "SPY")
    console = Console()

    tv_symbols: list[str]
    universe_note: str | None = None
    if tickers:
        tv_symbols = [t.strip() for t in tickers.split(",") if t.strip()]
    elif universe_file:
        from pathlib import Path

        content = Path(universe_file).read_text()
        tv_symbols = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        resolved_universe = universe or ("nifty50" if market == "india" else "sp500")
        loaded = load_current_universe(
            resolved_universe,
            as_of=end_date,
            use_cache=not no_universe_cache,
        )
        tv_symbols = list(loaded.symbols)
        universe_note = (
            f"{loaded.name}: {len(loaded.symbols)} symbols from {loaded.source}; "
            f"cache={loaded.cached_path}"
        )

    if not tv_symbols:
        raise click.UsageError("No tickers resolved for the sweep.")

    yf_by_tv = {tv: tv_to_yf(tv, market) for tv in tv_symbols}
    yf_symbols = list(dict.fromkeys(list(yf_by_tv.values()) + [bench]))
    candidate_lookbacks: list[int] = []
    if any(ind in ("sma", "ema", "sma_rsi") for ind in indicators):
        candidate_lookbacks.append(max(slow_values))
    if any(ind in ("breakout", "vol_breakout", "breakout_rsi") for ind in indicators):
        candidate_lookbacks.append(max(breakout_windows))
    if "bbands" in indicators:
        candidate_lookbacks.append(max(bbands_windows))
    if "supertrend" in indicators:
        candidate_lookbacks.append(max(supertrend_periods))
    if "keltner" in indicators:
        candidate_lookbacks.append(max(keltner_windows))
    if "obv_trend" in indicators:
        candidate_lookbacks.append(max(obv_ema_windows))
    if "macd" in indicators:
        candidate_lookbacks.append(DEFAULT_MACD_SLOW + DEFAULT_MACD_SIGNAL)
    if "rsi" in indicators:
        candidate_lookbacks.append(DEFAULT_RSI_PERIOD * 3)
    if "vol_breakout" in indicators:
        candidate_lookbacks.append(DEFAULT_VOL_MA_WINDOW)
    max_lookback = max(candidate_lookbacks) if candidate_lookbacks else max(slow_values)
    warmup_days = max(max_lookback * 3 + 30, 90)
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)).date()
    fetch_end = end_date

    fetcher: PriceFetcher = click.get_current_context().obj or build_price_fetcher()
    price_panel = fetcher.fetch(yf_symbols, fetch_start, fetch_end)
    equity_symbols = [yf_by_tv[tv] for tv in tv_symbols if yf_by_tv[tv] in price_panel]
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    close = build_close_panel(
        price_panel,
        equity_symbols,
        start=start_ts,
        end=end_ts,
    )
    try:
        open_panel = build_open_panel(
            price_panel,
            equity_symbols,
            start=start_ts,
            end=end_ts,
        )
        open_panel = open_panel.reindex(index=close.index, columns=close.columns)
    except ValueError:
        open_panel = None

    high_panel: pd.DataFrame | None = None
    low_panel: pd.DataFrame | None = None
    volume_panel: pd.DataFrame | None = None
    if any(ind in HL_INDICATORS for ind in indicators):
        try:
            high_panel = build_high_panel(
                price_panel, equity_symbols, start=start_ts, end=end_ts
            ).reindex(index=close.index, columns=close.columns)
            low_panel = build_low_panel(
                price_panel, equity_symbols, start=start_ts, end=end_ts
            ).reindex(index=close.index, columns=close.columns)
        except ValueError:
            high_panel = None
            low_panel = None
    if any(ind in VOLUME_INDICATORS for ind in indicators):
        try:
            volume_panel = build_volume_panel(
                price_panel, equity_symbols, start=start_ts, end=end_ts
            ).reindex(index=close.index, columns=close.columns)
        except ValueError:
            volume_panel = None

    results = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        indicators=indicators,
        breakout_windows=breakout_windows,
        bbands_windows=bbands_windows,
        supertrend_periods=supertrend_periods,
        keltner_windows=keltner_windows,
        rsi_thresholds=rsi_thresholds,
        obv_ema_windows=obv_ema_windows,
        high=high_panel,
        low=low_panel,
        volume=volume_panel,
        open_=open_panel,
        initial_capital=INITIAL_CAPITAL_DEFAULT,
    )
    ranked = rank_results(results, cast(MetricName, metric))

    if output_csv:
        click.echo(ranked.to_csv(index=False))
        return

    console.print(
        f"[dim]Window: {start_date.isoformat()} to {end_date.isoformat()}  "
        f"indicators={','.join(indicators)}  symbols={close.shape[1]}  "
        f"combos={len(results)}  capital={INITIAL_CAPITAL_DEFAULT:,.0f}  "
        f"slippage=0[/dim]"
    )
    if universe_note:
        console.print(f"[dim]Universe: {universe_note}[/dim]")
    print_results_table(ranked, top_n=int(top), metric=cast(MetricName, metric))
