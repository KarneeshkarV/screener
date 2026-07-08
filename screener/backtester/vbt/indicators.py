"""Indicator combo generation and signal panel construction for vbt sweeps."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from screener.backtester.vbt.config import (
    DEFAULT_BBANDS_STD,
    DEFAULT_BBANDS_WINDOWS,
    DEFAULT_BREAKOUT_WINDOWS,
    DEFAULT_KELTNER_MULT,
    DEFAULT_KELTNER_WINDOWS,
    DEFAULT_MACD_FAST,
    DEFAULT_MACD_SIGNAL,
    DEFAULT_MACD_SLOW,
    DEFAULT_OBV_EMA_WINDOWS,
    DEFAULT_RSI_PERIOD,
    DEFAULT_RSI_THRESHOLDS,
    DEFAULT_SUPERTREND_MULT,
    DEFAULT_SUPERTREND_PERIODS,
    DEFAULT_VOL_MA_WINDOW,
    DEFAULT_VOL_MULTIPLIER,
    HL_INDICATORS,
    StrategyBuilder,
    StrategyName,
    VOLUME_INDICATORS,
)


@dataclass(frozen=True)
class IndicatorCombo:
    """Normalized parameter tuple used by the signal builder."""

    indicator: str
    fast: int
    slow: int
    hold: int

    @classmethod
    def from_tuple(cls, raw: tuple[str, int, int, int]) -> IndicatorCombo:
        indicator, fast, slow, hold = raw
        return cls(indicator, int(fast), int(slow), int(hold))

    def as_tuple(self) -> tuple[str, int, int, int]:
        return (self.indicator, self.fast, self.slow, self.hold)


def _sma(close: pd.DataFrame, window: int, vbt: Any) -> pd.DataFrame:
    # vbt is Any (optional dep), so ma_out and its operations are Any.
    ma_out = vbt.MA.run(close, window=window).ma
    if isinstance(ma_out.columns, pd.MultiIndex):
        return cast(pd.DataFrame, ma_out.xs(window, axis=1, level="ma_window"))
    return cast(pd.DataFrame, ma_out)


def _fixed_hold_exits_np(arr: np.ndarray, hold: int) -> np.ndarray:
    """Numpy-only equivalent of :func:`_fixed_hold_exits` for vectorized builds."""
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
        # xs(axis=1, level=...) on a MultiIndex column frame returns a DataFrame;
        # pandas-stubs types it as DataFrame | Series, so narrow with cast.
        return cast(pd.DataFrame, ma_panel.xs(window, axis=1, level="ma_window"))
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

    def _pair_grid(name: str) -> list[IndicatorCombo]:
        out: list[IndicatorCombo] = []
        for fast, slow, hold in itertools.product(
            fast_values, slow_values, hold_values
        ):
            if slow <= fast:
                continue
            out.append(IndicatorCombo(name, int(fast), int(slow), int(hold)))
        return out

    def _window_grid(name: str, windows: list[int]) -> list[IndicatorCombo]:
        return [
            IndicatorCombo(name, int(w), 0, int(hold))
            for w, hold in itertools.product(windows, hold_values)
        ]

    combos: list[IndicatorCombo] = []
    for ind in indicators:
        if ind in ("sma", "ema", "sma_rsi"):
            combos.extend(_pair_grid(ind))
        elif ind == "macd":
            combos.extend(
                IndicatorCombo("macd", DEFAULT_MACD_FAST, DEFAULT_MACD_SLOW, int(hold))
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
                IndicatorCombo("rsi", int(thresh), 0, int(hold))
                for thresh, hold in itertools.product(rsi_thresholds, hold_values)
            )
        else:
            raise ValueError(f"Unknown indicator: {ind!r}")
    return [combo.as_tuple() for combo in combos]


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
    """Per-column Supertrend direction state machine."""
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


@dataclass
class SignalCache:
    """Precomputed per-indicator arrays for all requested combos."""

    sma_above_fast_by_w: dict[int, np.ndarray]
    sma_crossed_above_by_w: dict[int, np.ndarray]
    sma_crossed_below_by_w: dict[int, np.ndarray]
    ema_above_fast_by_w: dict[int, np.ndarray]
    ema_crossed_above_by_w: dict[int, np.ndarray]
    ema_crossed_below_by_w: dict[int, np.ndarray]
    breakout_entries_by_w: dict[int, np.ndarray]
    bbands_entries_by_w: dict[int, np.ndarray]
    bbands_exits_by_w: dict[int, np.ndarray]
    supertrend_signals_by_p: dict[int, tuple[np.ndarray, np.ndarray]]
    keltner_entries_by_w: dict[int, np.ndarray]
    keltner_exits_by_w: dict[int, np.ndarray]
    rsi_threshold_signals: dict[int, tuple[np.ndarray, np.ndarray]]
    rsi_filter_above_50: np.ndarray | None
    macd_entries: np.ndarray | None
    macd_exits: np.ndarray | None
    vol_above_ma: np.ndarray | None
    obv_entries_by_w: dict[int, np.ndarray]
    obv_exits_by_w: dict[int, np.ndarray]

    @classmethod
    def build(
        cls,
        close: pd.DataFrame,
        combos: list[IndicatorCombo],
        *,
        vbt: Any,
        high: pd.DataFrame | None,
        low: pd.DataFrame | None,
        volume: pd.DataFrame | None,
    ) -> SignalCache:
        from vectorbt.generic.nb import crossed_above_nb

        close_arr = np.ascontiguousarray(close.to_numpy(dtype=float))
        needs_high_low = any(c.indicator in HL_INDICATORS for c in combos)
        needs_volume = any(c.indicator in VOLUME_INDICATORS for c in combos)
        if needs_high_low and (high is None or low is None):
            raise ValueError(
                "supertrend / keltner indicators require high & low panels."
            )
        if needs_volume and volume is None:
            raise ValueError(
                "vol_breakout / obv_trend indicators require a volume panel."
            )

        # ---- SMA intermediates (preserves frozen regression bit-for-bit) ----
        sma_fast_set = sorted(
            {c.fast for c in combos if c.indicator in ("sma", "sma_rsi")}
        )
        sma_slow_set = sorted(
            {c.slow for c in combos if c.indicator in ("sma", "sma_rsi")}
        )
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
        ema_fast_set = sorted({c.fast for c in combos if c.indicator == "ema"})
        ema_slow_set = sorted({c.slow for c in combos if c.indicator == "ema"})
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

        breakout_w_set = sorted(
            {
                c.fast
                for c in combos
                if c.indicator in ("breakout", "vol_breakout", "breakout_rsi")
            }
        )
        breakout_entries_by_w: dict[int, np.ndarray] = {}
        for w in breakout_w_set:
            rmax = close.rolling(w).max().shift(1).to_numpy(dtype=float)
            breakout_entries_by_w[w] = crossed_above_nb(
                close_arr, np.ascontiguousarray(rmax)
            )

        bbands_w_set = sorted({c.fast for c in combos if c.indicator == "bbands"})
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
            bbands_exits_by_w[w] = crossed_above_nb(
                np.ascontiguousarray(middle_arr), close_arr
            )

        supertrend_p_set = sorted(
            {c.fast for c in combos if c.indicator == "supertrend"}
        )
        supertrend_signals_by_p: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if supertrend_p_set:
            assert high is not None and low is not None
            high_arr = high.to_numpy(dtype=float)
            low_arr = low.to_numpy(dtype=float)
            for p in supertrend_p_set:
                supertrend_signals_by_p[p] = _supertrend_signals_np(
                    close_arr, high_arr, low_arr, p, DEFAULT_SUPERTREND_MULT
                )

        keltner_w_set = sorted({c.fast for c in combos if c.indicator == "keltner"})
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

        rsi_needed = any(
            c.indicator in ("rsi", "sma_rsi", "breakout_rsi") for c in combos
        )
        rsi_arr: np.ndarray | None = (
            _rsi_wilder(close, DEFAULT_RSI_PERIOD) if rsi_needed else None
        )
        rsi_threshold_signals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        rsi_filter_above_50: np.ndarray | None = None
        if rsi_arr is not None:
            rsi_filter_above_50 = (rsi_arr > 50).astype(bool)
            for thresh in sorted({c.fast for c in combos if c.indicator == "rsi"}):
                thresh_arr = np.full_like(rsi_arr, float(thresh))
                entries_t = crossed_above_nb(
                    np.ascontiguousarray(rsi_arr), np.ascontiguousarray(thresh_arr)
                )
                exits_t = crossed_above_nb(
                    np.ascontiguousarray(thresh_arr), np.ascontiguousarray(rsi_arr)
                )
                rsi_threshold_signals[thresh] = (entries_t, exits_t)

        macd_needed = any(c.indicator == "macd" for c in combos)
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

        vol_above_ma: np.ndarray | None = None
        if any(c.indicator == "vol_breakout" for c in combos):
            assert volume is not None
            vol_arr = volume.to_numpy(dtype=float)
            vol_ma = volume.rolling(DEFAULT_VOL_MA_WINDOW).mean().to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                vol_above_ma = vol_arr > (DEFAULT_VOL_MULTIPLIER * vol_ma)
            vol_above_ma[~np.isfinite(vol_ma)] = False

        obv_w_set = sorted({c.fast for c in combos if c.indicator == "obv_trend"})
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

        return cls(
            sma_above_fast_by_w=sma_above_fast_by_w,
            sma_crossed_above_by_w=sma_crossed_above_by_w,
            sma_crossed_below_by_w=sma_crossed_below_by_w,
            ema_above_fast_by_w=ema_above_fast_by_w,
            ema_crossed_above_by_w=ema_crossed_above_by_w,
            ema_crossed_below_by_w=ema_crossed_below_by_w,
            breakout_entries_by_w=breakout_entries_by_w,
            bbands_entries_by_w=bbands_entries_by_w,
            bbands_exits_by_w=bbands_exits_by_w,
            supertrend_signals_by_p=supertrend_signals_by_p,
            keltner_entries_by_w=keltner_entries_by_w,
            keltner_exits_by_w=keltner_exits_by_w,
            rsi_threshold_signals=rsi_threshold_signals,
            rsi_filter_above_50=rsi_filter_above_50,
            macd_entries=macd_entries,
            macd_exits=macd_exits,
            vol_above_ma=vol_above_ma,
            obv_entries_by_w=obv_entries_by_w,
            obv_exits_by_w=obv_exits_by_w,
        )

    def signals_for(self, combo: IndicatorCombo) -> tuple[np.ndarray, np.ndarray]:
        ind = combo.indicator
        fast = combo.fast
        slow = combo.slow
        if ind == "sma":
            entries = self.sma_crossed_above_by_w[slow] & self.sma_above_fast_by_w[fast]
            return entries, self.sma_crossed_below_by_w[slow]
        if ind == "ema":
            entries = self.ema_crossed_above_by_w[slow] & self.ema_above_fast_by_w[fast]
            return entries, self.ema_crossed_below_by_w[slow]
        if ind == "breakout":
            entries = self.breakout_entries_by_w[fast]
            return entries, np.zeros_like(entries, dtype=bool)
        if ind == "bbands":
            return self.bbands_entries_by_w[fast], self.bbands_exits_by_w[fast]
        if ind == "supertrend":
            return self.supertrend_signals_by_p[fast]
        if ind == "keltner":
            return self.keltner_entries_by_w[fast], self.keltner_exits_by_w[fast]
        if ind == "rsi":
            return self.rsi_threshold_signals[fast]
        if ind == "macd":
            assert self.macd_entries is not None and self.macd_exits is not None
            return self.macd_entries, self.macd_exits
        if ind == "vol_breakout":
            assert self.vol_above_ma is not None
            entries = self.breakout_entries_by_w[fast] & self.vol_above_ma
            return entries, np.zeros_like(entries, dtype=bool)
        if ind == "obv_trend":
            return self.obv_entries_by_w[fast], self.obv_exits_by_w[fast]
        if ind == "sma_rsi":
            assert self.rsi_filter_above_50 is not None
            entries = (
                self.sma_crossed_above_by_w[slow]
                & self.sma_above_fast_by_w[fast]
                & self.rsi_filter_above_50
            )
            return entries, self.sma_crossed_below_by_w[slow]
        if ind == "breakout_rsi":
            assert self.rsi_filter_above_50 is not None
            entries = self.breakout_entries_by_w[fast] & self.rsi_filter_above_50
            return entries, np.zeros_like(entries, dtype=bool)
        raise ValueError(f"Unknown indicator: {ind!r}")


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
    """
    normalized = [IndicatorCombo.from_tuple(c) for c in combos]
    tickers = list(close.columns)
    close_arr = np.ascontiguousarray(close.to_numpy(dtype=float))
    n_days, n_tickers = close_arr.shape
    cache = SignalCache.build(
        close, normalized, vbt=vbt, high=high, low=low, volume=volume
    )

    n_combos = len(normalized)
    entries_panel = np.empty((n_days, n_combos * n_tickers), dtype=bool)
    exits_panel = np.empty((n_days, n_combos * n_tickers), dtype=bool)

    for k, combo in enumerate(normalized):
        entries_k, exits_k = cache.signals_for(combo)
        if combo.hold > 0:
            exits_k = exits_k | _fixed_hold_exits_np(entries_k, combo.hold)
        start = k * n_tickers
        stop = start + n_tickers
        entries_panel[:, start:stop] = entries_k
        exits_panel[:, start:stop] = exits_k

    col_index = pd.MultiIndex.from_tuples(
        [
            (combo.indicator, combo.fast, combo.slow, combo.hold, ticker)
            for combo in normalized
            for ticker in tickers
        ],
        names=["indicator", "fast", "slow", "hold", "ticker"],
    )
    entries_df = pd.DataFrame(entries_panel, index=close.index, columns=col_index)
    exits_df = pd.DataFrame(exits_panel, index=close.index, columns=col_index)
    return entries_df, exits_df
