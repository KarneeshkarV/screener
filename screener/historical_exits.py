"""Exit-strategy evaluators for the historical backtester.

The backtester previously held every position until a fixed ``--hold`` cutoff.
This module adds configurable exits — stop-loss, take-profit, trailing stop,
and technical-signal exits — resolved per-ticker with first-triggered-wins
semantics.

Signal registry
---------------
Built-in exit signals are defined in ``EXIT_SIGNALS``.  Each signal is a
function ``fn(df, idx) -> bool`` where ``df`` is a full OHLCV DataFrame with
pre-computed indicator columns (see ``_attach_indicators``) and ``idx`` is the
absolute row index of the current bar.  Register new signals by adding an
entry to ``EXIT_SIGNALS`` — the CLI picks up keys automatically.

Priority within a bar (when more than one exit triggers on the same day):
``stop`` > ``target`` > ``trail`` > ``signal:*`` > ``time``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


# ── policy ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExitPolicy:
    """Configuration for per-ticker exit resolution.

    All percentages are expressed as decimals (``0.08`` = 8%).  ``None`` on a
    level-based field disables that exit type.
    """
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    signals: tuple[str, ...] = ()
    time_stop_bars: Optional[int] = None

    def is_noop(self) -> bool:
        return (
            self.stop_loss is None
            and self.take_profit is None
            and self.trailing_stop is None
            and not self.signals
        )


# ── indicator precompute ────────────────────────────────────────────────────

def _attach_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *ohlcv* with indicator columns attached.

    Computed once per ticker before the bar-walk so signal evaluators are
    constant-time per bar.  Prefixed with ``_`` to avoid colliding with
    existing columns.
    """
    df = ohlcv.copy()
    close = pd.to_numeric(df["close"], errors="coerce").ffill()
    df["_close"] = close

    df["_ema5"]   = close.ewm(span=5,   adjust=False).mean()
    df["_ema20"]  = close.ewm(span=20,  adjust=False).mean()
    df["_ema50"]  = close.ewm(span=50,  adjust=False).mean()
    df["_ema100"] = close.ewm(span=100, adjust=False).mean()
    df["_ema200"] = close.ewm(span=200, adjust=False).mean()

    df["_sma10"] = close.rolling(10).mean()
    df["_sma20"] = close.rolling(20).mean()
    df["_high_52w"] = close.rolling(252, min_periods=20).max()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["_macd"]        = ema12 - ema26
    df["_macd_signal"] = df["_macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["_rsi"] = (100 - 100 / (1 + rs)).fillna(100.0)

    ag2 = delta.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    al2 = (-delta.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    rs2 = ag2 / al2.replace(0, float("nan"))
    df["_rsi2"] = (100 - 100 / (1 + rs2)).fillna(100.0)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["_bb_upper"] = bb_mid + 2.0 * bb_std

    return df


# ── built-in signal evaluators ──────────────────────────────────────────────

def _sig_rsi_overbought(df: pd.DataFrame, idx: int) -> bool:
    v = df["_rsi"].iloc[idx]
    return bool(pd.notna(v) and v >= 70)


def _sig_ema20_break(df: pd.DataFrame, idx: int) -> bool:
    if idx == 0:
        return False
    c_now, c_prev = df["_close"].iloc[idx], df["_close"].iloc[idx - 1]
    e_now, e_prev = df["_ema20"].iloc[idx], df["_ema20"].iloc[idx - 1]
    if pd.isna(e_now) or pd.isna(e_prev):
        return False
    return bool(c_prev >= e_prev and c_now < e_now)


def _sig_macd_cross_down(df: pd.DataFrame, idx: int) -> bool:
    if idx == 0:
        return False
    m_now, m_prev = df["_macd"].iloc[idx], df["_macd"].iloc[idx - 1]
    s_now, s_prev = df["_macd_signal"].iloc[idx], df["_macd_signal"].iloc[idx - 1]
    if pd.isna(m_now) or pd.isna(s_now) or pd.isna(m_prev) or pd.isna(s_prev):
        return False
    return bool(m_prev >= s_prev and m_now < s_now)


def _sig_ema_stack_break(df: pd.DataFrame, idx: int) -> bool:
    e5, e20 = df["_ema5"].iloc[idx], df["_ema20"].iloc[idx]
    if pd.isna(e5) or pd.isna(e20):
        return False
    return bool(e5 < e20)


def _sig_ema_stack_full_break(df: pd.DataFrame, idx: int) -> bool:
    """Mirror of the `ema` entry criterion — fires when the EMA5>EMA20>EMA100>EMA200
    bullish stack is no longer strictly ordered (any inequality broken)."""
    e5   = df["_ema5"].iloc[idx]
    e20  = df["_ema20"].iloc[idx]
    e100 = df["_ema100"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(e5) or pd.isna(e20) or pd.isna(e100) or pd.isna(e200):
        return False
    return not bool(e5 > e20 > e100 > e200 > 0)


def _sig_bb_upper_tag(df: pd.DataFrame, idx: int) -> bool:
    c, u = df["_close"].iloc[idx], df["_bb_upper"].iloc[idx]
    if pd.isna(u):
        return False
    return bool(c >= u)


# ── criterion-invalidation signals ──────────────────────────────────────────
# Each fires when the entry criterion of the same name is no longer satisfied.

def _sig_breakout_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: close within 10% of 52w high.  Exit: close drops below that band."""
    c = df["_close"].iloc[idx]
    h = df["_high_52w"].iloc[idx]
    if pd.isna(c) or pd.isna(h) or h <= 0:
        return False
    return bool(c < h * 0.90)


def _sig_pullback_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: |close - EMA20|/EMA20 ≤ 3% and EMA20 > EMA100 > EMA200.
    Exit: price drifts >3% from EMA20 OR the EMA20>EMA100>EMA200 trend breaks."""
    c   = df["_close"].iloc[idx]
    e20 = df["_ema20"].iloc[idx]
    e100 = df["_ema100"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(c) or pd.isna(e20) or pd.isna(e100) or pd.isna(e200) or e20 <= 0:
        return False
    drift_ok = abs(c / e20 - 1.0) <= 0.03
    trend_ok = e20 > e100 > e200 > 0
    return not (drift_ok and trend_ok)


def _sig_oversold_rsi_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: RSI<35 in EMA100>EMA200 regime.  Exit: RSI rebounds out of oversold
    (mean reversion complete) or the regime breaks."""
    r    = df["_rsi"].iloc[idx]
    e100 = df["_ema100"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(r) or pd.isna(e100) or pd.isna(e200):
        return False
    regime_ok = e100 > e200
    return bool(r >= 50 or not regime_ok)


def _sig_golden_cross_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: EMA50 crossed above EMA200.  Exit: EMA50 back below EMA200."""
    e50  = df["_ema50"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(e50) or pd.isna(e200):
        return False
    return bool(e50 < e200)


def _sig_rsi2_oversold_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: Connors RSI(2)<10 + rising EMA200.  Exit: RSI2 mean-reverts (>70)
    or the EMA200 rising-regime breaks (close below EMA200)."""
    r2   = df["_rsi2"].iloc[idx]
    c    = df["_close"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(r2) or pd.isna(c) or pd.isna(e200):
        return False
    return bool(r2 >= 70 or c < e200)


def _sig_macd_cross_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: MACD>signal and above zero.  Exit: MACD back below signal line."""
    m = df["_macd"].iloc[idx]
    s = df["_macd_signal"].iloc[idx]
    if pd.isna(m) or pd.isna(s):
        return False
    return bool(m < s)


def _sig_sma_cross_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: SMA10 crossed above SMA20 + price above EMA200.  Exit: SMA10<SMA20
    OR price drops below EMA200."""
    s10  = df["_sma10"].iloc[idx]
    s20  = df["_sma20"].iloc[idx]
    c    = df["_close"].iloc[idx]
    e200 = df["_ema200"].iloc[idx]
    if pd.isna(s10) or pd.isna(s20) or pd.isna(c) or pd.isna(e200):
        return False
    return bool(s10 < s20 or c < e200)


def _sig_ema_breakout_invalid(df: pd.DataFrame, idx: int) -> bool:
    """Entry: ema AND breakout.  Exit: EITHER the EMA stack breaks OR price
    leaves the breakout band (>10% below 52w high)."""
    return _sig_ema_stack_full_break(df, idx) or _sig_breakout_invalid(df, idx)


SignalFn = Callable[[pd.DataFrame, int], bool]

EXIT_SIGNALS: dict[str, SignalFn] = {
    "rsi_overbought":        _sig_rsi_overbought,
    "ema20_break":           _sig_ema20_break,
    "macd_cross_down":       _sig_macd_cross_down,
    "ema_stack_break":       _sig_ema_stack_break,
    "ema_stack_full_break":  _sig_ema_stack_full_break,
    "bb_upper_tag":          _sig_bb_upper_tag,
    # criterion-invalidation signals
    "breakout_invalid":      _sig_breakout_invalid,
    "pullback_invalid":      _sig_pullback_invalid,
    "oversold_rsi_invalid":  _sig_oversold_rsi_invalid,
    "golden_cross_invalid":  _sig_golden_cross_invalid,
    "rsi2_oversold_invalid": _sig_rsi2_oversold_invalid,
    "macd_cross_invalid":    _sig_macd_cross_invalid,
    "sma_cross_invalid":     _sig_sma_cross_invalid,
    "ema_breakout_invalid":  _sig_ema_breakout_invalid,
}


# ── resolver ────────────────────────────────────────────────────────────────

def resolve_exit(
    ohlcv_full: pd.DataFrame,
    entry_abs_idx: int,
    forward_len: int,
    policy: ExitPolicy,
) -> tuple[int, str]:
    """Walk forward bars from *entry_abs_idx*; return ``(exit_bar, reason)``.

    ``exit_bar`` is the offset from entry (0 = entry day itself).  Matches the
    ``trading_days`` convention used in the per-ticker ledger.

    Reasons: ``stop``, ``target``, ``trail``, ``signal:<name>``, ``time``.
    """
    if forward_len <= 0:
        return 0, "time"

    # Upper bound on the walk: either the caller's time cap or all available
    # forward bars.  len-1 because bar 0 is the entry day.
    max_bar = forward_len - 1
    if policy.time_stop_bars is not None:
        max_bar = min(max_bar, int(policy.time_stop_bars))
    if max_bar <= 0:
        return 0, "time"

    df = _attach_indicators(ohlcv_full)
    entry_close = float(df["_close"].iloc[entry_abs_idx])
    if entry_close <= 0 or pd.isna(entry_close):
        return max_bar, "time"

    running_max = entry_close
    signal_fns: list[tuple[str, SignalFn]] = [
        (name, EXIT_SIGNALS[name]) for name in policy.signals if name in EXIT_SIGNALS
    ]

    for k in range(1, max_bar + 1):
        abs_idx = entry_abs_idx + k
        if abs_idx >= len(df):
            break
        bar_close = float(df["_close"].iloc[abs_idx])
        if pd.isna(bar_close):
            continue
        if bar_close > running_max:
            running_max = bar_close

        if policy.stop_loss is not None:
            if bar_close <= entry_close * (1.0 - policy.stop_loss):
                return k, "stop"
        if policy.take_profit is not None:
            if bar_close >= entry_close * (1.0 + policy.take_profit):
                return k, "target"
        if policy.trailing_stop is not None:
            if bar_close <= running_max * (1.0 - policy.trailing_stop):
                return k, "trail"
        for name, fn in signal_fns:
            if fn(df, abs_idx):
                return k, f"signal:{name}"

    return max_bar, "time"


# ── re-entry resolver ───────────────────────────────────────────────────────

Trade = tuple[int, int, str]  # (entry_bar_rel, exit_bar_rel, reason)


def resolve_trades(
    ohlcv_full: pd.DataFrame,
    entry_abs_idx: int,
    forward_len: int,
    policy: ExitPolicy,
    entry_eval_fn: Callable[[int], bool],
    cooldown_bars: int = 1,
) -> list[Trade]:
    """Walk the hold window and return a sequence of round-trip trades.

    Bar indices in the returned tuples are relative to the initial entry
    (bar 0 = ``entry_abs_idx``).  Trade 1 is seeded from bar 0.  Subsequent
    trades open only on invalid→valid transitions of ``entry_eval_fn`` while
    flat, after a ``cooldown_bars`` gap from the previous exit.

    Each trade's exit is resolved by ``resolve_exit`` against a per-trade
    ``ExitPolicy`` whose ``time_stop_bars`` is capped to the remaining hold
    window (measured from the initial entry), so the total hold cap is
    honoured across all trades.
    """
    if forward_len <= 0:
        return [(0, 0, "time")]

    # Resolve trade 1 using the original policy (its time_stop_bars is
    # already anchored to the initial entry).
    exit_rel, reason = resolve_exit(ohlcv_full, entry_abs_idx, forward_len, policy)
    trades: list[Trade] = [(0, exit_rel, reason)]

    # Effective hold cap from the initial entry.  None means no time-stop.
    hold_cap = policy.time_stop_bars
    if hold_cap is not None:
        hold_cap = min(int(hold_cap), forward_len - 1)
    else:
        hold_cap = forward_len - 1

    cursor = exit_rel + max(1, int(cooldown_bars))
    prev_valid = bool(entry_eval_fn(entry_abs_idx + exit_rel))

    while cursor <= hold_cap:
        abs_idx = entry_abs_idx + cursor
        if abs_idx >= len(ohlcv_full):
            break
        now_valid = bool(entry_eval_fn(abs_idx))
        if now_valid and not prev_valid:
            remaining = hold_cap - cursor
            if remaining <= 0:
                break
            trade_policy = ExitPolicy(
                stop_loss=policy.stop_loss,
                take_profit=policy.take_profit,
                trailing_stop=policy.trailing_stop,
                signals=policy.signals,
                time_stop_bars=remaining,
            )
            sub_exit_rel, sub_reason = resolve_exit(
                ohlcv_full, abs_idx, remaining + 1, trade_policy
            )
            trade_entry = cursor
            trade_exit = cursor + sub_exit_rel
            trades.append((trade_entry, trade_exit, sub_reason))
            cursor = trade_exit + max(1, int(cooldown_bars))
            # Seed prev_valid from the new exit bar so the next edge-based
            # transition is measured from there.
            tail_idx = entry_abs_idx + trade_exit
            prev_valid = bool(entry_eval_fn(tail_idx)) if tail_idx < len(ohlcv_full) else False
        else:
            prev_valid = now_valid
            cursor += 1

    return trades
