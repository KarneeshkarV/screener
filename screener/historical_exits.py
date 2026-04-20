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

    df["_ema5"]  = close.ewm(span=5,  adjust=False).mean()
    df["_ema20"] = close.ewm(span=20, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["_macd"]        = ema12 - ema26
    df["_macd_signal"] = df["_macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["_rsi"] = (100 - 100 / (1 + rs)).fillna(100.0)

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


def _sig_bb_upper_tag(df: pd.DataFrame, idx: int) -> bool:
    c, u = df["_close"].iloc[idx], df["_bb_upper"].iloc[idx]
    if pd.isna(u):
        return False
    return bool(c >= u)


SignalFn = Callable[[pd.DataFrame, int], bool]

EXIT_SIGNALS: dict[str, SignalFn] = {
    "rsi_overbought":  _sig_rsi_overbought,
    "ema20_break":     _sig_ema20_break,
    "macd_cross_down": _sig_macd_cross_down,
    "ema_stack_break": _sig_ema_stack_break,
    "bb_upper_tag":    _sig_bb_upper_tag,
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
