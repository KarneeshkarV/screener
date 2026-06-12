"""Deterministic pinned-OHLC bar builders for hand-computed trade scenarios.

NO RNG — every high/low/open/close is explicitly chosen so intrabar assertions
are exact.  All frames use a Monday-anchored business-date index.

Convention (verified from core.py):
  signal_idx=3 → entry_idx=4 (next-bar open fill, entry_order_type='moo').
"""

from __future__ import annotations


import pandas as pd


# ---------------------------------------------------------------------------
# Bar builder helpers
# ---------------------------------------------------------------------------

_START = "2024-01-01"  # Monday


def _bdate_range(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(_START, periods=n)


def _make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build an OHLCV frame from a list of dicts (open,high,low,close,volume)."""
    idx = _bdate_range(len(rows))
    df = pd.DataFrame(rows, index=idx)
    # Ensure correct column order and types
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Scenario 1 — buy-and-hold to EOD (force-close at last bar)
# ---------------------------------------------------------------------------
# signal_idx=3 (bar3, as_of date), entry at bar4 open=100.0
# hold=100 (far beyond), 10 bars total → exits at bar9 close=115.0
#
# entry_fill  = 100.0  (slippage_bps=0)
# shares      = 100_000 / 100.0 = 1000.0  (commission_bps=0)
# entry_cost  = 100_000.0
# exit_fill   = 115.0  (slippage_bps=0, sell at last-bar close)
# exit_value  = 1000.0 * 115.0 = 115_000.0
# pnl         = 115_000.0 - 100_000.0 = 15_000.0
# return_pct  = 0.15


def bars_s1_buy_and_hold() -> pd.DataFrame:
    rows = [
        # bar0–bar3: pre-signal; close rises gently
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        # bar3 = signal bar (as_of date)
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4 = entry bar; open=100.0 → entry_fill=100.0
        {"open": 100.0, "high": 102.0, "low": 99.5, "close": 101.0, "volume": 10_000},
        # bars 5–9: price drifts up; no stop/target
        {"open": 101.0, "high": 103.0, "low": 100.5, "close": 102.0, "volume": 10_000},
        {"open": 102.0, "high": 105.0, "low": 101.5, "close": 104.0, "volume": 10_000},
        {"open": 104.0, "high": 108.0, "low": 103.5, "close": 107.0, "volume": 10_000},
        {"open": 107.0, "high": 111.0, "low": 106.5, "close": 110.0, "volume": 10_000},
        # bar9 = last bar; close=115.0 → force-close fill
        {"open": 110.0, "high": 116.0, "low": 109.5, "close": 115.0, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 2 — stop hit intrabar → fill at stop_ref (gap_fills=True, open>stop)
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# stop_loss=0.05 → stop_ref = 100.0 * 0.95 = 95.0
# bar5: open=99.0 > stop_ref=95.0, low=89.0 ≤ 95.0 → fill at stop_ref=95.0
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 95.0
# exit_value = 1000.0 * 95.0 = 95_000.0
# pnl        = 95_000.0 - 100_000.0 = -5_000.0
# return_pct = -0.05


def bars_s2_stop_intrabar() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0; high > low → no exit triggers (loop starts at i=5)
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5: open=99.0 > stop_ref=95.0, low=89.0 → stop hit, fill at 95.0
        {"open": 99.0, "high": 99.5, "low": 89.0, "close": 94.0, "volume": 10_000},
        {"open": 94.0, "high": 95.0, "low": 93.0, "close": 94.5, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 3 — target hit intrabar → fill at target_ref (open < target)
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# take_profit=0.10 → target_ref = 100.0 * 1.10 = 110.0
# bar5: open=102.0 < target_ref=110.0, high=115.0 ≥ 110.0 → fill at 110.0
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 110.0
# exit_value = 1000.0 * 110.0 = 110_000.0
# pnl        = 110_000.0 - 100_000.0 = 10_000.0
# return_pct = 0.10


def bars_s3_target_intrabar() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0; no exit at entry bar (loop starts at i=5)
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5: open=102.0 < target=110.0, high=115.0 → target hit, fill at 110.0
        {"open": 102.0, "high": 115.0, "low": 101.5, "close": 112.0, "volume": 10_000},
        {"open": 112.0, "high": 113.0, "low": 111.5, "close": 112.5, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 4a — gap-DOWN through stop: gap_fills=True → fill at OPEN
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# stop_loss=0.05 → stop_ref = 95.0
# bar5: open=90.0 ≤ stop_ref=95.0 AND gap_fills=True → fill at open=90.0
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 90.0  (gap fill at open)
# exit_value = 90_000.0
# pnl        = -10_000.0
# return_pct = -0.10


def bars_s4_gap_down() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5: GAPS DOWN; open=90.0 ≤ stop_ref=95.0
        {"open": 90.0, "high": 91.0, "low": 88.0, "close": 89.0, "volume": 10_000},
        {"open": 89.0, "high": 90.0, "low": 88.5, "close": 89.5, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 5 — gap-UP through target: gap_fills=True → fill at OPEN
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# take_profit=0.10 → target_ref = 110.0
# bar5: open=115.0 ≥ target_ref=110.0 AND gap_fills=True → fill at open=115.0
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 115.0  (gap fill at open)
# exit_value = 115_000.0
# pnl        = 15_000.0
# return_pct = 0.15


def bars_s5_gap_up() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5: GAPS UP; open=115.0 ≥ target_ref=110.0
        {"open": 115.0, "high": 116.0, "low": 114.0, "close": 115.5, "volume": 10_000},
        {"open": 115.5, "high": 116.0, "low": 115.0, "close": 115.8, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 6 — trailing-stop ratchet
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# trailing_stop=0.10 (10%)
# Initial trail_ref = peak * 0.9 = 100.0 * 0.9 = 90.0
#
# bar5 (i=5): open=100.0, high=120.0, low=99.5, close=119.0
#   high=120 > peak=100 → peak updates to 120.0 after no-exit check
#   trail_ref = 120.0 * 0.9 = 108.0
#   low=99.5 > old trail_ref=90.0 → no hit at this bar
#   NOTE: peak is updated at END of _check_exit_at_bar after all exit checks fail
#
# bar6 (i=6): open=119.0, high=119.5, low=105.0, close=112.0
#   trail_ref = 120.0 * 0.9 = 108.0
#   low=105.0 ≤ 108.0 → trail hit
#   bar6 open=119.0 > trail_ref=108.0 → fill at trail_ref=108.0  (no gap fill)
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 108.0
# exit_value = 1000.0 * 108.0 = 108_000.0
# pnl        = 108_000.0 - 100_000.0 = 8_000.0
# return_pct = 0.08


def bars_s6_trailing_stop() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0  (loop starts at i=5)
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5: high=120.0 → peak lifts to 120; low=99.5 > trail_ref=90 → no exit
        {"open": 100.0, "high": 120.0, "low": 99.5, "close": 119.0, "volume": 10_000},
        # bar6: trail_ref=108.0; low=105.0 ≤ 108.0 → trail hit; open=119>108 → fill@108
        {"open": 119.0, "high": 119.5, "low": 105.0, "close": 112.0, "volume": 10_000},
        {"open": 112.0, "high": 113.0, "low": 111.5, "close": 112.5, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 7 — partial scale-out (50% at +10%) then time exit
# ---------------------------------------------------------------------------
# Driven via run_backtest; as_of = bars.index[3].date()
# entry_order_type='moo', entry at bar4 open=100.0
# partial_exits=((0.10, 0.5),): first tranche at entry_fill*1.10=110.0, 50% of position
# hold=5 → hold_limit_idx = 4 + 5 = 9 → time exit fires at i=9
#
# --- Tranche A (bar5, i=5): high=115 ≥ partial_target=110.0 ---
#   open=102.0 < 110.0 → no gap fill → fill at 110.0 (target_ref exactly)
#   close_shares = 1000.0 * 0.5 = 500.0
#   pro_rata_cost = 100_000.0 * 0.5 = 50_000.0
#   exit_value_A = 500.0 * 110.0 = 55_000.0  (commission=0)
#   pnl_A = 55_000.0 - 50_000.0 = 5_000.0
#   After partial: remaining 500 shares, remaining_cost = 50_000.0
#   stop_ref raised to entry_fill=100.0 (was None → no raise applies)
#
# --- Time exit (bar9, i=9): i >= hold_limit_idx=9 ---
#   remaining_shares=500, close=112.0
#   exit_value_B = 500.0 * 112.0 = 56_000.0
#   pnl_B = 56_000.0 - 50_000.0 = 6_000.0
#
# Total pnl = 5_000.0 + 6_000.0 = 11_000.0
# final_cash after both exits: starts at 100_000 → spend 100_000 entry
#   → receive 55_000 partial exit → receive 56_000 final exit
#   final_cash = 0 - 100_000 + 55_000 + 56_000 = 11_000
#   But portfolio starts at 100_000 cash; spends 100_000 on entry → cash=0
#   partial close: cash += 55_000 → cash = 55_000
#   final close: cash += 56_000 → cash = 111_000
#   final_equity = 111_000


def bars_s7_partial_then_time() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        # bar3 = signal/as_of bar; close > open triggers 'close > 0' always
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5 (i=5): partial target=110 hit; open=102<110, high=115≥110 → partial fill@110
        {"open": 102.0, "high": 115.0, "low": 101.5, "close": 112.0, "volume": 10_000},
        # bars 6-9: drifts; no further exit triggers except time at i=9
        {"open": 112.0, "high": 113.0, "low": 111.0, "close": 111.5, "volume": 10_000},
        {"open": 111.5, "high": 112.0, "low": 110.5, "close": 111.0, "volume": 10_000},
        {"open": 111.0, "high": 112.0, "low": 110.0, "close": 111.5, "volume": 10_000},
        # bar9 (i=9): time exit at close=112.0
        {"open": 111.5, "high": 113.0, "low": 111.0, "close": 112.0, "volume": 10_000},
        # bar10: extra bar so loop can reach i=9
        {"open": 112.0, "high": 114.0, "low": 111.5, "close": 113.0, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 8 — time-based exit (max-hold-days)
# ---------------------------------------------------------------------------
# signal_idx=3, entry at bar4 open=100.0
# hold=3 → hold_limit_idx = 4 + 3 = 7
# Loop (entry_idx+1 = 5): i=5 → 5>=7? No; i=6 → 6>=7? No; i=7 → 7>=7? Yes
# bar7: close=105.0 → time exit
#
# entry_fill = 100.0
# shares     = 1000.0
# entry_cost = 100_000.0
# exit_fill  = 105.0  (close of bar7)
# exit_value = 105_000.0
# pnl        = 5_000.0
# return_pct = 0.05


def bars_s8_time_exit() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.0
        {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 10_000},
        # bar5 (i=5): 5 < 7, no exit
        {"open": 100.2, "high": 101.0, "low": 99.5, "close": 101.5, "volume": 10_000},
        # bar6 (i=6): 6 < 7, no exit
        {"open": 101.5, "high": 102.0, "low": 100.8, "close": 103.0, "volume": 10_000},
        # bar7 (i=7): 7 >= 7 → time exit at close=105.0
        {"open": 103.0, "high": 106.0, "low": 102.5, "close": 105.0, "volume": 10_000},
        {"open": 105.0, "high": 107.0, "low": 104.5, "close": 106.0, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Scenario 9 — commission + slippage: verify shares/pnl against plan's numbers
# ---------------------------------------------------------------------------
# slippage_bps=10 (0.1%), commission_bps=0
# entry_ref = bar4 open = 100.5
# entry_fill = 100.5 * (1 + 10/10_000) = 100.5 * 1.001 = 100.6005
# Portfolio.open (commission_bps=0):
#   gross_per_share = entry_fill * (1 + 0) = 100.6005
#   shares = 100_000 / 100.6005 = 994.030844777...  (≈ 994.0308 as stated in plan)
#   entry_cost = shares * entry_fill = 100_000.0  (exactly, by construction)
#
# take_profit=0.185 → target_ref = 100.6005 * 1.185 = 119.21159...
# bar5: GAPS UP; open=119.4 ≥ target_ref=119.2116 → gap fill at open=119.4
#   exit_ref = 119.4, gap_fills=True → exit_fill = 119.4 * (1 - 10/10_000) = 119.4 * 0.999
#   exit_fill = 119.2806
#   exit_value = 994.030844... * 119.2806 = 118_568.5956...
#   pnl = 118_568.5956 - 100_000.0 = 18_568.5956  (matches plan's 18568.5956)
#
# EXACT computed values:
#   shares     = 100_000 / (100.5 * 1.001)    = 994.030844777...
#   entry_cost = 100_000.0                    (no commission)
#   exit_fill  = 119.4 * 0.999               = 119.2806
#   exit_value = shares * 119.2806           = 118568.5956...
#   pnl        = 18568.5956...
#   return_pct = pnl / entry_cost            = 0.185685956...


def bars_s9_commission_slippage() -> pd.DataFrame:
    rows = [
        {"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 10_000},
        {"open": 98.5, "high": 99.5, "low": 98.0, "close": 99.0, "volume": 10_000},
        {"open": 99.0, "high": 100.0, "low": 98.8, "close": 99.5, "volume": 10_000},
        {"open": 99.5, "high": 100.5, "low": 99.0, "close": 100.0, "volume": 10_000},
        # bar4: entry open=100.5 → entry_fill=100.5*1.001=100.6005
        {"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.8, "volume": 10_000},
        # bar5: open=119.4 ≥ target_ref=119.2116 (take_profit=0.185)
        #   gap_fills=True → exit_fill = 119.4 * 0.999 = 119.2806
        {"open": 119.4, "high": 120.0, "low": 119.0, "close": 119.8, "volume": 10_000},
        {"open": 119.8, "high": 120.5, "low": 119.5, "close": 120.0, "volume": 10_000},
    ]
    return _make_frame(rows)


# ---------------------------------------------------------------------------
# Single-ticker universe helpers for run_backtest
# ---------------------------------------------------------------------------


def make_spy_bars(n: int) -> pd.DataFrame:
    """Flat SPY bars used as benchmark placeholder."""
    idx = _bdate_range(n)
    rows = [
        {
            "open": 500.0,
            "high": 501.0,
            "low": 499.0,
            "close": 500.0,
            "volume": 1_000_000,
        }
        for _ in range(n)
    ]
    return pd.DataFrame(rows, index=idx).astype(float)
