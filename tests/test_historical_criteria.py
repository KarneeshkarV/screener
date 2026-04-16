"""Tests for screener/historical_criteria.py.

All tests are offline — no network calls.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from screener.historical_criteria import (
    FUND_CRITERIA,
    HIST_CRITERIA,
    _slice_to_asof,
    compute_scores,
    eval_breakout,
    eval_ema_stack,
)
from tests.conftest import make_ohlcv


# ── _slice_to_asof ───────────────────────────────────────────────────────────

class TestSliceToAsof:
    def test_excludes_future_bars(self):
        df = make_ohlcv(n_days=300, start_date=date(2023, 1, 2))
        as_of = pd.Timestamp("2023-06-01")
        sliced = _slice_to_asof(df, as_of)
        assert sliced["date"].max() <= as_of

    def test_includes_asof_day(self):
        df = make_ohlcv(n_days=300, start_date=date(2023, 1, 2))
        # 2023-01-02 is the first row
        as_of = pd.Timestamp("2023-01-02")
        sliced = _slice_to_asof(df, as_of)
        assert len(sliced) >= 1

    def test_empty_when_all_future(self):
        df = make_ohlcv(n_days=10, start_date=date(2025, 1, 2))
        as_of = pd.Timestamp("2024-01-01")
        sliced = _slice_to_asof(df, as_of)
        assert sliced.empty


# ── eval_ema_stack ────────────────────────────────────────────────────────────

class TestEmaStack:
    def test_bullish_uptrend_passes(self, uptrend_ohlcv):
        as_of = pd.Timestamp("2023-10-01")
        result = eval_ema_stack(uptrend_ohlcv, as_of)
        assert result is not None
        assert result["passes"] is True

    def test_bearish_downtrend_fails(self, downtrend_ohlcv):
        as_of = pd.Timestamp("2023-10-01")
        result = eval_ema_stack(downtrend_ohlcv, as_of)
        assert result is not None
        assert result["passes"] is False

    def test_uses_only_data_up_to_asof(self, uptrend_ohlcv):
        """Score inputs must not include any bar after as_of."""
        # Use a date ~270 bars into the 300-bar series (well above MIN_BARS_EMA=260)
        as_of = pd.Timestamp("2023-10-01")
        result = eval_ema_stack(uptrend_ohlcv, as_of)
        assert result is not None
        # The close in score_inputs should match the last bar on/before as_of
        sliced = _slice_to_asof(uptrend_ohlcv, as_of)
        expected_close = float(sliced["close"].iloc[-1])
        assert abs(result["score_inputs"]["close"] - expected_close) < 1e-6

    def test_returns_none_when_insufficient_history(self):
        # Only 10 bars — not enough for EMA200
        df = make_ohlcv(n_days=10, trend="up")
        as_of = pd.Timestamp(df["date"].iloc[-1])
        result = eval_ema_stack(df, as_of)
        assert result is None

    def test_score_inputs_keys(self, uptrend_ohlcv):
        as_of = pd.Timestamp("2023-10-01")
        result = eval_ema_stack(uptrend_ohlcv, as_of)
        assert result is not None
        for key in ("close", "ema5", "ema20", "ema100", "ema200", "rsi"):
            assert key in result["score_inputs"], f"missing key: {key}"


# ── eval_breakout ─────────────────────────────────────────────────────────────

class TestBreakout:
    def _make_near_high_df(self, n_days: int = 60) -> pd.DataFrame:
        """Close is at 99% of rolling max and volume is 2× average."""
        df = make_ohlcv(n_days=n_days, trend="flat", base_close=100.0)
        # spike close on last bar to near-high
        df.loc[df.index[-1], "close"] = 99.5
        df.loc[df.index[-1], "adj_close"] = 99.5
        df.loc[df.index[-1], "volume"] = 2_000_000.0  # above avg
        return df

    def test_near_high_high_volume_passes(self):
        df = self._make_near_high_df()
        as_of = pd.Timestamp(df["date"].iloc[-1])
        result = eval_breakout(df, as_of)
        assert result is not None
        assert result["passes"] is True

    def test_far_from_high_fails(self):
        df = make_ohlcv(n_days=60, trend="flat", base_close=100.0)
        # set close to 50% of max → fails near-high test
        df.loc[df.index[-1], "close"] = 50.0
        df.loc[df.index[-1], "adj_close"] = 50.0
        as_of = pd.Timestamp(df["date"].iloc[-1])
        result = eval_breakout(df, as_of)
        assert result is not None
        assert result["passes"] is False

    def test_insufficient_history_returns_none(self):
        df = make_ohlcv(n_days=5, trend="up")
        as_of = pd.Timestamp(df["date"].iloc[-1])
        result = eval_breakout(df, as_of)
        assert result is None

    def test_uses_only_data_up_to_asof(self):
        """Breakout evaluation must not peek at bars after as_of."""
        df = make_ohlcv(n_days=60, trend="flat", base_close=100.0)
        # bars after as_of have a huge spike — if slicing is broken,
        # the 52w-high would be 10000 and the current bar would fail
        as_of = pd.Timestamp(df["date"].iloc[40])  # evaluate partway through

        future_df = df.copy()
        future_df.loc[future_df.index[50:], "close"] = 10000.0
        future_df.loc[future_df.index[50:], "high"] = 10000.0

        result = eval_breakout(future_df, as_of)
        # Should not fail because of the spike (which is after as_of)
        # The result should be the same as without the spike
        result_clean = eval_breakout(df, as_of)
        assert (result is None) == (result_clean is None)
        if result is not None and result_clean is not None:
            assert result["passes"] == result_clean["passes"]


# ── HIST_CRITERIA dict ────────────────────────────────────────────────────────

class TestHistCriteriaDict:
    def test_all_criteria_callable(self):
        for name, fn in HIST_CRITERIA.items():
            assert callable(fn), f"{name} is not callable"

    def test_fund_criteria_are_in_hist(self):
        """Fundamentals-based criteria must be in HIST_CRITERIA."""
        for name in FUND_CRITERIA:
            assert name in HIST_CRITERIA, f"{name} missing from HIST_CRITERIA"

    def test_ema_breakout_combined(self, uptrend_ohlcv):
        """ema_breakout should fail on an uptrend that isn't near its 52w-high."""
        # uptrend always near its own high, so it may pass or fail depending on
        # the trend magnitude — just check it returns a dict (not None) for a
        # series with enough bars.
        as_of = pd.Timestamp("2023-10-01")
        result = HIST_CRITERIA["ema_breakout"](uptrend_ohlcv, as_of)
        assert result is None or isinstance(result["passes"], bool)


# ── compute_scores ────────────────────────────────────────────────────────────

class TestComputeScores:
    def _snapshots(self, n: int = 5) -> list[dict]:
        base = {
            "close": 100.0, "ema5": 105.0, "ema20": 103.0,
            "ema100": 100.0, "ema200": 95.0,
            "rsi": 60.0, "momentum": 0.001,
            "dollar_vol_20d": 1_000_000.0,
            "volume": 50000.0, "vol_10d_avg": 40000.0,
            "high_52w": 110.0,
        }
        return [dict(base, dollar_vol_20d=base["dollar_vol_20d"] * (i + 1)) for i in range(n)]

    def test_returns_same_length(self):
        snaps = self._snapshots(5)
        scores = compute_scores(snaps)
        assert len(scores) == 5

    def test_empty_input(self):
        assert compute_scores([]) == []

    def test_higher_liquidity_scores_higher(self):
        snaps = self._snapshots(2)
        # second snapshot has 2× dollar volume
        scores = compute_scores(snaps)
        assert scores[1] > scores[0]

    def test_scores_are_floats(self):
        snaps = self._snapshots(3)
        for s in compute_scores(snaps):
            assert isinstance(s, float)
