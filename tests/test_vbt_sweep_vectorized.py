"""Regression tests for vectorized vbt parameter sweep (requires vectorbt extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")

from screener.backtester.vbt_sweep import (
    iter_param_combos,
    rank_results,
    run_combo_backtest,
    run_parameter_sweep,
    _require_vectorbt,
)

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "vbt_sweep_aapl_msft_nvda.csv"


def _synthetic_close_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    idx = pd.bdate_range("2019-01-01", periods=260)
    symbols = ["AAPL", "MSFT", "NVDA"]
    close = pd.DataFrame(
        {
            sym: 100.0 + i * 10 + np.cumsum(np.random.randn(len(idx)))
            for i, sym in enumerate(symbols)
        },
        index=idx,
    )
    return close, close * 0.998


def _synthetic_ohlcv_panel() -> dict[str, pd.DataFrame]:
    """OHLCV-like panel with high/low approximated as close ± noise and volume positive."""
    np.random.seed(7)
    close, open_ = _synthetic_close_panel()
    # high / low derived from close +/- daily noise (bounded so they bracket close).
    rng = np.random.default_rng(11)
    noise = rng.uniform(0.5, 2.0, size=close.shape)
    high = close + noise
    low = close - noise
    # Volume: positive, with some variation.
    volume = pd.DataFrame(
        rng.uniform(1e6, 5e6, size=close.shape),
        index=close.index,
        columns=close.columns,
    )
    return {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
    }


def test_run_parameter_sweep_matches_frozen_csv() -> None:
    close, open_ = _synthetic_close_panel()
    actual = run_parameter_sweep(
        close,
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[0],
        open_=open_,
    )
    expected = pd.read_csv(FIXTURE_CSV)
    metric_cols = [
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
    ]
    for col in metric_cols:
        np.testing.assert_allclose(
            actual[col].to_numpy(),
            expected[col].to_numpy(),
            rtol=0,
            atol=1e-9,
        )
    pd.testing.assert_frame_equal(
        actual[["fast", "slow", "hold", "trades"]].astype(int).reset_index(drop=True),
        expected[["fast", "slow", "hold", "trades"]].astype(int).reset_index(drop=True),
    )


def test_run_parameter_sweep_matches_per_combo_loop() -> None:
    close, open_ = _synthetic_close_panel()
    fast_values, slow_values, hold_values = [10, 20], [50, 100], [0, 10]
    vbt = _require_vectorbt()
    loop_df = pd.DataFrame(
        [
            run_combo_backtest(
                close, fast, slow, hold, vbt=vbt, open_=open_, initial_capital=100_000
            )
            for fast, slow, hold in iter_param_combos(
                fast_values, slow_values, hold_values
            )
        ]
    )
    vec_df = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        open_=open_,
    )
    metric_cols = [
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    ]
    np.testing.assert_allclose(
        loop_df[metric_cols].to_numpy(),
        vec_df[metric_cols].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def test_rank_results_unchanged_on_vectorized_output() -> None:
    close, open_ = _synthetic_close_panel()
    results = run_parameter_sweep(
        close,
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[0],
        open_=open_,
    )
    ranked = rank_results(results, "sharpe")
    assert ranked.iloc[0]["fast"] == 20
    assert ranked.iloc[0]["slow"] == 100


def _assert_schema_ok(df: pd.DataFrame) -> None:
    expected_cols = {
        "indicator",
        "fast",
        "slow",
        "hold",
        "sharpe",
        "total_return",
        "calmar",
        "max_drawdown",
        "win_rate",
        "trades",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"missing columns: {expected_cols - set(df.columns)}"
    )
    assert not df["sharpe"].isna().all()


def test_ema_differs_from_sma_same_grid() -> None:
    close, open_ = _synthetic_close_panel()
    fast_values, slow_values, hold_values = [10, 20], [50, 100], [0]
    sma = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        indicators=["sma"],
        open_=open_,
    )
    ema = run_parameter_sweep(
        close,
        fast_values=fast_values,
        slow_values=slow_values,
        hold_values=hold_values,
        indicators=["ema"],
        open_=open_,
    )
    _assert_schema_ok(ema)
    assert (ema["indicator"] == "ema").all()
    # EMA result should not be numerically identical to SMA on the same grid.
    assert not np.allclose(
        sma["sharpe"].to_numpy(), ema["sharpe"].to_numpy(), equal_nan=True
    )


def test_breakout_schema_and_trades() -> None:
    close, open_ = _synthetic_close_panel()
    df = run_parameter_sweep(
        close,
        fast_values=[10],
        slow_values=[50],
        hold_values=[5, 10],
        indicators=["breakout"],
        breakout_windows=[10, 20, 50],
        open_=open_,
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "breakout").all()
    assert df["slow"].isna().all(), "breakout should have NaN slow"
    assert int(df["trades"].sum()) > 0, "expected at least one breakout trade"


def test_bbands_schema_and_signal_activity() -> None:
    close, open_ = _synthetic_close_panel()
    df = run_parameter_sweep(
        close,
        fast_values=[10],
        slow_values=[50],
        hold_values=[5],
        indicators=["bbands"],
        bbands_windows=[20, 50],
        open_=open_,
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "bbands").all()
    assert df["slow"].isna().all()


def test_macd_schema() -> None:
    close, open_ = _synthetic_close_panel()
    df = run_parameter_sweep(
        close,
        fast_values=[10],
        slow_values=[50],
        hold_values=[5, 10],
        indicators=["macd"],
        open_=open_,
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "macd").all()
    # MACD reports the configured fast/slow defaults.
    assert (df["fast"] == 12).all()
    assert (df["slow"] == 26).all()


def test_rsi_schema() -> None:
    close, open_ = _synthetic_close_panel()
    df = run_parameter_sweep(
        close,
        fast_values=[10],
        slow_values=[50],
        hold_values=[5],
        indicators=["rsi"],
        rsi_thresholds=[50, 60],
        open_=open_,
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "rsi").all()
    assert set(df["fast"].astype(int)) == {50, 60}


def test_supertrend_schema_with_high_low() -> None:
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10],
        slow_values=[50],
        hold_values=[5],
        indicators=["supertrend"],
        supertrend_periods=[7, 10],
        high=panels["high"],
        low=panels["low"],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "supertrend").all()
    assert df["slow"].isna().all()


def test_keltner_schema_with_high_low() -> None:
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10],
        slow_values=[50],
        hold_values=[5],
        indicators=["keltner"],
        keltner_windows=[20, 50],
        high=panels["high"],
        low=panels["low"],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "keltner").all()


def test_vol_breakout_schema_with_volume() -> None:
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10],
        slow_values=[50],
        hold_values=[5, 10],
        indicators=["vol_breakout"],
        breakout_windows=[20, 50],
        volume=panels["volume"],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "vol_breakout").all()


def test_obv_trend_schema_with_volume() -> None:
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10],
        slow_values=[50],
        hold_values=[5],
        indicators=["obv_trend"],
        obv_ema_windows=[20, 50],
        volume=panels["volume"],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    assert (df["indicator"] == "obv_trend").all()


def test_combo_indicators_run() -> None:
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[5],
        indicators=["sma_rsi", "breakout_rsi"],
        breakout_windows=[20, 50],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    assert set(df["indicator"].unique()) == {"sma_rsi", "breakout_rsi"}


def test_multi_indicator_single_call() -> None:
    """All indicators in one Portfolio.from_signals call (the hot path)."""
    panels = _synthetic_ohlcv_panel()
    df = run_parameter_sweep(
        panels["close"],
        fast_values=[10, 20],
        slow_values=[50, 100],
        hold_values=[5],
        indicators=[
            "sma",
            "ema",
            "breakout",
            "bbands",
            "macd",
            "rsi",
            "supertrend",
            "keltner",
            "vol_breakout",
            "obv_trend",
            "sma_rsi",
            "breakout_rsi",
        ],
        breakout_windows=[20, 50],
        bbands_windows=[20],
        supertrend_periods=[10],
        keltner_windows=[20],
        rsi_thresholds=[50],
        obv_ema_windows=[20],
        high=panels["high"],
        low=panels["low"],
        volume=panels["volume"],
        open_=panels["open"],
    )
    _assert_schema_ok(df)
    # Every requested indicator should appear at least once in the output.
    assert set(df["indicator"].unique()) == {
        "sma",
        "ema",
        "breakout",
        "bbands",
        "macd",
        "rsi",
        "supertrend",
        "keltner",
        "vol_breakout",
        "obv_trend",
        "sma_rsi",
        "breakout_rsi",
    }
