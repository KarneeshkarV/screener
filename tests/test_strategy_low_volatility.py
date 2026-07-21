"""Unit tests for the Ang-Hodrick-Xing-Zhang low-volatility strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _noisy_trend(start: float, daily_growth: float, noise: float, volume: float):
    """Deterministic drift + a fixed sawtooth wiggle controlling realized vol."""
    drift = start * (1.0 + daily_growth) ** np.arange(_N)
    wiggle = noise * start * np.sin(np.arange(_N))
    close = pd.Series(drift + wiggle, index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) + abs(noise) * start + 0.1
    low = pd.concat([openp, close], axis=1).min(axis=1) - abs(noise) * start - 0.1
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(volume, index=_INDEX, dtype=float),
        }
    )


def test_strategy_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("low_volatility")
    assert spec is not None
    assert spec.entry == "vol_252 > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback() == 253


def test_realized_vol_is_causal_and_ordered() -> None:
    close = pd.Series(100.0 + np.sin(np.arange(_N)), index=_INDEX)
    vol = realized_volatility(close)
    assert vol.iloc[:252].isna().all()
    t = 300
    expected = close.pct_change().iloc[t - 251 : t + 1].std()
    assert vol.iloc[t] == pytest.approx(expected)


def test_low_vol_selects_calmest_over_liquidity() -> None:
    # CALM has the smallest wiggle (lowest realized vol) but the LOWEST volume;
    # a dollar-volume ranker would skip it. The factor ranker must pick it.
    data = {
        "CALM": _noisy_trend(50.0, 0.0008, noise=0.002, volume=5_000.0),
        "CHOPPY": _noisy_trend(50.0, 0.0008, noise=0.05, volume=900_000.0),
        "WILD": _noisy_trend(50.0, 0.0008, noise=0.10, volume=900_000.0),
        "SPY": _noisy_trend(400.0, 0.0005, noise=0.005, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name="low_volatility",
        entry_expr="vol_252 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("CALM", "CHOPPY", "WILD"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"CALM"}, traded
