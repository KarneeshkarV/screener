"""Unit tests for the combined momentum + low-volatility strategy.

The dataset is engineered so the blended winner (``BEST``) is NEITHER the pure
momentum winner (``HIMOM``) NOR the pure low-vol winner (``LOVOL``), proving the
cross-sectional blend genuinely combines both factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _series(growth: float, noise: float, volume: float) -> pd.DataFrame:
    drift = 50.0 * (1.0 + growth) ** np.arange(_N)
    wiggle = noise * 50.0 * np.sin(np.arange(_N))
    close = pd.Series(drift + wiggle, index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) + abs(noise) * 50.0 + 0.1
    low = pd.concat([openp, close], axis=1).min(axis=1) - abs(noise) * 50.0 - 0.1
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(volume, index=_INDEX, dtype=float),
        }
    )


# momentum order (growth): HIMOM > BEST > MEH > LOVOL
# low-vol  order (noise) : LOVOL < BEST < MEH < HIMOM  (less noise = lower vol)
# => BEST has the highest 0.5*mom_pct + 0.5*invvol_pct blend.
_DATA = {
    "HIMOM": _series(growth=0.0020, noise=0.030, volume=900_000.0),
    "BEST": _series(growth=0.0015, noise=0.006, volume=900_000.0),
    "MEH": _series(growth=0.0010, noise=0.015, volume=900_000.0),
    "LOVOL": _series(growth=0.0006, noise=0.002, volume=900_000.0),
    "SPY": _series(growth=0.0005, noise=0.004, volume=1_000_000.0),
}


def test_strategy_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("mom_lowvol_combo")
    assert spec is not None
    assert spec.entry == "mom_12_1 > 0 and vol_252 > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback() == 253


def test_combo_blends_both_factors() -> None:
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name="mom_lowvol_combo",
        entry_expr="mom_12_1 > 0 and vol_252 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("HIMOM", "BEST", "MEH", "LOVOL"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(_DATA),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"BEST"}, traded
