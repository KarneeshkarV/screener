"""Unit tests for the Jegadeesh-Titman 12-1 momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _trend(start: float, daily_growth: float, volume: float) -> pd.DataFrame:
    """A clean geometric trend so 12-1 momentum has a deterministic sign/size."""
    close = pd.Series(start * (1.0 + daily_growth) ** np.arange(_N), index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0] / (1.0 + daily_growth))
    high = pd.concat([openp, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([openp, close], axis=1).min(axis=1) * 0.999
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
    spec = registry.get_optional("momentum_12_1")
    assert spec is not None
    assert spec.entry == "mom_12_1 > 0"
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 252


def test_momentum_score_is_causal() -> None:
    close = pd.Series(np.linspace(100.0, 200.0, _N), index=_INDEX)
    mom = momentum_12_1_score(close)
    # Undefined until 252 prior closes exist.
    assert mom.iloc[:252].isna().all()
    t = 300
    expected = close.iloc[t - 21] / close.iloc[t - 252] - 1.0
    assert mom.iloc[t] == expected


def test_momentum_selects_high_momentum_over_liquidity() -> None:
    # WIN has the strongest trend but the LOWEST volume; a dollar-volume ranker
    # would skip it. The factor ranker must pick it. FLAT/LOSE have non-positive
    # momentum and are excluded by the entry gate.
    data = {
        "WIN": _trend(50.0, 0.0018, volume=5_000.0),
        "MID": _trend(50.0, 0.0008, volume=900_000.0),
        "FLAT": _trend(50.0, 0.0, volume=900_000.0),
        "LOSE": _trend(50.0, -0.0015, volume=900_000.0),
        "SPY": _trend(400.0, 0.0005, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name="momentum_12_1",
        entry_expr="mom_12_1 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("WIN", "MID", "FLAT", "LOSE"),
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
    assert traded == {"WIN"}, traded
    assert result.metrics["trade_count"] > 0
