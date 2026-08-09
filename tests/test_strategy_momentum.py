"""Unit tests for the Jegadeesh-Titman 12-1 momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.momentum_12_1 import (
    ENTRY_DEFENSIVE,
    ENTRY_PURE,
    ENTRY_RISKADJ,
    ENTRY_TREND,
    momentum_12_1_score,
    risk_adjusted_momentum,
)
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _ohlcv(close: pd.Series, volume: float) -> pd.DataFrame:
    openp = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([openp, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([openp, close], axis=1).min(axis=1) * 0.999
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(volume, index=close.index, dtype=float),
        }
    )


def _trend(start: float, daily_growth: float, volume: float) -> pd.DataFrame:
    """A clean geometric trend so 12-1 momentum has a deterministic sign/size."""
    close = pd.Series(start * (1.0 + daily_growth) ** np.arange(_N), index=_INDEX)
    return _ohlcv(close, volume)


def _trend_then_crash(
    start: float,
    daily_growth: float,
    crash_start: int,
    crash_end_price: float,
    volume: float,
) -> pd.DataFrame:
    """Strong trend that collapses late so price sits below SMA200 while 12-1
    momentum can still be positive (formation ends before the crash leg)."""
    close = start * (1.0 + daily_growth) ** np.arange(_N)
    pre = close[crash_start - 1]
    crash_len = _N - crash_start
    # Linear drop to crash_end_price over the crash window.
    close[crash_start:] = np.linspace(pre, crash_end_price, crash_len + 1)[1:]
    return _ohlcv(pd.Series(close, index=_INDEX), volume)


def _rolling_cfg(
    strategy_name: str,
    entry_expr: str,
    tickers: tuple[str, ...],
) -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name=strategy_name,
        entry_expr=entry_expr,
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=tickers,
        min_price=None,
        min_avg_dollar_volume=None,
    )


def test_strategy_registered() -> None:
    discover_plugins()
    pure = registry.get_optional("momentum_12_1")
    assert pure is not None
    assert pure.entry == ENTRY_PURE
    assert pure.prepare_bars is not None
    assert pure.required_lookback is not None
    assert pure.required_lookback() == 252

    trend = registry.get_optional("momentum_12_1_trend")
    assert trend is not None
    assert trend.entry == ENTRY_TREND
    assert "sma(close, 200)" in trend.entry
    assert trend.prepare_bars is pure.prepare_bars
    assert trend.required_lookback is not None
    assert trend.required_lookback() == 252

    riskadj = registry.get_optional("momentum_12_1_riskadj")
    assert riskadj is not None
    assert riskadj.entry == ENTRY_RISKADJ
    assert riskadj.prepare_bars is not None
    assert riskadj.required_lookback is not None
    assert riskadj.required_lookback() == 253

    defensive = registry.get_optional("momentum_12_1_defensive")
    assert defensive is not None
    assert defensive.entry == ENTRY_DEFENSIVE
    assert defensive.prepare_bars is not None
    assert defensive.required_lookback is not None
    assert defensive.required_lookback() == 252


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
    result = run_rolling_backtest(
        _rolling_cfg("momentum_12_1", ENTRY_PURE, ("WIN", "MID", "FLAT", "LOSE")),
        StubPriceFetcher(data),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"WIN"}, traded
    assert result.metrics["trade_count"] > 0


def test_trend_filter_skips_crashed_winner() -> None:
    """Highest 12-1 name can still have positive mom after a crash, but sits
    below SMA200 — pure momentum buys it; dual-momentum picks the intact trend.
    """
    # CRASH: steeper pre-crash growth than STEADY so 12-1 stays larger through
    # ~t=309, then collapses under its 200-day average. STEADY never breaks SMA.
    crash = _trend_then_crash(
        start=50.0,
        daily_growth=0.0025,
        crash_start=270,
        crash_end_price=35.0,
        volume=900_000.0,
    )
    steady = _trend(50.0, 0.0010, volume=900_000.0)
    spy = _trend(400.0, 0.0005, volume=1_000_000.0)

    # Window where CRASH still leads on mom_12_1 but is already below SMA200.
    t = 300
    crash_close = crash["close"]
    steady_close = steady["close"]
    crash_mom = momentum_12_1_score(crash_close).iloc[t]
    steady_mom = momentum_12_1_score(steady_close).iloc[t]
    crash_sma = crash_close.iloc[t - 199 : t + 1].mean()
    steady_sma = steady_close.iloc[t - 199 : t + 1].mean()
    assert crash_mom > steady_mom > 0
    assert crash_close.iloc[t] < crash_sma
    assert steady_close.iloc[t] > steady_sma

    data = {"CRASH": crash, "STEADY": steady, "SPY": spy}
    tickers = ("CRASH", "STEADY")
    # End before CRASH's 12-1 score falls below STEADY (~t=310) so pure ranking
    # is unambiguous for the whole span.
    start = _INDEX[290].date()
    end = _INDEX[309].date()

    pure = run_rolling_backtest(
        _rolling_cfg("momentum_12_1", ENTRY_PURE, tickers),
        StubPriceFetcher(data),
        start_date=start,
        end_date=end,
    )
    pure_traded = {tr.ticker for tr in pure.trades}
    assert pure_traded == {"CRASH"}, pure_traded

    filtered = run_rolling_backtest(
        _rolling_cfg("momentum_12_1_trend", ENTRY_TREND, tickers),
        StubPriceFetcher(data),
        start_date=start,
        end_date=end,
    )
    filtered_traded = {tr.ticker for tr in filtered.trades}
    assert filtered_traded == {"STEADY"}, filtered_traded
    assert filtered.metrics["trade_count"] > 0


def test_defensive_momentum_blocks_risk_off_benchmark() -> None:
    """Positive stock momentum alone cannot enter while the benchmark is risk-off."""
    winner = _trend(50.0, 0.0018, volume=900_000.0)
    runner_up = _trend(50.0, 0.0010, volume=900_000.0)
    tickers = ("WIN", "MID")
    start = _INDEX[260].date()
    end = _INDEX[-1].date()

    risk_off_data = {
        "WIN": winner,
        "MID": runner_up,
        "SPY": _trend(400.0, -0.0010, volume=1_000_000.0),
    }
    blocked = run_rolling_backtest(
        _rolling_cfg("momentum_12_1_defensive", ENTRY_DEFENSIVE, tickers),
        StubPriceFetcher(risk_off_data),
        start_date=start,
        end_date=end,
    )
    assert blocked.trades == []

    risk_on_data = {
        "WIN": winner,
        "MID": runner_up,
        "SPY": _trend(400.0, 0.0005, volume=1_000_000.0),
    }
    allowed = run_rolling_backtest(
        _rolling_cfg("momentum_12_1_defensive", ENTRY_DEFENSIVE, tickers),
        StubPriceFetcher(risk_on_data),
        start_date=start,
        end_date=end,
    )
    assert {trade.ticker for trade in allowed.trades} == {"WIN"}

    missing_benchmark = run_rolling_backtest(
        _rolling_cfg("momentum_12_1_defensive", ENTRY_DEFENSIVE, tickers),
        StubPriceFetcher({"WIN": winner, "MID": runner_up}),
        start_date=start,
        end_date=end,
    )
    assert missing_benchmark.trades == []


def test_riskadj_score_penalizes_volatility() -> None:
    """Same drift, higher noise => lower mom/vol rank_score."""
    calm = _series_with_noise(growth=0.0015, noise=0.002)
    wild = _series_with_noise(growth=0.0015, noise=0.040)
    t = 300
    _, _, calm_score = risk_adjusted_momentum(calm)
    _, _, wild_score = risk_adjusted_momentum(wild)
    calm_mom = momentum_12_1_score(calm).iloc[t]
    wild_mom = momentum_12_1_score(wild).iloc[t]
    # Formation returns are similar; risk-adj must still prefer calm.
    assert calm_mom > 0 and wild_mom > 0
    assert calm_score.iloc[t] > wild_score.iloc[t]


def _series_with_noise(growth: float, noise: float) -> pd.Series:
    drift = 50.0 * (1.0 + growth) ** np.arange(_N)
    wiggle = noise * 50.0 * np.sin(np.arange(_N))
    return pd.Series(drift + wiggle, index=_INDEX)


def test_riskadj_prefers_calm_winner_over_wild_winner() -> None:
    """Raw mom would pick the wild high-momentum name; risk-adj picks calm."""
    # WILD: steeper growth + huge noise → high raw mom, mediocre mom/vol.
    # CALM: solid growth + tiny noise → slightly lower mom, much higher mom/vol.
    wild = _ohlcv(_series_with_noise(growth=0.0022, noise=0.045), volume=900_000.0)
    calm = _ohlcv(_series_with_noise(growth=0.0014, noise=0.002), volume=900_000.0)
    flat = _trend(50.0, 0.0, volume=900_000.0)
    spy = _trend(400.0, 0.0005, volume=1_000_000.0)

    t = 300
    wild_mom = momentum_12_1_score(wild["close"]).iloc[t]
    calm_mom = momentum_12_1_score(calm["close"]).iloc[t]
    _, _, wild_ra = risk_adjusted_momentum(wild["close"])
    _, _, calm_ra = risk_adjusted_momentum(calm["close"])
    assert wild_mom > calm_mom > 0
    assert calm_ra.iloc[t] > wild_ra.iloc[t]

    data = {"WILD": wild, "CALM": calm, "FLAT": flat, "SPY": spy}
    tickers = ("WILD", "CALM", "FLAT")
    start = _INDEX[260].date()
    end = _INDEX[-1].date()

    pure = run_rolling_backtest(
        _rolling_cfg("momentum_12_1", ENTRY_PURE, tickers),
        StubPriceFetcher(data),
        start_date=start,
        end_date=end,
    )
    assert {tr.ticker for tr in pure.trades} == {"WILD"}

    riskadj = run_rolling_backtest(
        _rolling_cfg("momentum_12_1_riskadj", ENTRY_RISKADJ, tickers),
        StubPriceFetcher(data),
        start_date=start,
        end_date=end,
    )
    assert {tr.ticker for tr in riskadj.trades} == {"CALM"}
    assert riskadj.metrics["trade_count"] > 0
