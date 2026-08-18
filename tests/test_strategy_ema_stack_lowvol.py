"""Unit tests for the default EMA stack and its low-downside-volatility variant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.strategies.plugins.ema_stack_lowvol import (
    ENTRY_STACK,
    downside_volatility,
)
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _noisy_trend(start: float, daily_growth: float, noise: float, volume: float):
    """Deterministic drift plus a fixed sawtooth wiggle controlling volatility.

    The drift keeps the EMA stack in bullish order for every name, so the only
    thing separating them in the tests below is how violently they wiggle.
    """
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


def test_both_strategies_registered() -> None:
    discover_plugins()
    base = registry.get_optional("ema_stack")
    filtered = registry.get_optional("ema_stack_lowvol")
    assert base is not None and filtered is not None
    # The filter must not change the entry gate: it only orders the candidates.
    assert base.entry == ENTRY_STACK
    assert filtered.entry == ENTRY_STACK
    assert base.prepare_bars is None
    assert filtered.prepare_bars is not None
    assert base.required_lookback() == 200
    assert filtered.required_lookback() == 200


def test_entry_is_the_default_screen_criterion() -> None:
    # The live `ema` criterion is EMA5 > EMA20 > EMA100 > EMA200 with EMA200
    # positive. Pin the translation so a future edit cannot quietly drift from
    # the screen this strategy claims to represent.
    assert ENTRY_STACK == (
        "ema(close, 5) > ema(close, 20) "
        "and ema(close, 20) > ema(close, 100) "
        "and ema(close, 100) > ema(close, 200) "
        "and ema(close, 200) > 0"
    )


def test_downside_volatility_ignores_upside_moves() -> None:
    # A series that only ever rises has no downside deviation at all, which is
    # the whole point of using semi-deviation instead of realized volatility.
    rising = pd.Series(100.0 * (1.01 ** np.arange(_N)), index=_INDEX)
    vol = downside_volatility(rising)
    assert vol.iloc[-1] == pytest.approx(0.0)


def test_downside_volatility_is_causal_and_matches_its_definition() -> None:
    close = pd.Series(100.0 + np.sin(np.arange(_N)), index=_INDEX)
    vol = downside_volatility(close)
    # 60-bar window on a diffed series: undefined until 60 returns exist.
    assert vol.iloc[:60].isna().all()
    assert vol.iloc[60:].notna().all()

    t = 300
    log_returns = np.log(close).diff()
    window = log_returns.iloc[t - 59 : t + 1]
    losses = window.where(window < 0.0, 0.0)
    expected = np.sqrt((losses**2).mean()) * np.sqrt(252.0)
    assert vol.iloc[t] == pytest.approx(expected)


def test_downside_volatility_is_truncation_invariant() -> None:
    # The value at bar t must not depend on bars after t.
    close = pd.Series(100.0 + np.sin(np.arange(_N)), index=_INDEX)
    full = downside_volatility(close)
    for probe in (120, 200, 300):
        truncated = downside_volatility(close.iloc[: probe + 1])
        assert truncated.iloc[probe] == pytest.approx(full.iloc[probe])


def _cfg(strategy: str, **overrides) -> BacktestConfig:
    base = dict(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name=strategy,
        entry_expr=ENTRY_STACK,
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
    base.update(overrides)
    return BacktestConfig(**base)


_DATA = {
    # CALM wiggles least but trades the least, so a dollar-volume ranker would
    # pass it over. All three hold the bullish EMA stack.
    "CALM": _noisy_trend(50.0, 0.0008, noise=0.002, volume=5_000.0),
    "CHOPPY": _noisy_trend(50.0, 0.0008, noise=0.05, volume=900_000.0),
    "WILD": _noisy_trend(50.0, 0.0008, noise=0.10, volume=900_000.0),
    "SPY": _noisy_trend(400.0, 0.0005, noise=0.005, volume=1_000_000.0),
}


def test_lowvol_variant_picks_the_calmest_name_in_the_stack() -> None:
    result = run_rolling_backtest(
        _cfg("ema_stack_lowvol"),
        StubPriceFetcher(_DATA),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded == {"CALM"}, traded


def test_bare_stack_is_not_volatility_selective() -> None:
    # Without the volatility leg there is no rank_score, so the engine ranks by
    # dollar volume and cycles through whatever holds the stack - including the
    # violent names. That is what the filter is measured against.
    result = run_rolling_backtest(
        _cfg("ema_stack"),
        StubPriceFetcher(_DATA),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    traded = {t.ticker for t in result.trades}
    assert traded & {"CHOPPY", "WILD"}, traded

    filtered = run_rolling_backtest(
        _cfg("ema_stack_lowvol"),
        StubPriceFetcher(_DATA),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    # Same entry gate, strictly narrower selection: the filter only reorders.
    filtered_traded = {t.ticker for t in filtered.trades}
    assert filtered_traded < traded, (filtered_traded, traded)


def test_the_filter_does_not_admit_names_the_stack_rejects() -> None:
    # A name in a downtrend fails the stack. It must stay out of the filtered
    # variant however calm it is, because the volatility leg only reorders.
    data = dict(_DATA)
    data["FALLING"] = _noisy_trend(50.0, -0.0015, noise=0.0005, volume=900_000.0)
    result = run_rolling_backtest(
        _cfg("ema_stack_lowvol", tickers=("CALM", "CHOPPY", "WILD", "FALLING"), top=3),
        StubPriceFetcher(data),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    assert "FALLING" not in {t.ticker for t in result.trades}
