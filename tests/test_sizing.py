"""Unit + integration coverage for rule-based per-entry position sizing.

Rules are exercised through ``entry_budget_for`` with a real ``Portfolio`` and
deterministic synthetic OHLCV frames, and the rolling engine is driven
end-to-end (via ``StubPriceFetcher``) to prove the budget reaches the fills.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.sizing import entry_budget_for
from tests.conftest import StubPriceFetcher

_START = "2024-01-01"
_N = 40
_INDEX = pd.bdate_range(_START, periods=_N)


def _constant_bars(close: float, spread: float = 1.0, n: int = _N) -> pd.DataFrame:
    """Flat OHLCV frame: constant close, high/low a fixed ``spread`` around it."""
    idx = pd.bdate_range(_START, periods=n)
    close_s = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s,
            "high": close_s + spread,
            "low": close_s - spread,
            "close": close_s,
            "volume": pd.Series(1_000_000.0, index=idx, dtype=float),
        }
    )


def _sizing_cfg(
    sizing_rule: str = "equal_slot",
    *,
    top: int = 1,
    initial_capital: float = 100_000.0,
    stop_loss: float | None = None,
    **sizing_kwargs: float,
) -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        benchmark="SPY",
        hold=5,
        top=top,
        strategy_name=None,
        entry_expr="close > 0",
        exit_expr=None,
        stop_loss=stop_loss,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=initial_capital,
        tickers=("AAA",),
        sizing_rule=sizing_rule,
        **sizing_kwargs,
    )


def test_equal_slot_returns_entry_budget_exactly():
    portfolio = Portfolio(100_000.0, 4)
    cfg = _sizing_cfg("equal_slot", top=4)
    bars = _constant_bars(100.0)
    assert entry_budget_for(cfg, portfolio, bars, 20) == portfolio.entry_budget()
    assert entry_budget_for(cfg, portfolio, bars, 20) == pytest.approx(25_000.0)


@pytest.mark.parametrize(
    ("current_equity", "expected_budget"),
    [(120_000.0, 30_000.0), (80_000.0, 20_000.0)],
)
def test_reinvested_equal_slot_tracks_current_equity(
    current_equity: float, expected_budget: float
):
    portfolio = Portfolio(100_000.0, 4)
    cfg = _sizing_cfg("reinvested_equal_slot", top=4)
    bars = _constant_bars(100.0)

    budget = entry_budget_for(cfg, portfolio, bars, 20, current_equity=current_equity)

    assert budget == pytest.approx(expected_budget)


def test_reinvested_open_can_grow_above_initial_slot_ceiling():
    portfolio = Portfolio(100_000.0, 4)

    portfolio.open(
        "AAA",
        _INDEX[0].date(),
        100.0,
        budget=30_000.0,
        allow_slot_growth=True,
    )

    position = portfolio.get_position("AAA")
    assert position is not None
    assert position.slot_capital == pytest.approx(30_000.0)
    assert portfolio.cash() == pytest.approx(70_000.0)


def test_fixed_fraction_sizes_by_equity():
    portfolio = Portfolio(100_000.0, 4)
    cfg = _sizing_cfg("fixed_fraction", top=4, sizing_position_pct=0.10)
    bars = _constant_bars(100.0)
    # equity * position_pct = 100_000 * 0.10 = 10_000, below the 25_000 slot.
    assert entry_budget_for(cfg, portfolio, bars, 20) == pytest.approx(10_000.0)


def test_fixed_fraction_clamps_at_slot_ceiling():
    portfolio = Portfolio(100_000.0, 4)
    cfg = _sizing_cfg("fixed_fraction", top=4, sizing_position_pct=0.50)
    bars = _constant_bars(100.0)
    # equity * 0.50 = 50_000 > slot ceiling 25_000 -> clamped down.
    assert entry_budget_for(cfg, portfolio, bars, 20) == pytest.approx(25_000.0)


def test_fixed_risk_math():
    portfolio = Portfolio(100_000.0, 4)
    cfg = _sizing_cfg("fixed_risk", top=4, stop_loss=0.08, sizing_risk_pct=0.01)
    bars = _constant_bars(100.0)
    # equity * risk_pct / stop_loss = 100_000 * 0.01 / 0.08 = 12_500.
    assert entry_budget_for(cfg, portfolio, bars, 20) == pytest.approx(12_500.0)


def test_atr_risk_with_constant_true_range():
    portfolio = Portfolio(100_000.0, 1)
    cfg = _sizing_cfg(
        "atr_risk",
        top=1,
        sizing_risk_pct=0.01,
        sizing_atr_window=14,
        sizing_atr_multiple=2.0,
    )
    # close constant 100, high=101, low=99 -> true range 2 -> ATR 2 once warm.
    # stop_fraction = mult * ATR / close = 2 * 2 / 100 = 0.04.
    # budget = equity * risk_pct / stop_fraction = 100_000 * 0.01 / 0.04 = 25_000.
    bars = _constant_bars(100.0)
    assert entry_budget_for(cfg, portfolio, bars, 20) == pytest.approx(25_000.0)


def test_atr_risk_warmup_falls_back_to_base():
    portfolio = Portfolio(100_000.0, 1)
    cfg = _sizing_cfg("atr_risk", top=1)
    bars = _constant_bars(100.0)
    # signal_idx 0 -> ATR undefined (min_periods=14) -> base budget.
    assert entry_budget_for(cfg, portfolio, bars, 0) == portfolio.entry_budget()
    assert entry_budget_for(cfg, portfolio, bars, 0) == pytest.approx(100_000.0)


def test_inverse_vol_zero_volatility_falls_back_to_base():
    portfolio = Portfolio(100_000.0, 1)
    cfg = _sizing_cfg("inverse_vol", top=1, sizing_vol_window=20)
    # Constant close -> zero return volatility -> nan -> base budget.
    bars = _constant_bars(100.0)
    assert entry_budget_for(cfg, portfolio, bars, 25) == portfolio.entry_budget()


def test_open_clamps_oversized_budget_to_entry_budget():
    portfolio = Portfolio(100_000.0, 4)
    portfolio.open("AAA", _INDEX[0].date(), 100.0, budget=1_000_000.0)
    pos = portfolio.get_position("AAA")
    assert pos is not None
    # Clamped to the 25_000 slot ceiling; zero commission -> shares = 25_000/100.
    assert pos.shares == pytest.approx(250.0)
    assert portfolio.cash() == pytest.approx(75_000.0)


def test_open_smaller_budget_spends_exactly_that_budget():
    portfolio = Portfolio(100_000.0, 4)
    portfolio.open("AAA", _INDEX[0].date(), 100.0, budget=5_000.0)
    pos = portfolio.get_position("AAA")
    assert pos is not None
    assert pos.shares == pytest.approx(50.0)
    assert portfolio.cash() == pytest.approx(95_000.0)


def test_portfolio_policy_rejects_unknown_sizing_rule():
    with pytest.raises(ValidationError):
        _sizing_cfg(sizing_rule="nonsense")


def test_config_rejects_fixed_risk_without_stop_loss():
    with pytest.raises(ValidationError):
        _sizing_cfg("fixed_risk", stop_loss=None)


# --- Integration: budget flows through the rolling engine to the fills. -------


def _ramp(start_px: float, end_px: float, volume: float) -> pd.DataFrame:
    close = pd.Series(np.linspace(start_px, end_px, _N), index=_INDEX, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=_INDEX, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# Two steadily rising tickers keep every slot funded (cash never erodes below a
# slot), so the equal-slot budget stays a flat 50_000 for the whole run.
_RISING_DATA = {
    "AAA": _ramp(100.0, 160.0, 500_000.0),
    "BBB": _ramp(100.0, 150.0, 300_000.0),
    "SPY": _ramp(400.0, 440.0, 1_000_000.0),
}


def _rolling_cfg(**overrides) -> BacktestConfig:
    base = dict(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=5,
        top=2,
        strategy_name=None,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB"),
    )
    base.update(overrides)
    return BacktestConfig(**base)


def test_rolling_fixed_fraction_spends_configured_budget():
    fetcher = StubPriceFetcher(_RISING_DATA)
    cfg = _rolling_cfg(sizing_rule="fixed_fraction", sizing_position_pct=0.05)
    result = run_rolling_backtest(
        cfg, fetcher, start_date=_INDEX[0].date(), end_date=_INDEX[-1].date()
    )
    assert result.trades
    for trade in result.trades:
        # equity * 0.05 = 5_000, fully spent at zero commission.
        assert trade.entry_cost == pytest.approx(5_000.0)


def test_rolling_default_matches_legacy_slot_sizing():
    fetcher = StubPriceFetcher(_RISING_DATA)
    cfg = _rolling_cfg()  # equal_slot default
    result = run_rolling_backtest(
        cfg, fetcher, start_date=_INDEX[0].date(), end_date=_INDEX[-1].date()
    )
    assert result.trades
    for trade in result.trades:
        # top=2 -> slot_capital = 100_000 / 2 = 50_000, fully spent.
        assert trade.entry_cost == pytest.approx(50_000.0)


def test_rolling_reinvested_equal_slot_compounds_later_entry_budgets():
    fetcher = StubPriceFetcher(_RISING_DATA)
    cfg = _rolling_cfg(sizing_rule="reinvested_equal_slot")

    result = run_rolling_backtest(
        cfg, fetcher, start_date=_INDEX[0].date(), end_date=_INDEX[-1].date()
    )

    entry_costs = [trade.entry_cost for trade in result.trades]
    assert entry_costs[:2] == pytest.approx([50_000.0, 50_000.0])
    assert max(entry_costs[2:]) > 50_000.0


def test_reinvested_budget_splits_cash_across_the_slots_being_refilled():
    """One refill batch must not let the first slot absorb the whole balance.

    Two of four slots are open and marked far above cost, so ``equity / top``
    exceeds the cash on hand. Sizing the first of the two free slots against
    total cash would leave the second with a zero budget (and a zero-share
    position parked in its slot).
    """
    cfg = _rolling_cfg(sizing_rule="reinvested_equal_slot", top=4)
    portfolio = Portfolio(100_000.0, 4)
    for ticker in ("AAA", "BBB"):
        portfolio.assign(ticker, 1, _INDEX[0].date().isoformat())
        portfolio.open(ticker, _INDEX[0], 100.0, budget=25_000.0)
    assert portfolio.cash() == pytest.approx(50_000.0)
    bars = _constant_bars(100.0)

    first = entry_budget_for(
        cfg, portfolio, bars, len(bars) - 1, current_equity=400_000.0, free_slots=2
    )
    assert first == pytest.approx(25_000.0)

    portfolio.assign("CCC", 1, _INDEX[0].date().isoformat())
    portfolio.open("CCC", _INDEX[0], 100.0, budget=first, allow_slot_growth=True)
    second = entry_budget_for(
        cfg, portfolio, bars, len(bars) - 1, current_equity=400_000.0, free_slots=1
    )
    assert second == pytest.approx(25_000.0)


def test_open_with_quoted_shares_cannot_overdraw_available_cash():
    """A slipped buy fill priced off the pre-slippage quote must not go negative.

    ``FillModel.entry_quote`` sizes shares against the unslipped reference, so
    the share count handed to ``Portfolio.open`` costs more than the budget it
    was quoted for once slippage is applied. The cash cap trims it.
    """
    portfolio = Portfolio(100_000.0, 4)
    quoted_shares = 25_000.0 / 100.0  # sized off a 100.0 reference
    portfolio.assign("AAA", 1, _INDEX[0].date().isoformat())
    position = portfolio.open(
        "AAA",
        _INDEX[0],
        110.0,  # slipped fill: 250 shares would cost 27_500
        budget=25_000.0,
        shares=quoted_shares,
    )

    assert position.slot_capital <= 25_000.0 + 1e-9
    assert portfolio.cash() >= 0.0
