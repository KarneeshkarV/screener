"""Public-engine regressions for causal fills, exit ordering and cash caps."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester import run_backtest, run_rolling_backtest
from screener.backtester.costs import VestedUSCosts
from screener.backtester.models import BacktestConfig
from screener.backtester.slippage import VolumeImpactSlippage
from screener.backtester.portfolio import Portfolio
from tests.conftest import StubPriceFetcher


def bars(n=8, *, unit="us", price=100.0, volume=10000.0):
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": volume,
        },
        index=pd.bdate_range("2024-01-01", periods=n).as_unit(unit),
    )


def config(**overrides):
    values = dict(
        market="us",
        as_of=date(2024, 1, 1),
        benchmark="SPY",
        tickers=("AAA",),
        entry_expr="close > 0",
        exit_expr=None,
        hold=5,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10000.0,
        min_price=0,
        min_avg_dollar_volume=0,
        price_adjustment="none",
    )
    values.update(overrides)
    return BacktestConfig(**values)


def run_engine(engine, frames, **overrides):
    cfg = config(tickers=tuple(frames), **overrides)
    benchmark = bars(max(len(frame) for frame in frames.values()))
    symbols = {
        (
            ticker.removeprefix("NSE:") + ".NS" if cfg.market == "india" else ticker
        ): frame
        for ticker, frame in frames.items()
    }
    fetcher = StubPriceFetcher({**symbols, "SPY": benchmark})
    if engine == "historical":
        return run_backtest(cfg, fetcher)
    return run_rolling_backtest(
        cfg, fetcher, start_date=cfg.as_of, end_date=benchmark.index[-1].date()
    )


@pytest.fixture(params=["historical", "rolling"])
def engine(request):
    return request.param


def first_lot(result):
    return [t for t in result.trades if t.signal_date == date(2024, 1, 1)]


@pytest.mark.parametrize("gap_open, expected", [(115.0, 108.0), (105.0, 105.0)])
def test_tighter_protective_stop(engine, gap_open, expected):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [100, 120, 100, 115]
    frame.loc[frame.index[3], ["open", "high", "low", "close"]] = [
        gap_open,
        116,
        90,
        95,
    ]
    result = run_engine(engine, {"AAA": frame}, stop_loss=0.1, trailing_stop=0.1)
    trade = first_lot(result)[0]
    assert trade.exit_price == pytest.approx(expected)
    assert trade.exit_reason == "trail"


def test_full_target_precedes_higher_partial_target(engine):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [101, 125, 100, 120]
    result = run_engine(
        engine, {"AAA": frame}, take_profit=0.1, partial_exits=((0.2, 0.5),)
    )
    trades = first_lot(result)
    assert len(trades) == 1
    assert trades[0].shares == pytest.approx(100)
    assert trades[0].exit_price == pytest.approx(110)


def test_partial_targets_are_price_ordered(engine):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [101, 125, 100, 120]
    results = [
        run_engine(engine, {"AAA": frame}, partial_exits=tiers)
        for tiers in [((0.1, 0.5), (0.2, 0.5)), ((0.2, 0.5), (0.1, 0.5))]
    ]
    for result in results:
        trades = first_lot(result)
        assert [(t.shares, round(t.exit_price)) for t in trades[:2]] == [
            (50, 110),
            (25, 120),
        ]
    assert sum(t.pnl for t in first_lot(results[0])) == pytest.approx(
        sum(t.pnl for t in first_lot(results[1]))
    )


@pytest.mark.parametrize("partial_target", [0.1, 0.2])
def test_full_target_wins_tie_and_gap(engine, partial_target):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [125, 126, 120, 125]
    result = run_engine(
        engine, {"AAA": frame}, take_profit=0.1, partial_exits=((partial_target, 0.5),)
    )
    trades = first_lot(result)
    assert len(trades) == 1
    assert trades[0].shares == pytest.approx(100)
    assert trades[0].exit_price == pytest.approx(125)


def test_target_gapped_through_at_open_beats_later_stop(engine):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [125, 126, 90, 95]
    result = run_engine(engine, {"AAA": frame}, stop_loss=0.05, take_profit=0.1)
    trade = first_lot(result)[0]
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(125)

@pytest.mark.parametrize(
    "tier",
    [(0.0, 0.5), (float("inf"), 0.5), (0.1, 0.0), (0.1, 1.1), (0.1, float("nan"))],
)
def test_invalid_partial_targets_are_rejected(engine, tier):
    with pytest.raises(ValueError, match="Partial exit"):
        run_engine(engine, {"AAA": bars()}, partial_exits=(tier,))


def test_complete_partial_stops_target_processing(engine):
    frame = bars()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = [101, 125, 100, 120]
    result = run_engine(engine, {"AAA": frame}, partial_exits=((0.1, 1.0), (0.2, 0.5)))
    assert len(first_lot(result)) == 1
    assert first_lot(result)[0].exit_price == pytest.approx(110)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_fillable_pending_orders_share_cash(engine, unit):
    frame = bars(unit=unit, volume=30000)
    second, third = bars(unit=unit, volume=20000), bars(unit=unit)
    frame.loc[frame.index[2] :, ["open", "high", "low", "close"]] = [400, 401, 399, 400]
    for other in [second, third]:
        other.loc[other.index[1:3], ["open", "high", "low", "close"]] = [
            150,
            151,
            149,
            150,
        ]
    result = run_engine(
        engine,
        {"AAA": frame, "BBB": second, "CCC": third},
        top=3,
        initial_capital=30000,
        hold=20,
        entry_order_type="limit",
        entry_limit_bps=0,
        sizing_rule="reinvested_equal_slot",
    )
    trades = first_lot(result)
    assert {t.ticker for t in trades} == {"AAA", "BBB", "CCC"}
    assert [t.entry_cost for t in trades] == pytest.approx([10000] * 3)


def test_untouched_limit_does_not_reduce_fillable_cash_share(engine):
    frame = bars(unit="ns", volume=30000)
    second, third = bars(unit="ns", volume=20000), bars(unit="ns")
    frame.loc[frame.index[2] :, ["open", "high", "low", "close"]] = [400, 401, 399, 400]
    second.loc[second.index[1:3], ["open", "high", "low", "close"]] = [
        150,
        151,
        149,
        150,
    ]
    third.loc[third.index[1] :, ["open", "high", "low", "close"]] = [150, 151, 149, 150]
    result = run_engine(
        engine,
        {"AAA": frame, "BBB": second, "CCC": third},
        top=3,
        initial_capital=30000,
        hold=20,
        entry_order_type="limit",
        entry_limit_bps=0,
        sizing_rule="reinvested_equal_slot",
    )
    assert {t.ticker: t.entry_cost for t in first_lot(result)} == pytest.approx(
        {"AAA": 10000, "BBB": 20000}
    )


@pytest.mark.parametrize("order_type", ["moo", "moc"])
def test_future_market_price_cannot_select_earlier_lower_rank(engine, order_type):
    results = []
    for future_price in [100.0, 101.0]:
        first, second = bars(volume=20000), bars()
        first = first.drop(first.index[1])
        first.loc[first.index[1], ["open", "high", "low", "close"]] = [
            future_price,
            102,
            99,
            future_price,
        ]
        results.append(
            run_engine(
                engine,
                {"NSE:AAA": first, "NSE:BBB": second},
                market="india",
                initial_capital=100,
                hold=20,
                entry_order_type=order_type,
            )
        )
    # January 2 is before AAA's execution bar on January 3.
    for result in results:
        assert not any(
            t.ticker == "NSE:BBB" and t.entry_date <= date(2024, 1, 2)
            for t in result.trades
        )
    assert any(
        t.ticker == "NSE:BBB" and t.entry_date == date(2024, 1, 4)
        for t in results[1].trades
    )
    if engine == "rolling":
        assert results[0].selection.iloc[0]["ticker"] == "NSE:AAA"
        assert results[1].selection.iloc[0]["ticker"] == "NSE:AAA"


@pytest.mark.parametrize("order_type", ["moo", "moc"])
def test_pending_market_order_does_not_spend_cash_early(
    engine, order_type, monkeypatch
):
    open_dates = []
    original_open = Portfolio.open

    def record_open(self, ticker, entry_date, entry_price, **kwargs):
        open_dates.append(entry_date)
        return original_open(self, ticker, entry_date, entry_price, **kwargs)

    monkeypatch.setattr(Portfolio, "open", record_open)
    first, second = bars(volume=20000), bars()
    first = first.drop(first.index[1:3])
    second.loc[second.index[2] :, ["open", "high", "low", "close"]] = [
        400,
        401,
        399,
        400,
    ]
    second["dividend"] = 0.0
    second.loc[second.index[2], "dividend"] = 100.0
    result = run_engine(
        engine,
        {"AAA": first, "BBB": second},
        top=2,
        hold=20,
        sizing_rule="reinvested_equal_slot",
        entry_order_type=order_type,
    )
    first_trade = next(t for t in first_lot(result) if t.ticker == "AAA")
    # AAA is sized when it executes, after BBB's gain and dividend are known.
    # On the signal day there is no debit for either queued order.
    assert first_trade.entry_date == date(2024, 1, 4)
    assert first_trade.entry_cost == pytest.approx(10000)
    assert open_dates == sorted(open_dates)
    assert result.equity_curve.loc["2024-01-02"] == pytest.approx(10000)


def test_capped_buy_fees_are_exactly_affordable(engine):
    result = run_engine(
        engine, {"AAA": bars()}, initial_capital=20000, cost_model="us_vested"
    )
    trade = first_lot(result)[0]
    assert trade.entry_cost <= 20000
    assert trade.shares == pytest.approx(199.65)


@pytest.mark.parametrize("dividend", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_dividend_replay_matches_ledger(engine, dividend):
    frame = bars()
    frame["dividend"] = 0.0
    frame.loc[frame.index[2], "dividend"] = dividend
    result = run_engine(engine, {"AAA": frame})
    assert np.isfinite(result.equity_curve).all()
    assert result.metrics["final_equity"] == pytest.approx(
        10000 + sum(t.pnl for t in result.trades)
    )


def test_historical_fetch_covers_reserve_simulation_horizon():
    result = run_engine(
        "historical",
        {
            "AAA": bars(120, volume=30000),
            "BBB": bars(120, volume=20000),
            "CCC": bars(120),
        },
        hold=20,
    )
    trade = result.trades[-1]
    assert trade.ticker == "CCC"
    assert trade.exit_date == date(2024, 3, 28)
    assert trade.exit_reason == "time"


@pytest.mark.parametrize("whole_shares", [False, True])
def test_quoted_shares_respect_explicit_budget_and_cash(whole_shares):
    portfolio = Portfolio(20000, 1, VestedUSCosts(), whole_shares=whole_shares)
    position = portfolio.open("AAA", date(2024, 1, 1), 101, budget=1000, shares=100)
    assert position.slot_capital <= 1000
    assert portfolio.cash() >= 19000


@pytest.mark.parametrize("capital", [100.0, 13999.0, 14000.0, 14035.0, 20000.0])
@pytest.mark.parametrize("quoted_shares", [None, 10000.0])
def test_exact_capped_fee_boundary_never_overdraws(capital, quoted_shares):
    portfolio = Portfolio(capital, 1, VestedUSCosts())
    position = portfolio.open("AAA", date(2024, 1, 1), 100, shares=quoted_shares)
    assert position.slot_capital <= capital
    assert 0 <= portfolio.cash() < 1e-9


def test_impact_fill_respects_risk_budget(engine):
    frame = bars()
    # The signal has a defined volatility and an impact-sensitive share quote.
    frame.loc[frame.index[0], ["open", "high", "low", "close"]] = [80, 81, 79, 80]
    frame.loc[frame.index[1], ["open", "high", "low", "close"]] = [90, 91, 89, 90]
    result = run_engine(
        engine,
        {"AAA": frame},
        as_of=date(2024, 1, 3),
        slippage_model=VolumeImpactSlippage(k=1),
        cost_model="us_vested",
        sizing_rule="fixed_fraction",
        sizing_position_pct=0.1,
    )
    trade = result.trades[0]
    assert trade.entry_price > 100
    assert trade.entry_cost <= 1000
