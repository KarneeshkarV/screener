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


@pytest.mark.parametrize("slippage_bps", [10.0, 50.0])
def test_buy_limit_never_fills_above_its_limit(engine, slippage_bps):
    """Slippage on a touched buy limit is capped at the limit price."""
    frame = bars()
    frame.iloc[1] = [100.0, 100.5, 98.0, 99.5, 10000.0]
    result = run_engine(
        engine,
        {"AAA": frame},
        entry_order_type="limit",
        entry_limit_bps=100.0,
        slippage_bps=slippage_bps,
    )
    trade = first_lot(result)[0]
    assert trade.entry_date == date(2024, 1, 2)
    assert trade.entry_price == pytest.approx(99.0)


def test_buy_limit_keeps_slippage_below_the_limit(engine):
    """A gap below the limit still pays slippage, up to the limit."""
    frame = bars()
    frame.iloc[1] = [95.0, 96.0, 94.0, 95.5, 10000.0]
    result = run_engine(
        engine,
        {"AAA": frame},
        entry_order_type="limit",
        entry_limit_bps=100.0,
        slippage_bps=10.0,
    )
    assert first_lot(result)[0].entry_price == pytest.approx(95.0 * 1.001)


@pytest.mark.parametrize("tickers", [("AAA", "ZZZ"), ("ZZZ", "AAA")])
def test_engines_break_dollar_volume_ties_the_same_way(tickers):
    """Equal dollar volume: both engines buy the first listed name."""
    frames = {ticker: bars() for ticker in tickers}
    picks = {
        engine: first_lot(run_engine(engine, frames))[0].ticker
        for engine in ("historical", "rolling")
    }
    assert picks == {"historical": tickers[0], "rolling": tickers[0]}


def test_moc_does_not_open_and_force_close_on_final_bar(engine):
    """A pending MOC order on the final bar has no next session to hold into.

    Filling it at the final bar's close and immediately force-closing at the
    exact same close would create a phantom 0-bar trade that pays round-trip
    slippage and commissions.
    """
    # 4 bars: d0 signals AAA -> MOC fills on d1 close -> hold=1 exits on d2 close
    # -> d2 queues a new MOC order (rolling refill or historical reserve BBB)
    # whose execution bar would be d3 (the final bar).
    frames = {
        "AAA": bars(4, volume=20000),
        "BBB": bars(4, volume=10000),
    }
    result = run_engine(
        engine,
        frames,
        top=1,
        hold=1,
        reinvest=True,
        reserve_multiple=2,
        entry_order_type="moc",
        slippage_bps=50.0,
        commission_bps=50.0,
    )
    assert len(result.trades) == 1
    assert result.trades[0].ticker == "AAA"
    assert result.trades[0].entry_date == date(2024, 1, 2)
    assert result.trades[0].exit_date == date(2024, 1, 3)
    assert result.trades[0].exit_reason == "time"


def test_historical_single_post_signal_bar_skips_final_bar_moc():
    """With only one bar after as_of, MOC must not open and close on that bar."""
    result = run_engine(
        "historical",
        {"AAA": bars(2)},
        top=1,
        hold=5,
        entry_order_type="moc",
        slippage_bps=50.0,
        commission_bps=50.0,
    )
    assert result.trades == []
    assert result.metrics["final_equity"] == pytest.approx(10_000.0)


def test_rolling_reinvested_equal_slot_queueing_uses_constant_cash_ceiling():
    """Queueing does not debit cash, so every free slot shares cash / len(free_slots)."""
    idx = pd.bdate_range("2024-01-01", periods=5)

    def _frame(closes: list[float], sigs: list[float], volume: float) -> pd.DataFrame:
        c = pd.Series(closes, index=idx, dtype=float)
        return pd.DataFrame(
            {
                "open": c,
                "high": c + 1.0,
                "low": c - 1.0,
                "close": c,
                "volume": volume,
                "sig": sigs,
            },
            index=idx,
        )

    # Day 0: only HOLD signals -> buys 1 share at 100 on day 1 (cash=200).
    # Day 2: HOLD rallies to 700 -> marked equity=900 (equity/3=300), cash=200,
    # 2 free slots => fair cash ceiling per slot is 100.
    # CHEAP1 (80) fits; EXPENSIVE (150) exceeds 100 and must be skipped for
    # slot 2 too so CHEAP2 (80) is queued instead.
    frames = {
        "NSE:HOLD": _frame([100, 100, 700, 700, 700], [1, 0, 0, 0, 0], 40_000),
        "NSE:CHEAP1": _frame([80, 80, 80, 80, 80], [0, 0, 1, 0, 0], 30_000),
        "NSE:EXPENSIVE": _frame([150, 150, 150, 150, 150], [0, 0, 1, 0, 0], 12_000),
        "NSE:CHEAP2": _frame([80, 80, 80, 80, 80], [0, 0, 1, 0, 0], 10_000),
    }
    result = run_engine(
        "rolling",
        frames,
        market="india",
        top=3,
        initial_capital=300.0,
        hold=10,
        entry_expr="sig > 0",
        sizing_rule="reinvested_equal_slot",
    )
    day3_entries = {t.ticker for t in result.trades if t.entry_date == date(2024, 1, 4)}
    assert day3_entries == {"NSE:CHEAP1", "NSE:CHEAP2"}


def test_historical_reserve_queueing_uses_constant_cash_ceiling():
    """Reserve queueing under reinvested_equal_slot shares cash / len(refillable)."""
    idx = pd.bdate_range("2024-01-01", periods=5)

    def _frame(price: float, volume: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": volume,
            },
            index=idx,
        )

    hold_frame = _frame(100.0, 60_000)
    hold_frame.loc[idx[2:], ["open", "high", "low", "close"]] = [700, 701, 699, 700]

    stop1 = _frame(100.0, 50_000)
    stop1.loc[idx[2], ["open", "high", "low", "close"]] = [100, 101, 80, 90]

    stop2 = _frame(100.0, 40_000)
    stop2.loc[idx[2], ["open", "high", "low", "close"]] = [100, 101, 80, 90]

    # Day 1: HOLD, STOP1, STOP2 each buy 1 share @ 100 (cash=0).
    # Day 2: STOP1 & STOP2 stop at 90 (cash=180), HOLD marks 700 (equity=880,
    # equity/3=293.33). Two refillable slots => fair cash ceiling is 180/2=90.
    # CHEAP1 (80) fits; EXPENSIVE (140) exceeds 90 and must be deferred for
    # slot 2 too so CHEAP2 (80) is queued instead.
    frames = {
        "NSE:HOLD": hold_frame,
        "NSE:STOP1": stop1,
        "NSE:STOP2": stop2,
        "NSE:CHEAP1": _frame(80.0, 30_000),
        "NSE:EXPENSIVE": _frame(140.0, 12_000),
        "NSE:CHEAP2": _frame(80.0, 10_000),
    }
    result = run_engine(
        "historical",
        frames,
        market="india",
        top=3,
        reserve_multiple=2,
        reinvest=True,
        initial_capital=300.0,
        hold=10,
        stop_loss=0.1,
        sizing_rule="reinvested_equal_slot",
    )
    day3_entries = {t.ticker for t in result.trades if t.entry_date == date(2024, 1, 4)}
    assert day3_entries == {"NSE:CHEAP1", "NSE:CHEAP2"}
