"""Engine + portfolio accuracy tests with offline synthetic data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.engine import run_backtest, simulate_ticker
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.pine import parse
from screener.backtester.portfolio import Portfolio, build_equity_curve

from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=10,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        strategy_name=None,
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _trend_bars(
    *,
    start: str = "2024-01-01",
    n: int = 80,
    start_px: float = 100.0,
    end_px: float = 150.0,
    volume: float = 100_000.0,
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    close = pd.Series(
        np.linspace(start_px, end_px, n),
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(volume, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


# ── entry/exit mechanics ──────────────────────────────────────────────


def test_entry_fills_next_day_open():
    bars = make_bars(n=10)
    # signal on bar index 3 → entry on bar 4
    outcome = simulate_ticker(bars, signal_idx=3, cfg=_cfg(hold=2))
    assert outcome.trade is not None
    assert outcome.trade.entry_date == bars.index[4].date()
    assert outcome.trade.entry_price == pytest.approx(float(bars.iloc[4]["open"]))


def test_no_post_signal_bar_emits_warning_and_no_trade():
    bars = make_bars(n=5)
    outcome = simulate_ticker(bars, signal_idx=4, cfg=_cfg(hold=2))
    assert outcome.trade is None
    assert outcome.warning and "no post-signal" in outcome.warning


def test_stop_loss_triggers_from_low():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 100.0, "close": 100.2},
            5: {"open": 100.2, "high": 100.5, "low": 89.0, "close": 95.0},
        },
    )
    cfg = _cfg(hold=10, stop_loss=0.05)  # 5% stop → stop_price = 95.0
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "stop"
    expected_stop = 100.0 * (1 - 0.05)
    assert outcome.trade.exit_price == pytest.approx(expected_stop)
    assert outcome.trade.exit_date == bars.index[5].date()


def test_take_profit_triggers_from_high():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.2},
            5: {"open": 100.2, "high": 130.0, "low": 100.0, "close": 110.0},
        },
    )
    cfg = _cfg(hold=10, take_profit=0.10)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "target"
    assert outcome.trade.exit_price == pytest.approx(100.0 * 1.10)


def test_same_bar_stop_and_target_stop_wins():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            5: {"open": 100.0, "high": 130.0, "low": 85.0, "close": 100.0},
        },
    )
    cfg = _cfg(hold=10, stop_loss=0.05, take_profit=0.10)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "stop"


def test_same_bar_trail_and_target_trail_wins():
    # Entry at 100; bar5 lifts peak to 109 without hitting 10% target;
    # bar6 hits both trail (ref=109*0.9=98.1; low=95) and target (ref=110;
    # high=115) on the same bar — trail must win per the docstring rule.
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            5: {"open": 100.0, "high": 109.0, "low": 99.8, "close": 109.0},
            6: {"open": 109.0, "high": 115.0, "low": 95.0, "close": 100.0},
        },
    )
    cfg = _cfg(hold=10, trailing_stop=0.10, take_profit=0.10)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "trail"
    assert outcome.trade.exit_price == pytest.approx(109.0 * 0.9)


def test_trailing_stop_tracks_peak():
    # Entry at 100, bar1 runs up to 120, bar2 drops to 100 → trail_ref = 120*(1-0.10)=108; low=100 hits trail
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            5: {"open": 100.0, "high": 120.0, "low": 99.8, "close": 118.0},
            6: {"open": 118.0, "high": 118.5, "low": 100.0, "close": 101.0},
        },
    )
    cfg = _cfg(hold=10, trailing_stop=0.10)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "trail"
    assert outcome.trade.exit_price == pytest.approx(120.0 * 0.9)


def test_exit_expression_triggers_at_close():
    bars = make_bars(n=15, seed=2)
    exit_ast = parse("close < open")
    # force bar 7 to have close<open, prior bars close>=open
    for i in range(5, 7):
        bars.iat[i, bars.columns.get_loc("close")] = float(bars.iat[i, bars.columns.get_loc("open")]) + 1.0
    bars.iat[7, bars.columns.get_loc("close")] = float(bars.iat[7, bars.columns.get_loc("open")]) - 2.0
    cfg = _cfg(hold=20)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg, exit_ast=exit_ast)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "exit_expr"
    assert outcome.trade.exit_date == bars.index[7].date()
    assert outcome.trade.exit_price == pytest.approx(float(bars.iloc[7]["close"]))


def test_time_exit_after_N_bars():
    bars = make_bars(n=20)
    cfg = _cfg(hold=5)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "time"
    # entry at bar 4, hold=5 → exit at close of bar 4+5=9
    assert outcome.trade.exit_date == bars.index[9].date()


# ── slippage / commission ────────────────────────────────────────────


def test_slippage_reduces_return_vs_zero_slip():
    bars = make_bars(n=20)
    # find a reliable entry bar and run with 0 and 50 bps slip
    o0 = simulate_ticker(bars, signal_idx=3, cfg=_cfg(hold=5, slippage_bps=0.0))
    o1 = simulate_ticker(bars, signal_idx=3, cfg=_cfg(hold=5, slippage_bps=50.0))
    assert o0.trade is not None and o1.trade is not None
    # slipped entry > zero-slip entry and slipped exit < zero-slip exit
    assert o1.trade.entry_price > o0.trade.entry_price
    assert o1.trade.exit_price < o0.trade.exit_price


def test_commission_reduces_realized_return():
    bars = make_bars(n=20, drift=0.2, seed=7)
    portfolio_a = Portfolio(100_000, slot_count=1)
    portfolio_a.assign("AAA", 1, bars.index[3].date())
    outcome = simulate_ticker(bars, signal_idx=3, cfg=_cfg(hold=5))
    assert outcome.trade is not None
    portfolio_a.open("AAA", outcome.trade.entry_date, outcome.trade.entry_price, 0.0)
    trade_a = portfolio_a.close("AAA", outcome.trade.exit_date, outcome.trade.exit_price, "time", 0.0)

    portfolio_b = Portfolio(100_000, slot_count=1)
    portfolio_b.assign("AAA", 1, bars.index[3].date())
    portfolio_b.open("AAA", outcome.trade.entry_date, outcome.trade.entry_price, 50.0)
    trade_b = portfolio_b.close("AAA", outcome.trade.exit_date, outcome.trade.exit_price, "time", 50.0)

    assert trade_b.pnl < trade_a.pnl


def test_run_backtest_rs_breakout_us_selects_relative_strength_breakout():
    aaa = _trend_bars(end_px=150.0)
    aaa.iloc[69, aaa.columns.get_loc("volume")] = 250_000.0
    aaa.iloc[70, aaa.columns.get_loc("open")] = 151.0
    bbb = _trend_bars(start_px=100.0, end_px=108.0)
    spy = _trend_bars(start_px=100.0, end_px=110.0)
    as_of = aaa.index[69].date()

    fetcher = StubPriceFetcher({"AAA": aaa, "BBB": bbb, "SPY": spy})
    cfg = _cfg(
        as_of=as_of,
        hold=3,
        top=1,
        tickers=("AAA", "BBB"),
        strategy_name="rs_breakout",
        entry_expr="rs_breakout_entry > 0",
    )

    result = run_backtest(cfg, fetcher)

    assert result.selection["ticker"].tolist() == ["AAA"]
    assert result.trades
    assert result.trades[0].ticker == "AAA"
    assert result.trades[0].entry_date == aaa.index[70].date()


def test_run_backtest_rs_breakout_india_requires_rising_delivery(monkeypatch):
    aaa = _trend_bars(end_px=150.0)
    aaa.iloc[69, aaa.columns.get_loc("volume")] = 250_000.0
    aaa.iloc[70, aaa.columns.get_loc("open")] = 151.0
    bbb = _trend_bars(end_px=149.0)
    bbb.iloc[69, bbb.columns.get_loc("volume")] = 250_000.0
    nifty = _trend_bars(start_px=100.0, end_px=110.0)
    as_of = aaa.index[69].date()

    delivery_panel = pd.DataFrame(
        [
            {"SYMBOL": "AAA", "date": aaa.index[68].date(), "DELIV_PER": 40.0},
            {"SYMBOL": "AAA", "date": aaa.index[69].date(), "DELIV_PER": 55.0},
            {"SYMBOL": "BBB", "date": bbb.index[68].date(), "DELIV_PER": 55.0},
            {"SYMBOL": "BBB", "date": bbb.index[69].date(), "DELIV_PER": 40.0},
        ]
    )

    monkeypatch.setattr(
        "screener.unusual_volume.delivery.load_delivery_panel",
        lambda symbols, as_of, history_days=40: delivery_panel,
    )

    fetcher = StubPriceFetcher(
        {"AAA.NS": aaa, "BBB.NS": bbb, "^NSEI": nifty}
    )
    cfg = _cfg(
        market="india",
        benchmark="^NSEI",
        as_of=as_of,
        hold=3,
        top=2,
        tickers=("AAA", "BBB"),
        strategy_name="rs_breakout",
        entry_expr="rs_breakout_entry > 0",
    )

    result = run_backtest(cfg, fetcher)

    assert result.selection["ticker"].tolist() == ["AAA"]
    assert [trade.ticker for trade in result.trades] == ["AAA"]


# ── portfolio accounting ─────────────────────────────────────────────


def test_cash_stays_cash_after_exit_two_ticker_portfolio():
    # Ticker A exits early at a known price; equity must stay constant for A after exit.
    bars_a = make_bars(n=20, seed=1, open_base=100.0)
    bars_b = make_bars(n=20, seed=2, open_base=50.0)
    # Force A: entry bar (index 4) open=100; we'll close at bar 6 by time exit (hold=2)
    bars_a.iat[4, bars_a.columns.get_loc("open")] = 100.0
    bars_a.iat[6, bars_a.columns.get_loc("close")] = 110.0
    # Force B: long hold via hold=15, open=50, close smoothly
    bars_b.iat[4, bars_b.columns.get_loc("open")] = 50.0

    trade_a = _simulate_and_record(bars_a, "AAA", rank=1, hold=2, as_of_idx=3, initial=100_000, slot=2)
    trade_b = _simulate_and_record(bars_b, "BBB", rank=2, hold=15, as_of_idx=3, initial=100_000, slot=2)

    calendar = pd.DatetimeIndex(
        sorted(set(bars_a.index.tolist()) | set(bars_b.index.tolist()))
    )
    panel = {"AAA": bars_a, "BBB": bars_b}
    equity = build_equity_curve(
        calendar, [trade_a, trade_b], panel, initial_capital=100_000
    )

    # After A's exit, cash portion from A is fixed at trade_a.exit_value. A's
    # contribution to equity on every subsequent day is constant.
    exit_day = pd.Timestamp(trade_a.exit_date)
    days_after = equity.loc[equity.index > exit_day]
    # Recompute B contribution ourselves and verify total = B_shares * B_close + (cash)
    b_shares = trade_b.shares
    # "cash" = initial - A entry_cost - B entry_cost + A exit_value
    static_cash = 100_000 - trade_a.entry_cost - trade_b.entry_cost + trade_a.exit_value
    for day in days_after.index:
        if day > pd.Timestamp(trade_b.exit_date):
            # after both exit
            expected = (
                100_000
                - trade_a.entry_cost
                + trade_a.exit_value
                - trade_b.entry_cost
                + trade_b.exit_value
            )
            assert equity.loc[day] == pytest.approx(expected, rel=1e-9)
        else:
            expected = static_cash + b_shares * float(bars_b.loc[day, "close"])
            assert equity.loc[day] == pytest.approx(expected, rel=1e-9)


def _simulate_and_record(
    bars: pd.DataFrame,
    ticker: str,
    rank: int,
    hold: int,
    as_of_idx: int,
    initial: float,
    slot: int,
) -> Trade:
    """Helper: simulate one ticker and push into a fresh 1-slot portfolio."""
    outcome = simulate_ticker(bars, signal_idx=as_of_idx, cfg=_cfg(hold=hold))
    assert outcome.trade is not None
    p = Portfolio(initial, slot_count=slot)
    p.assign(ticker, rank, bars.index[as_of_idx].date())
    p.open(ticker, outcome.trade.entry_date, outcome.trade.entry_price, 0.0)
    return p.close(
        ticker, outcome.trade.exit_date, outcome.trade.exit_price, outcome.trade.exit_reason, 0.0
    )


# ── selection + ranking ──────────────────────────────────────────────


def test_output_rank_preserves_selection_rank_not_realized_return(stub_fetcher_factory):
    # Three tickers, dollar volume AAA > BBB > CCC. AAA will LOSE money; CCC will WIN.
    bars_aaa = make_bars(n=60, seed=1, open_base=100.0)
    bars_bbb = make_bars(n=60, seed=2, open_base=50.0)
    bars_ccc = make_bars(n=60, seed=3, open_base=10.0)
    # Volumes: AAA highest, CCC lowest
    bars_aaa["volume"] = 1_000_000
    bars_bbb["volume"] = 500_000
    bars_ccc["volume"] = 100_000
    # Force close on as-of (bar 39) above sma so all pass entry
    for b in (bars_aaa, bars_bbb, bars_ccc):
        b.iat[39, b.columns.get_loc("close")] = float(b.iloc[39]["close"]) + 20
    # AAA price drops after; CCC rises
    for i in range(40, 60):
        bars_aaa.iat[i, bars_aaa.columns.get_loc("close")] = 50.0
        bars_aaa.iat[i, bars_aaa.columns.get_loc("open")] = 50.0
        bars_aaa.iat[i, bars_aaa.columns.get_loc("high")] = 51.0
        bars_aaa.iat[i, bars_aaa.columns.get_loc("low")] = 49.0
        bars_ccc.iat[i, bars_ccc.columns.get_loc("close")] = 40.0
        bars_ccc.iat[i, bars_ccc.columns.get_loc("open")] = 40.0
        bars_ccc.iat[i, bars_ccc.columns.get_loc("high")] = 41.0
        bars_ccc.iat[i, bars_ccc.columns.get_loc("low")] = 39.0

    fetcher = stub_fetcher_factory(
        {"AAA": bars_aaa, "BBB": bars_bbb, "CCC": bars_ccc, "SPY": bars_bbb.copy()}
    )
    cfg = _cfg(
        as_of=bars_aaa.index[39].date(),
        hold=10,
        top=3,
        entry_expr="close > sma(close, 3)",
        tickers=("AAA", "BBB", "CCC"),
    )
    result = run_backtest(cfg, fetcher)
    ranks = [t.rank for t in sorted(result.trades, key=lambda t: t.rank)]
    tickers = [t.ticker for t in sorted(result.trades, key=lambda t: t.rank)]
    assert ranks == [1, 2, 3]
    assert tickers == ["AAA", "BBB", "CCC"]


def test_insufficient_lookback_emits_warning(stub_fetcher_factory):
    # Only 30 bars, but sma(close, 200) needs 200
    bars = make_bars(n=30)
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[-1].date(),
        hold=5,
        top=1,
        entry_expr="close > sma(close, 200)",
        tickers=("AAA",),
    )
    result = run_backtest(cfg, fetcher)
    assert any("insufficient lookback" in w for w in result.warnings)


# ── metrics ──────────────────────────────────────────────────────────


# ── universe, filters, reallocation ──────────────────────────────────


def test_run_backtest_errors_when_no_universe_provided():
    """The TradingView fallback was removed; no universe → ValueError."""
    from screener.backtester.engine import _resolve_universe

    cfg = _cfg(tickers=None)
    with pytest.raises(ValueError, match="No universe provided"):
        _resolve_universe(cfg)


def test_min_price_filter_excludes_penny_stocks(stub_fetcher_factory):
    # PENNY close ~ $0.50; REAL close ~ $100. Both flash the same entry signal.
    bars_penny = make_bars(n=60, seed=1, open_base=0.5)
    bars_real = make_bars(n=60, seed=2, open_base=100.0)
    # Pin penny's last three closes to sub-dollar so both the filter and the
    # sma(close, 3) comparison are deterministic.
    for i in range(37, 40):
        bars_penny.iat[i, bars_penny.columns.get_loc("close")] = 0.30
    bars_penny.iat[39, bars_penny.columns.get_loc("close")] = 0.80  # > sma but still < $1
    bars_real.iat[39, bars_real.columns.get_loc("close")] = (
        float(bars_real.iloc[39]["close"]) + 5
    )

    fetcher = stub_fetcher_factory(
        {"PENNY": bars_penny, "REAL": bars_real, "SPY": bars_real.copy()}
    )
    cfg = _cfg(
        as_of=bars_real.index[39].date(),
        hold=3,
        top=5,
        entry_expr="close > sma(close, 3)",
        tickers=("PENNY", "REAL"),
        min_price=1.0,
    )
    result = run_backtest(cfg, fetcher)
    tickers_traded = {t.ticker for t in result.trades}
    assert "PENNY" not in tickers_traded
    assert "REAL" in tickers_traded
    assert any("filtered" in w and "price/liquidity" in w for w in result.warnings)


def test_min_avg_dollar_volume_filter_excludes_illiquid(stub_fetcher_factory):
    bars_liquid = make_bars(n=60, seed=1, open_base=100.0)
    bars_illiquid = make_bars(n=60, seed=2, open_base=100.0)
    bars_liquid["volume"] = 50_000_000.0  # dollar-vol ~ 5B
    bars_illiquid["volume"] = 1.0  # dollar-vol ~ 100
    for b in (bars_liquid, bars_illiquid):
        b.iat[39, b.columns.get_loc("close")] = float(b.iloc[39]["close"]) + 5

    fetcher = stub_fetcher_factory(
        {
            "LIQ": bars_liquid,
            "ILLIQ": bars_illiquid,
            "SPY": bars_liquid.copy(),
        }
    )
    cfg = _cfg(
        as_of=bars_liquid.index[39].date(),
        hold=3,
        top=5,
        entry_expr="close > sma(close, 3)",
        tickers=("LIQ", "ILLIQ"),
        min_avg_dollar_volume=1_000_000.0,
        avg_dollar_volume_window=20,
    )
    result = run_backtest(cfg, fetcher)
    tickers_traded = {t.ticker for t in result.trades}
    assert "ILLIQ" not in tickers_traded
    assert "LIQ" in tickers_traded


def test_reserve_reallocation_fills_slot_on_early_exit(stub_fetcher_factory):
    """With reinvest=True and a stop-loss exit on day 2, a reserve should be
    opened to replace the freed slot."""
    bars_active = make_bars(n=60, seed=1, open_base=100.0)
    bars_reserve = make_bars(n=60, seed=2, open_base=100.0)
    # Flatten RESERVE's series to make the entry signal deterministic at
    # BOTH as_of (bar 39) AND the active's exit_date (bar 41).
    for i in range(30, 60):
        bars_reserve.iat[i, bars_reserve.columns.get_loc("open")] = 100.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("high")] = 101.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("low")] = 99.0
        bars_reserve.iat[i, bars_reserve.columns.get_loc("close")] = 100.0
    # Pump RESERVE's close at bar 41 so close > sma(close, 3) when slot frees.
    bars_reserve.iat[41, bars_reserve.columns.get_loc("close")] = 105.0
    # Also keep bar 39 signal true for initial selection
    bars_reserve.iat[39, bars_reserve.columns.get_loc("close")] = 105.0

    # ACTIVE: entry signal true at bar 39; entry bar 40; bar 41 stop-out.
    bars_active.iat[39, bars_active.columns.get_loc("close")] = (
        float(bars_active.iloc[39]["close"]) + 5
    )
    bars_active.iat[40, bars_active.columns.get_loc("open")] = 100.0
    bars_active.iat[41, bars_active.columns.get_loc("open")] = 99.0
    bars_active.iat[41, bars_active.columns.get_loc("high")] = 99.5
    bars_active.iat[41, bars_active.columns.get_loc("low")] = 90.0
    bars_active.iat[41, bars_active.columns.get_loc("close")] = 91.0
    # ACTIVE ranks #1 by dollar volume, RESERVE #2.
    bars_active["volume"] = 1_000_000.0
    bars_reserve["volume"] = 100_000.0

    fetcher = stub_fetcher_factory(
        {
            "ACTIVE": bars_active,
            "RESERVE": bars_reserve,
            "SPY": bars_reserve.copy(),
        }
    )
    cfg = _cfg(
        as_of=bars_active.index[39].date(),
        hold=10,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("ACTIVE", "RESERVE"),
        stop_loss=0.05,
        reserve_multiple=3,
        reinvest=True,
    )
    result = run_backtest(cfg, fetcher)
    trade_tickers = [t.ticker for t in result.trades]
    # We expect both ACTIVE (stopped out) and RESERVE (opened after) to trade.
    assert "ACTIVE" in trade_tickers
    assert "RESERVE" in trade_tickers
    active_trade = next(t for t in result.trades if t.ticker == "ACTIVE")
    reserve_trade = next(t for t in result.trades if t.ticker == "RESERVE")
    assert active_trade.exit_reason == "stop"
    # Reserve opens on the bar AFTER active's exit_date
    assert reserve_trade.entry_date > active_trade.exit_date


def test_no_reinvest_matches_legacy_leaves_cash_idle(stub_fetcher_factory):
    """With reinvest=False, a stop-out must NOT trigger reserve rotation even
    if reserves are available."""
    bars_active = make_bars(n=60, seed=1, open_base=100.0)
    bars_reserve = make_bars(n=60, seed=2, open_base=100.0)
    for i in range(30, 60):
        bars_reserve.iat[i, bars_reserve.columns.get_loc("close")] = 100.0
    bars_reserve.iat[39, bars_reserve.columns.get_loc("close")] = 105.0
    bars_reserve.iat[41, bars_reserve.columns.get_loc("close")] = 105.0
    bars_active.iat[39, bars_active.columns.get_loc("close")] = (
        float(bars_active.iloc[39]["close"]) + 5
    )
    bars_active.iat[40, bars_active.columns.get_loc("open")] = 100.0
    bars_active.iat[41, bars_active.columns.get_loc("low")] = 90.0
    bars_active.iat[41, bars_active.columns.get_loc("close")] = 91.0
    bars_active["volume"] = 1_000_000.0
    bars_reserve["volume"] = 100_000.0

    fetcher = stub_fetcher_factory(
        {
            "ACTIVE": bars_active,
            "RESERVE": bars_reserve,
            "SPY": bars_reserve.copy(),
        }
    )
    cfg = _cfg(
        as_of=bars_active.index[39].date(),
        hold=10,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("ACTIVE", "RESERVE"),
        stop_loss=0.05,
        reserve_multiple=3,
        reinvest=False,
    )
    result = run_backtest(cfg, fetcher)
    trade_tickers = {t.ticker for t in result.trades}
    assert "ACTIVE" in trade_tickers
    assert "RESERVE" not in trade_tickers


def test_reserve_filter_rechecked_on_exit_day(stub_fetcher_factory):
    """A reserve that was liquid at original as_of but whose price crashes
    below min_price before the active slot frees must NOT be promoted."""
    bars_active = make_bars(n=60, seed=1, open_base=100.0)
    bars_crash = make_bars(n=60, seed=2, open_base=5.0)
    bars_backup = make_bars(n=60, seed=3, open_base=100.0)

    # Flatten BACKUP and CRASH so the entry signal at bar 41 is deterministic.
    for i in range(30, 60):
        bars_backup.iat[i, bars_backup.columns.get_loc("close")] = 100.0
        bars_crash.iat[i, bars_crash.columns.get_loc("close")] = 5.0

    # All three flash the entry signal at as_of (bar 39, close > sma(close,3)).
    bars_active.iat[39, bars_active.columns.get_loc("close")] = (
        float(bars_active.iloc[39]["close"]) + 5
    )
    bars_crash.iat[39, bars_crash.columns.get_loc("close")] = 8.0  # > min_price
    bars_backup.iat[39, bars_backup.columns.get_loc("close")] = 105.0

    # ACTIVE stops out on bar 41
    bars_active.iat[40, bars_active.columns.get_loc("open")] = 100.0
    bars_active.iat[41, bars_active.columns.get_loc("low")] = 90.0
    bars_active.iat[41, bars_active.columns.get_loc("close")] = 91.0

    # CRASH: by bar 41 price is $0.50 — below min_price=1 filter on exit day.
    for i in range(40, 45):
        bars_crash.iat[i, bars_crash.columns.get_loc("open")] = 0.5
        bars_crash.iat[i, bars_crash.columns.get_loc("high")] = 0.6
        bars_crash.iat[i, bars_crash.columns.get_loc("low")] = 0.4
        bars_crash.iat[i, bars_crash.columns.get_loc("close")] = 0.5
    # Flip close on 41 so entry signal would fire if filter weren't re-checked.
    bars_crash.iat[41, bars_crash.columns.get_loc("close")] = 0.8

    # BACKUP has a clean entry signal on bar 41 and passes filter.
    bars_backup.iat[41, bars_backup.columns.get_loc("close")] = 105.0

    # Volumes: ACTIVE > CRASH > BACKUP (ranks 1, 2, 3).
    bars_active["volume"] = 1_000_000.0
    bars_crash["volume"] = 500_000.0
    bars_backup["volume"] = 100_000.0

    fetcher = stub_fetcher_factory(
        {
            "ACTIVE": bars_active,
            "CRASH": bars_crash,
            "BACKUP": bars_backup,
            "SPY": bars_backup.copy(),
        }
    )
    cfg = _cfg(
        as_of=bars_active.index[39].date(),
        hold=10,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("ACTIVE", "CRASH", "BACKUP"),
        stop_loss=0.05,
        reserve_multiple=3,
        reinvest=True,
        min_price=1.0,
    )
    result = run_backtest(cfg, fetcher)
    trade_tickers = {t.ticker for t in result.trades}
    assert "ACTIVE" in trade_tickers
    # CRASH was rank-2 at as_of but fails the price filter on the exit_date
    assert "CRASH" not in trade_tickers
    # BACKUP is rank-3 and still passes → promoted in CRASH's place
    assert "BACKUP" in trade_tickers


def test_invested_return_metric_ignores_idle_cash():
    """A single winning trade with entry_cost=$10k and pnl=$1k should yield
    invested_return=10%, regardless of the total equity base."""
    trade = Trade(
        ticker="X",
        rank=1,
        signal_date=date(2024, 1, 2),
        entry_date=date(2024, 1, 3),
        entry_price=100.0,
        exit_date=date(2024, 1, 5),
        exit_price=110.0,
        exit_reason="time",
        shares=100.0,
        entry_cost=10_000.0,
        exit_value=11_000.0,
        pnl=1_000.0,
        return_pct=0.10,
    )
    equity = pd.Series(
        [100_000.0, 100_500.0, 101_000.0],
        index=pd.bdate_range("2024-01-03", periods=3),
    )
    bench = pd.Series(
        [100.0, 100.0, 100.0], index=equity.index
    )
    m = compute_metrics(equity, bench, [trade], slot_count=10)
    assert m["invested_return"] == pytest.approx(0.10, abs=1e-6)
    # Total return on $100k portfolio is only 1% — exposes the dead-cash gap
    assert m["total_return"] == pytest.approx(0.01, abs=1e-6)


def test_metrics_on_known_ramp_series():
    n = 252
    equity = pd.Series(
        np.linspace(100_000, 110_000, n),
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    bench = equity.copy()  # identical
    m = compute_metrics(equity, bench, trades=[], slot_count=1)
    assert m["total_return"] == pytest.approx(0.10, abs=1e-6)
    # CAGR over ~1y ≈ 10%
    assert m["cagr"] == pytest.approx(0.10, abs=0.01)
    # monotone up → no drawdown
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    # identical to benchmark → beta ≈ 1
    assert m["beta"] == pytest.approx(1.0, abs=1e-6)
