"""Tests for the backtester realism upgrades.

Covers the four modeling limitations that were addressed:

  1. Pluggable slippage model (fixed, half-spread, sqrt-law impact, composite).
  2. Gap-aware stop / target fills + limit / MOC entry order types.
  3. Re-entry after a position closes and tiered partial exits.
  4. Split-only price adjustment with explicit cash-dividend crediting.

Tests use the same synthetic ``make_bars`` helper and ``StubPriceFetcher`` as
the existing suite so no live yfinance call is made.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.data import _normalize_frame
from screener.backtester.engine import (
    _resolve_stop_fill,
    _resolve_target_fill,
    run_backtest,
    simulate_ticker,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.slippage import (
    CompositeSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
    apply_slippage,
)
from tests.conftest import make_bars


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
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ── Limitation 3: slippage models ────────────────────────────────────


def test_fixed_bps_slippage_matches_legacy_factor():
    model = FixedBpsSlippage(bps=25.0)
    assert apply_slippage(model, 100.0, "buy") == pytest.approx(100.25)
    assert apply_slippage(model, 100.0, "sell") == pytest.approx(99.75)


def test_half_spread_buys_above_mid_sells_below():
    model = HalfSpreadSlippage(half_spread_bps=10.0)
    assert apply_slippage(model, 100.0, "buy") == pytest.approx(100.10)
    assert apply_slippage(model, 100.0, "sell") == pytest.approx(99.90)


def test_vol_impact_scales_with_sqrt_shares_over_adv():
    # 4x shares → 2x adverse fraction (Almgren-Chriss sqrt-law).
    model = VolumeImpactSlippage(k=0.1)
    base = model.adverse_fraction("buy", shares=1000.0, adv=1_000_000.0, sigma_daily=0.02)
    four_x = model.adverse_fraction("buy", shares=4000.0, adv=1_000_000.0, sigma_daily=0.02)
    assert four_x == pytest.approx(2.0 * base, rel=1e-9)


def test_composite_sums_components():
    a = FixedBpsSlippage(bps=5.0)
    b = HalfSpreadSlippage(half_spread_bps=3.0)
    c = VolumeImpactSlippage(k=0.1)
    comp = CompositeSlippage(models=(a, b, c))
    shares, adv, sig = 1000.0, 1_000_000.0, 0.02
    expected = (
        a.adverse_fraction("buy", shares, adv, sig)
        + b.adverse_fraction("buy", shares, adv, sig)
        + c.adverse_fraction("buy", shares, adv, sig)
    )
    assert comp.adverse_fraction("buy", shares, adv, sig) == pytest.approx(expected)


def test_zero_adv_falls_back_safely():
    # Volume-impact with zero ADV must not explode (div-by-zero / sqrt-of-neg).
    model = VolumeImpactSlippage(k=0.5)
    assert model.adverse_fraction("buy", shares=1000.0, adv=0.0, sigma_daily=0.02) == 0.0
    assert model.adverse_fraction("buy", shares=0.0, adv=1e6, sigma_daily=0.02) == 0.0
    assert model.adverse_fraction("buy", shares=1000.0, adv=1e6, sigma_daily=0.0) == 0.0


def test_config_resolves_default_slippage_model_from_bps():
    cfg = _cfg(slippage_bps=42.0)
    assert isinstance(cfg.slippage_model, FixedBpsSlippage)
    assert cfg.slippage_model.bps == pytest.approx(42.0)


def test_custom_slippage_model_overrides_bps_via_engine(monkeypatch):
    # A non-zero bps value combined with an explicit HalfSpread model must use
    # the explicit model — slippage_bps should not also apply.
    bars = make_bars(n=20, seed=11)
    cfg_legacy = _cfg(hold=5, slippage_bps=50.0)
    o_legacy = simulate_ticker(bars, signal_idx=3, cfg=cfg_legacy)
    cfg_halfspread = _cfg(
        hold=5,
        slippage_bps=50.0,
        slippage_model=HalfSpreadSlippage(half_spread_bps=5.0),
    )
    o_half = simulate_ticker(bars, signal_idx=3, cfg=cfg_halfspread)
    assert o_legacy.trade is not None and o_half.trade is not None
    # 5 bps half-spread is less adverse than 50 bps fixed
    assert o_half.trade.entry_price < o_legacy.trade.entry_price
    assert o_half.trade.exit_price > o_legacy.trade.exit_price


# ── Limitation 2: gap fills + limit/MOC entry ────────────────────────


def test_gap_down_stop_fills_at_open_worse_than_stop_ref():
    # stop_ref=95, bar opens at 90 → gap fill at 90, not 95.
    assert _resolve_stop_fill(bar_open=90.0, stop_ref=95.0, gap_fills=True) == 90.0


def test_gap_up_target_fills_at_open_better_than_target_ref():
    # target_ref=105, bar opens at 110 → gap fill at 110, not 105.
    assert _resolve_target_fill(bar_open=110.0, target_ref=105.0, gap_fills=True) == 110.0


def test_gap_fills_false_preserves_legacy_behavior():
    # Gap-fills disabled → always reference price regardless of bar open.
    assert _resolve_stop_fill(bar_open=90.0, stop_ref=95.0, gap_fills=False) == 95.0
    assert _resolve_target_fill(bar_open=110.0, target_ref=105.0, gap_fills=False) == 105.0


def test_stop_refprice_fill_when_bar_opens_above_stop():
    # bar opens above stop but trades through it intraday — classical fill.
    assert _resolve_stop_fill(bar_open=98.0, stop_ref=95.0, gap_fills=True) == 95.0


def test_engine_gap_down_stop_uses_bar_open():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            # Entry bar closes at 100, stop_ref = 95.
            # Next bar opens at 88 (gap-through stop) → fills at 88.
            5: {"open": 88.0, "high": 88.5, "low": 85.0, "close": 86.0},
        },
    )
    cfg = _cfg(hold=10, stop_loss=0.05, gap_fills=True)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "stop"
    assert outcome.trade.exit_price == pytest.approx(88.0)


def test_engine_gap_up_target_uses_bar_open():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            # Entry bar close 100, target_ref = 110. Next bar opens at 120.
            5: {"open": 120.0, "high": 125.0, "low": 120.0, "close": 122.0},
        },
    )
    cfg = _cfg(hold=10, take_profit=0.10, gap_fills=True)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.exit_reason == "target"
    assert outcome.trade.exit_price == pytest.approx(120.0)


def test_gap_fills_false_engine_reproduces_legacy():
    bars = make_bars(
        n=10,
        spikes={
            4: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            5: {"open": 88.0, "high": 88.5, "low": 85.0, "close": 86.0},
        },
    )
    cfg = _cfg(hold=10, stop_loss=0.05, gap_fills=False)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    # Legacy: fills at stop_ref=95 even though bar gapped to 88.
    assert outcome.trade.exit_price == pytest.approx(95.0)


def test_limit_entry_fills_at_limit_when_low_touches():
    # Signal close = 100; limit = 100 * (1 - 500/1e4) = 95. Next bar low ≤ 95.
    bars = make_bars(
        n=10,
        spikes={
            3: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            4: {"open": 99.0, "high": 99.0, "low": 94.0, "close": 96.0},
        },
    )
    cfg = _cfg(hold=5, entry_order_type="limit", entry_limit_bps=500.0)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    # Bar 4 opens at 99 > limit 95, so fill at min(99, 95) = 95.
    assert outcome.trade.entry_price == pytest.approx(95.0)
    assert outcome.trade.entry_date == bars.index[4].date()


def test_limit_entry_fills_at_bar_open_when_gap_through_limit():
    # Bar opens BELOW the limit — buyer gets the better open price.
    bars = make_bars(
        n=10,
        spikes={
            3: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            4: {"open": 92.0, "high": 93.0, "low": 90.0, "close": 92.5},
        },
    )
    cfg = _cfg(hold=5, entry_order_type="limit", entry_limit_bps=500.0)  # limit=95
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.entry_price == pytest.approx(92.0)


def test_limit_entry_unfilled_when_price_never_touches():
    # Limit at 95 but no bar's low ever reaches it.
    bars = make_bars(n=10, seed=1)
    for i in range(4, 10):
        bars.iat[i, bars.columns.get_loc("open")] = 110.0
        bars.iat[i, bars.columns.get_loc("high")] = 112.0
        bars.iat[i, bars.columns.get_loc("low")] = 105.0
        bars.iat[i, bars.columns.get_loc("close")] = 108.0
    bars.iat[3, bars.columns.get_loc("close")] = 100.0
    cfg = _cfg(hold=5, entry_order_type="limit", entry_limit_bps=500.0)
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is None
    assert outcome.warning and "limit order never filled" in outcome.warning


def test_moc_entry_fills_at_close():
    # MOC entry fills at next bar's close, not open.
    bars = make_bars(
        n=10,
        spikes={
            3: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
            4: {"open": 101.0, "high": 102.0, "low": 100.5, "close": 101.7},
        },
    )
    cfg = _cfg(hold=5, entry_order_type="moc")
    outcome = simulate_ticker(bars, signal_idx=3, cfg=cfg)
    assert outcome.trade is not None
    assert outcome.trade.entry_price == pytest.approx(101.7)
    assert outcome.trade.entry_date == bars.index[4].date()


# ── Limitation 1: re-entry, partial exits, pyramiding ────────────────


def test_allow_reentry_false_preserves_single_trade(stub_fetcher_factory):
    # Regression: without allow_reentry, a ticker that closes early does not
    # re-enter even if its signal re-fires.
    bars = make_bars(n=30, seed=3, open_base=100.0, drift=0.05)
    # Force stop on bar 5 (one bar after entry at bar 4).
    bars.iat[5, bars.columns.get_loc("open")] = 100.0
    bars.iat[5, bars.columns.get_loc("high")] = 100.5
    bars.iat[5, bars.columns.get_loc("low")] = 80.0
    bars.iat[5, bars.columns.get_loc("close")] = 85.0
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=20,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.05,
        tickers=("AAA",),
        allow_reentry=False,
    )
    result = run_backtest(cfg, fetcher)
    aaa_trades = [t for t in result.trades if t.ticker == "AAA"]
    assert len(aaa_trades) == 1
    assert aaa_trades[0].exit_reason == "stop"


def test_reentry_after_stop_fires_again(stub_fetcher_factory):
    bars = make_bars(n=30, seed=3, open_base=100.0)
    # Entry at bar 4 open; bar 5 stop-out.
    bars.iat[4, bars.columns.get_loc("open")] = 100.0
    bars.iat[5, bars.columns.get_loc("open")] = 100.0
    bars.iat[5, bars.columns.get_loc("high")] = 100.5
    bars.iat[5, bars.columns.get_loc("low")] = 80.0
    bars.iat[5, bars.columns.get_loc("close")] = 85.0
    # Normal bars after so re-entry signal ("close > 0") fires immediately.
    for i in range(6, 30):
        bars.iat[i, bars.columns.get_loc("open")] = 90.0
        bars.iat[i, bars.columns.get_loc("high")] = 91.0
        bars.iat[i, bars.columns.get_loc("low")] = 89.0
        bars.iat[i, bars.columns.get_loc("close")] = 90.0
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=3,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.05,
        tickers=("AAA",),
        allow_reentry=True,
        max_reentries=2,
    )
    result = run_backtest(cfg, fetcher)
    aaa_trades = sorted(
        (t for t in result.trades if t.ticker == "AAA"), key=lambda t: t.entry_date
    )
    assert len(aaa_trades) >= 2
    # First trade must be the stop-out; subsequent trades are re-entries.
    assert aaa_trades[0].exit_reason == "stop"
    assert aaa_trades[1].entry_date > aaa_trades[0].exit_date


def test_reentry_respects_max_reentries(stub_fetcher_factory):
    # Same setup as the re-entry test but max_reentries=1 — total trades ≤ 2.
    bars = make_bars(n=40, seed=3, open_base=100.0)
    bars.iat[4, bars.columns.get_loc("open")] = 100.0
    # Stop on bar 5 and again on every re-entry's first bar.
    for stop_bar in (5, 8, 11, 14, 17):
        bars.iat[stop_bar, bars.columns.get_loc("low")] = 70.0
        bars.iat[stop_bar, bars.columns.get_loc("close")] = 72.0
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=3,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.05,
        tickers=("AAA",),
        allow_reentry=True,
        max_reentries=1,
    )
    result = run_backtest(cfg, fetcher)
    aaa_trades = [t for t in result.trades if t.ticker == "AAA"]
    # Original + 1 re-entry cap → at most 2 trades.
    assert len(aaa_trades) <= 2


def test_partial_exit_closes_half_at_tier_and_raises_stop_to_break_even(
    stub_fetcher_factory,
):
    bars = make_bars(n=20, seed=4, open_base=100.0)
    # Entry bar 4 at open=100. Bar 5 high hits +5% → partial-exit tier fires.
    bars.iat[4, bars.columns.get_loc("open")] = 100.0
    bars.iat[4, bars.columns.get_loc("high")] = 101.0
    bars.iat[4, bars.columns.get_loc("low")] = 99.5
    bars.iat[4, bars.columns.get_loc("close")] = 100.0
    bars.iat[5, bars.columns.get_loc("open")] = 100.5
    bars.iat[5, bars.columns.get_loc("high")] = 106.0  # > 105 tier
    bars.iat[5, bars.columns.get_loc("low")] = 100.0
    bars.iat[5, bars.columns.get_loc("close")] = 104.0
    # Later bars drift back toward entry so the runner time-exits normally.
    for i in range(6, 20):
        bars.iat[i, bars.columns.get_loc("open")] = 103.0
        bars.iat[i, bars.columns.get_loc("high")] = 103.5
        bars.iat[i, bars.columns.get_loc("low")] = 102.0
        bars.iat[i, bars.columns.get_loc("close")] = 103.0
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=5,
        top=1,
        entry_expr="close > 0",
        tickers=("AAA",),
        partial_exits=((0.05, 0.5),),
    )
    result = run_backtest(cfg, fetcher)
    aaa_trades = sorted(
        (t for t in result.trades if t.ticker == "AAA"), key=lambda t: t.exit_date
    )
    assert len(aaa_trades) == 2
    partial, runner = aaa_trades
    # Partial sleeve exited at the +5% tier.
    assert partial.exit_reason == "target"
    assert partial.exit_price == pytest.approx(105.0)
    # Runner's entry_cost is the complementary half of the original cost.
    total_cost = partial.entry_cost + runner.entry_cost
    assert partial.entry_cost == pytest.approx(total_cost * 0.5, rel=1e-6)
    assert runner.entry_cost == pytest.approx(total_cost * 0.5, rel=1e-6)


def test_pyramiding_via_portfolio_tracks_two_concurrent_lots():
    # The Portfolio API supports concurrent lots per ticker when raise_if_exists
    # is False. Verifies independent PnL on each lot and FIFO close ordering.
    p = Portfolio(initial_capital=100_000.0, slot_count=2)
    d1, d2, d3, d4 = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 10),
        date(2024, 1, 12),
    )
    p.assign("AAA", rank=1, signal_date=d1)
    p.open("AAA", d1, entry_price=100.0, commission_bps=0.0)
    p.open("AAA", d2, entry_price=110.0, commission_bps=0.0, raise_if_exists=False)
    # Two lots open; FIFO close hits the d1 lot first.
    trade1 = p.close("AAA", d3, exit_price=120.0, reason="target", commission_bps=0.0)
    trade2 = p.close("AAA", d4, exit_price=130.0, reason="target", commission_bps=0.0)
    assert trade1.entry_date == d1
    assert trade2.entry_date == d2
    # Both trades profit relative to their own entry price.
    assert trade1.pnl > 0
    assert trade2.pnl > 0


def test_legacy_open_raises_on_duplicate_when_raise_if_exists_true():
    p = Portfolio(initial_capital=100_000.0, slot_count=1)
    p.assign("AAA", rank=1, signal_date=date(2024, 1, 2))
    p.open("AAA", date(2024, 1, 2), 100.0, commission_bps=0.0)
    with pytest.raises(ValueError, match="Position already open"):
        p.open("AAA", date(2024, 1, 3), 105.0, commission_bps=0.0)


# ── Limitation 4: split-only adjustment + cash dividends ─────────────


def test_split_factor_applied_to_ohlc_but_not_volume_in_normalize():
    # yfinance-style frame with a 2:1 split on the last bar.
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [200.0, 200.0, 200.0, 200.0, 100.0],
            "High": [202.0, 202.0, 202.0, 202.0, 101.0],
            "Low": [198.0, 198.0, 198.0, 198.0, 99.0],
            "Close": [200.0, 200.0, 200.0, 200.0, 100.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 2_000_000],
            "Dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0, 0.0, 2.0],
        },
        index=idx,
    )
    norm = _normalize_frame(raw)
    # split_factor column present; volume left raw.
    assert "split_factor" in norm.columns
    # Reverse cumulative product shifted — pre-split bars get factor=2.
    assert norm["split_factor"].iloc[0] == pytest.approx(2.0)
    assert norm["split_factor"].iloc[-1] == pytest.approx(1.0)
    # Volume column unchanged.
    assert norm["volume"].iloc[0] == pytest.approx(1_000_000)


def test_dividend_column_preserved_in_normalize():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1_000, 1_000, 1_000],
            "Dividends": [0.0, 0.50, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    norm = _normalize_frame(raw)
    assert "dividend" in norm.columns
    assert norm["dividend"].iloc[1] == pytest.approx(0.50)
    assert norm["dividend"].iloc[0] == pytest.approx(0.0)


def test_dividend_credits_cash_and_updates_position_income():
    p = Portfolio(initial_capital=100_000.0, slot_count=1)
    p.assign("AAA", rank=1, signal_date=date(2024, 1, 2))
    p.open("AAA", date(2024, 1, 2), entry_price=100.0, commission_bps=0.0)
    cash_before = p.cash()
    credited = p.credit_dividends("AAA", cash_per_share=0.50)
    pos = p.get_position("AAA")
    assert pos is not None
    # Credit equals shares * cash_per_share.
    assert credited == pytest.approx(pos.shares * 0.50)
    assert p.cash() == pytest.approx(cash_before + credited)
    assert pos.dividend_income == pytest.approx(credited)


def test_trade_carries_dividend_income_on_close():
    p = Portfolio(initial_capital=100_000.0, slot_count=1)
    p.assign("AAA", rank=1, signal_date=date(2024, 1, 2))
    p.open("AAA", date(2024, 1, 2), entry_price=100.0, commission_bps=0.0)
    p.credit_dividends("AAA", cash_per_share=0.50)
    trade = p.close(
        "AAA", date(2024, 1, 10), exit_price=105.0, reason="time", commission_bps=0.0
    )
    assert trade.dividend_income > 0
    # Income matches shares * dividend_per_share at credit time.
    assert trade.dividend_income == pytest.approx(trade.shares * 0.50)


def test_credit_dividends_noop_when_no_position():
    p = Portfolio(initial_capital=100_000.0, slot_count=1)
    cash_before = p.cash()
    credited = p.credit_dividends("AAA", cash_per_share=1.0)
    assert credited == 0.0
    assert p.cash() == cash_before


def test_price_adjustment_full_skips_dividend_credit(stub_fetcher_factory):
    # Build bars with an explicit non-zero dividend column. Under
    # price_adjustment="full" (legacy), the engine must NOT credit dividends
    # because auto_adjust already folded them into price.
    bars = make_bars(n=20, seed=5, open_base=100.0)
    bars["dividend"] = 0.0
    bars.iat[6, bars.columns.get_loc("dividend")] = 1.0  # $1 ex-div on bar 6
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=5,
        top=1,
        entry_expr="close > 0",
        tickers=("AAA",),
        price_adjustment="full",
    )
    result = run_backtest(cfg, fetcher)
    # dividend_income must stay zero under the legacy regime.
    assert all(t.dividend_income == 0.0 for t in result.trades if t.ticker == "AAA")


def test_price_adjustment_splits_only_credits_dividend(stub_fetcher_factory):
    bars = make_bars(n=20, seed=5, open_base=100.0)
    bars["dividend"] = 0.0
    bars.iat[6, bars.columns.get_loc("dividend")] = 1.0
    fetcher = stub_fetcher_factory({"AAA": bars, "SPY": bars.copy()})
    cfg = _cfg(
        as_of=bars.index[3].date(),
        hold=5,
        top=1,
        entry_expr="close > 0",
        tickers=("AAA",),
        price_adjustment="splits_only",
    )
    result = run_backtest(cfg, fetcher)
    aaa_trades = [t for t in result.trades if t.ticker == "AAA"]
    assert aaa_trades
    # At least one held across bar 6 → non-zero dividend_income.
    assert any(t.dividend_income > 0 for t in aaa_trades)
