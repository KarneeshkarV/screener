"""Unit tests for statutory fee models and Corwin-Schultz half-spread."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.costs import (
    FlatCommission,
    IndiaDeliveryCosts,
    build_cost_model,
    corwin_schultz_half_spread,
    cost_model_from_config,
)
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.slippage import (
    CompositeSlippage,
    EstimatedHalfSpreadSlippage,
    FixedBpsSlippage,
    apply_slippage,
)
from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=1,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ── FlatCommission ───────────────────────────────────────────────────


def test_flat_commission_matches_bps_fraction():
    model = FlatCommission(bps=10.0)
    assert model.side_cost_fraction("buy", 100_000.0) == pytest.approx(0.001)
    assert model.side_cost_fraction("sell", 50_000.0) == pytest.approx(0.001)


def test_flat_commission_zero_default():
    assert FlatCommission().side_cost_fraction("buy", 1.0) == 0.0


def test_build_cost_model_flat_and_india():
    flat = build_cost_model("flat", commission_bps=25.0)
    assert isinstance(flat, FlatCommission)
    assert flat.side_cost_fraction("sell", 1.0) == pytest.approx(0.0025)
    india = build_cost_model("india")
    assert isinstance(india, IndiaDeliveryCosts)
    with pytest.raises(ValueError, match="unknown cost_model"):
        build_cost_model("europe")


def test_cost_model_from_config():
    cfg = _cfg(cost_model="flat", commission_bps=5.0)
    model = cost_model_from_config(cfg)
    assert model.side_cost_fraction("buy", 1.0) == pytest.approx(0.0005)
    cfg_in = _cfg(cost_model="india", commission_bps=99.0)
    india = cost_model_from_config(cfg_in)
    assert isinstance(india, IndiaDeliveryCosts)


# ── IndiaDeliveryCosts ───────────────────────────────────────────────


def test_india_delivery_buy_and_sell_fractions_exact():
    """Hand-check every component against published default rates."""
    m = IndiaDeliveryCosts()
    # Buy: brokerage(0) + STT 0.1% + stamp 0.015% + exch 0.00297%
    #      + SEBI 0.0001% + GST 18%*(0+exch+sebi) + IPFT 0.0001%
    brokerage = 0.0
    stt = 0.001
    stamp = 0.00015
    exchange = 0.0000297
    sebi = 0.000001
    ipft = 0.000001
    gst = 0.18 * (brokerage + exchange + sebi)
    expected_buy = brokerage + stt + stamp + exchange + sebi + gst + ipft
    expected_sell = brokerage + stt + exchange + sebi + gst + ipft  # no stamp

    notional = 250_000.0
    assert m.side_cost_fraction("buy", notional) == pytest.approx(expected_buy)
    assert m.side_cost_fraction("sell", notional) == pytest.approx(expected_sell)
    # Stamp duty only on buy → buy cost > sell cost.
    assert expected_buy > expected_sell
    assert m.side_cost_fraction("buy", notional) - m.side_cost_fraction(
        "sell", notional
    ) == pytest.approx(stamp)


def test_india_delivery_custom_brokerage_and_gst_base():
    m = IndiaDeliveryCosts(brokerage_bps=5.0)  # 0.05%
    brokerage = 0.0005
    exchange = m.exchange_txn_rate
    sebi = m.sebi_turnover_rate
    gst = m.gst_rate * (brokerage + exchange + sebi)
    expected_sell = brokerage + m.stt_rate + exchange + sebi + gst + m.ipft_rate
    assert m.side_cost_fraction("sell", 10_000.0) == pytest.approx(expected_sell)


def test_portfolio_india_costs_reduce_cash_vs_flat_zero():
    india = IndiaDeliveryCosts()
    flat = FlatCommission(bps=0.0)

    pf_india = Portfolio(100_000.0, 1)
    pf_flat = Portfolio(100_000.0, 1)
    pf_india.open("AAA", date(2024, 1, 2), 100.0, cost_model=india)
    pf_flat.open("AAA", date(2024, 1, 2), 100.0, cost_model=flat)
    # Same slot budget, higher fees → fewer shares.
    assert pf_india.get_position("AAA").shares < pf_flat.get_position("AAA").shares

    t_india = pf_india.close("AAA", date(2024, 1, 10), 110.0, "time", cost_model=india)
    t_flat = pf_flat.close("AAA", date(2024, 1, 10), 110.0, "time", cost_model=flat)
    assert t_india.pnl < t_flat.pnl


# ── Corwin-Schultz ───────────────────────────────────────────────────


def _constant_spread_ohlc(
    n: int = 60,
    mid: float = 100.0,
    half_spread: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OHLC where each day's high/low embeds a known half-spread.

    High = mid * (1 + half_spread), low = mid * (1 - half_spread) so
    ln(H/L) = ln((1+s)/(1-s)). The CS estimator recovers a positive spread
    of the same order; we assert approximate recovery after rolling mean.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Small random walk of the mid so gamma is not degenerate.
    shocks = rng.normal(0.0, 0.001, size=n).cumsum()
    mids = mid * (1.0 + shocks)
    high = mids * (1.0 + half_spread)
    low = mids * (1.0 - half_spread)
    close = mids
    open_ = np.r_[mids[0], mids[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def test_corwin_schultz_recovers_constructed_spread_approximately():
    true_half = 0.01  # 1% half-spread → ~2% full spread
    bars = _constant_spread_ohlc(n=80, half_spread=true_half)
    est = corwin_schultz_half_spread(bars["high"], bars["low"], window=21)
    # After warmup, estimates should be positive and in the ballpark of true_half.
    tail = est.iloc[30:]
    assert (tail > 0).all()
    assert float(tail.mean()) == pytest.approx(true_half, rel=0.6, abs=0.005)


def test_corwin_schultz_zero_range_and_negative_gamma_edge():
    idx = pd.bdate_range("2024-01-01", periods=10)
    # Zero daily range → ln(H/L)=0 → beta/gamma → 0 spread.
    flat = pd.Series(100.0, index=idx)
    est = corwin_schultz_half_spread(flat, flat, window=5)
    assert (est == 0.0).all()

    # Missing / non-positive high-low → 0.
    high = pd.Series([100.0, np.nan, 0.0, 110.0, 105.0], index=idx[:5])
    low = pd.Series([99.0, 98.0, 0.0, 100.0, 104.0], index=idx[:5])
    est2 = corwin_schultz_half_spread(high, low, window=3)
    assert (est2 >= 0.0).all()
    assert not est2.isna().any()


def test_corwin_schultz_no_lookahead():
    bars = _constant_spread_ohlc(n=40, half_spread=0.005, seed=7)
    base = corwin_schultz_half_spread(bars["high"], bars["low"], window=10)
    t = 20
    value_at_t = float(base.iloc[t])

    mutated = bars.copy()
    # Inflate future highs/lows; day-t estimate must be unchanged.
    mutated.iloc[t + 1 :, mutated.columns.get_loc("high")] *= 1.5
    mutated.iloc[t + 1 :, mutated.columns.get_loc("low")] *= 0.5
    after = corwin_schultz_half_spread(mutated["high"], mutated["low"], window=10)
    assert float(after.iloc[t]) == pytest.approx(value_at_t)
    # And future values should generally move (sanity that mutation took effect).
    assert not np.allclose(
        after.iloc[t + 5 :].to_numpy(), base.iloc[t + 5 :].to_numpy()
    )


def test_corwin_schultz_rejects_nonpositive_window():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="window"):
        corwin_schultz_half_spread(s, s, window=0)


# ── EstimatedHalfSpreadSlippage ──────────────────────────────────────


def test_estimated_half_spread_slippage_uses_per_fill_value():
    model = EstimatedHalfSpreadSlippage()
    assert model.adverse_fraction("buy", 0, 0, 0, half_spread=0.0) == 0.0
    assert model.adverse_fraction("buy", 0, 0, 0, half_spread=0.012) == pytest.approx(
        0.012
    )
    assert model.adverse_fraction("sell", 0, 0, 0, half_spread=-1.0) == 0.0
    buy = apply_slippage(model, 100.0, "buy", half_spread=0.01)
    sell = apply_slippage(model, 100.0, "sell", half_spread=0.01)
    assert buy == pytest.approx(101.0)
    assert sell == pytest.approx(99.0)


def test_composite_with_estimated_half_spread():
    model = CompositeSlippage(
        models=(FixedBpsSlippage(bps=10.0), EstimatedHalfSpreadSlippage())
    )
    # 10 bps + 50 bps half-spread = 60 bps adverse.
    frac = model.adverse_fraction("buy", 0, 0, 0, half_spread=0.005)
    assert frac == pytest.approx(0.001 + 0.005)


# ── Flat cost_model regression (byte-identical legacy path) ──────────


def test_flat_cost_model_backtest_parity_with_legacy_commission():
    """cost_model='flat' must reproduce trades/metrics of the legacy path."""
    bars = make_bars(n=40, seed=11, open_base=100.0, drift=0.15)
    bars["entry_signal"] = 0.0
    # Force a clean entry: mark signal column unused; entry expr uses sma.
    spy = make_bars(n=40, seed=12, open_base=400.0)
    # Ensure volume for filters.
    bars["volume"] = 1_000_000.0
    spy["volume"] = 5_000_000.0
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    as_of = bars.index[15].date()

    cfg_legacy = _cfg(
        as_of=as_of,
        hold=5,
        top=1,
        tickers=("AAA",),
        commission_bps=10.0,
        slippage_bps=5.0,
        # default cost_model="flat"
    )
    cfg_explicit = cfg_legacy.model_copy(update={"cost_model": "flat"})

    r1 = run_backtest(cfg_legacy, fetcher)
    r2 = run_backtest(cfg_explicit, fetcher)
    assert len(r1.trades) == len(r2.trades)
    assert len(r1.trades) >= 1
    for t1, t2 in zip(r1.trades, r2.trades, strict=True):
        assert t1.model_dump() == t2.model_dump()
    assert r1.metrics == r2.metrics
    pd.testing.assert_series_equal(r1.equity_curve, r2.equity_curve)


def test_india_cost_model_changes_trade_cash_vs_flat_zero():
    bars = make_bars(n=40, seed=21, open_base=100.0, drift=0.2)
    bars["volume"] = 1_000_000.0
    spy = make_bars(n=40, seed=22, open_base=400.0)
    spy["volume"] = 5_000_000.0
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    as_of = bars.index[15].date()

    flat = run_backtest(
        _cfg(
            as_of=as_of,
            hold=5,
            top=1,
            tickers=("AAA",),
            commission_bps=0.0,
            cost_model="flat",
        ),
        fetcher,
    )
    india = run_backtest(
        _cfg(
            as_of=as_of,
            hold=5,
            top=1,
            tickers=("AAA",),
            commission_bps=0.0,
            cost_model="india",
        ),
        fetcher,
    )
    assert len(flat.trades) >= 1 and len(india.trades) >= 1
    # Higher statutory fees → lower net PnL all else equal.
    assert sum(t.pnl for t in india.trades) < sum(t.pnl for t in flat.trades)


def test_spread_proxy_worsens_fills_vs_off():
    bars = _constant_spread_ohlc(n=50, half_spread=0.02, seed=3)
    bars["volume"] = 1_000_000.0
    spy = bars.copy()
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    as_of = bars.index[20].date()
    base_kwargs = dict(
        as_of=as_of,
        hold=5,
        top=1,
        tickers=("AAA",),
        commission_bps=0.0,
        slippage_bps=0.0,
        entry_expr="close > 0",  # always true → entry on as_of
    )
    off = run_backtest(_cfg(**base_kwargs, spread_proxy=False), fetcher)
    on_model = CompositeSlippage(
        models=(FixedBpsSlippage(bps=0.0), EstimatedHalfSpreadSlippage())
    )
    on = run_backtest(
        _cfg(**base_kwargs, spread_proxy=True, slippage_model=on_model),
        fetcher,
    )
    assert len(off.trades) >= 1 and len(on.trades) >= 1
    # Buy-side half-spread raises entry; sell-side lowers exit → worse return.
    assert on.trades[0].entry_price >= off.trades[0].entry_price
    assert on.trades[0].return_pct <= off.trades[0].return_pct
