"""Tier 2–3: core event engine analytical ground truth (mostly `simulate_ticker` / `run_backtest`)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.backtester.core import simulate_ticker
from screener.backtester.engine import run_backtest
from screener.backtester.metrics import _sharpe, _sortino, compute_metrics
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.pine import parse
from screener.backtester.slippage import (
    CompositeSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
    apply_slippage,
)
from screener.backtester.vbt_adapter import run_vbt

from tests.conftest import StubPriceFetcher
from tests.correctness_fixtures import make_audit_panel


def _cfg(**kw: object) -> BacktestConfig:
    d = {
        "market": "us",
        "as_of": date(2024, 1, 2),
        "hold": 9999,
        "top": 1,
        "entry_expr": "entry_day0 > 0",
        "exit_expr": None,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "slippage_bps": 10.0,
        "commission_bps": 0.0,
        "initial_capital": 100_000.0,
        "benchmark": "SPY",
        "allow_reentry": False,
        "gap_fills": True,
        "reinvest": False,
    }
    d.update(kw)
    return BacktestConfig(**d)


def test_tier2_gap_fills_true_stop_fill_at_open() -> None:
    """gap_fills=True: gap through stop fills at the bar open (worse than stop ref)."""
    idx = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 92.0, 92.0, 92.0, 92.0],
            "high": [101.0, 101.0, 93.0, 93.0, 93.0, 93.0],
            "low": [99.0, 99.0, 91.0, 91.0, 91.0, 91.0],
            "close": [100.0, 100.0, 92.5, 92.5, 92.5, 92.5],
            "volume": [1e6] * 6,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 6,
        },
        index=idx,
    )
    cfg = _cfg(stop_loss=0.05, slippage_bps=0.0, gap_fills=True, as_of=idx[0].date())
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None and out.trade.exit_reason == "stop"
    assert out.trade.exit_price == pytest.approx(92.0, abs=1e-6)


def test_tier2_gap_fills_true_target_fill_at_open() -> None:
    """gap_fills=True: gap through take-profit fills at the bar open."""
    idx = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 115.0, 115.0, 115.0, 115.0],
            "high": [101.0, 101.0, 116.0, 116.0, 116.0, 116.0],
            "low": [99.0, 99.0, 114.0, 114.0, 114.0, 114.0],
            "close": [100.0, 100.0, 115.5, 115.5, 115.5, 115.5],
            "volume": [1e6] * 6,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 6,
        },
        index=idx,
    )
    cfg = _cfg(take_profit=0.10, slippage_bps=0.0, gap_fills=True, as_of=idx[0].date())
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None and out.trade.exit_reason == "target"
    assert out.trade.exit_price == pytest.approx(115.0, abs=1e-6)


def test_tier2_gap_fills_false_stop_fill_at_stop_ref() -> None:
    """gap_fills=False: stop fills at the stop reference, not the gap open."""
    idx = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 92.0, 92.0, 92.0, 92.0],
            "high": [101.0, 101.0, 93.0, 93.0, 93.0, 93.0],
            "low": [99.0, 99.0, 91.0, 91.0, 91.0, 91.0],
            "close": [100.0, 100.0, 92.5, 92.5, 92.5, 92.5],
            "volume": [1e6] * 6,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 6,
        },
        index=idx,
    )
    cfg = _cfg(stop_loss=0.05, slippage_bps=0.0, gap_fills=False, as_of=idx[0].date())
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None and out.trade.exit_reason == "stop"
    assert out.trade.exit_price == pytest.approx(95.0, abs=1e-6)


def test_tier2_gap_fills_false_target_fill_at_target_ref() -> None:
    """gap_fills=False: take-profit fills at the target reference."""
    idx = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 115.0, 115.0, 115.0, 115.0],
            "high": [101.0, 101.0, 116.0, 116.0, 116.0, 116.0],
            "low": [99.0, 99.0, 114.0, 114.0, 114.0, 114.0],
            "close": [100.0, 100.0, 115.5, 115.5, 115.5, 115.5],
            "volume": [1e6] * 6,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 6,
        },
        index=idx,
    )
    cfg = _cfg(take_profit=0.10, slippage_bps=0.0, gap_fills=False, as_of=idx[0].date())
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None and out.trade.exit_reason == "target"
    assert out.trade.exit_price == pytest.approx(110.0, abs=1e-6)


def test_tier2_partial_exit_then_breakeven_stop() -> None:
    """Partial take-profit books half; remaining stop ratchets to entry (breakeven)."""
    idx = pd.bdate_range("2024-01-02", periods=12)
    o = h = lo = c = [100.0] * 12
    bars = pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": [1e6] * 12,
            "entry_day0": [1.0] + [0.0] * 11,
            "dividend": [0.0] * 12,
        },
        index=idx,
    )
    entry_fill = 100.0 * 1.001
    tranche_px = entry_fill * 1.10
    bars.loc[idx[3], ["high", "close"]] = tranche_px + 1.0
    bars.loc[idx[3], "low"] = tranche_px - 0.5
    bars.loc[idx[3], "open"] = entry_fill + 0.2
    for i in range(4, 12):
        px = entry_fill - 0.5 * (i - 3)
        bars.loc[idx[i], ["open", "high", "low", "close"]] = px

    cfg = _cfg(
        partial_exits=((0.10, 0.5),),
        stop_loss=0.05,
        take_profit=None,
        slippage_bps=10.0,
        as_of=idx[0].date(),
        hold=9999,
        gap_fills=False,
    )
    spy = bars.copy()
    fetcher = StubPriceFetcher({"AAA": bars, "SPY": spy})
    res = run_backtest(
        cfg.model_copy(
            update={
                "tickers": ("AAA",),
                "benchmark": "SPY",
                "entry_expr": "entry_day0 > 0",
            }
        ),
        fetcher,
    )
    partials = [t for t in res.trades if t.exit_reason == "target"]
    stops = [t for t in res.trades if t.exit_reason == "stop"]
    assert len(partials) == 1 and partials[0].shares > 0
    assert len(stops) == 1
    assert stops[0].exit_price == pytest.approx(entry_fill * (1.0 - 0.001), abs=1e-5)


def test_tier2_moc_entry_uses_next_close() -> None:
    """MOC entry fills at next bar close."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame(
        {
            "open": [10.0, 20.0, 20.0, 20.0, 20.0],
            "high": [11.0, 21.0, 21.0, 21.0, 21.0],
            "low": [9.0, 19.0, 19.0, 19.0, 19.0],
            "close": [10.5, 25.0, 25.0, 25.0, 25.0],
            "volume": [1e6] * 5,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 5,
        },
        index=idx,
    )
    cfg = _cfg(
        entry_order_type="moc",
        slippage_bps=0.0,
        commission_bps=0.0,
        hold=9999,
        as_of=idx[0].date(),
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    assert out.trade.entry_price == pytest.approx(
        float(bars.iloc[1]["close"]), abs=1e-6
    )


def test_tier2_limit_entry_fills_when_touched() -> None:
    """Limit buy fills at min(open, limit) once low breaches the limit."""
    idx = pd.bdate_range("2024-01-02", periods=6)
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 99.0, 98.0, 98.0, 98.0],
            "high": [101.0, 102.0, 100.0, 99.0, 99.0, 99.0],
            "low": [99.0, 100.5, 96.0, 97.0, 97.0, 97.0],
            "close": [100.0, 101.0, 97.5, 98.0, 98.0, 98.0],
            "volume": [1e6] * 6,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 6,
        },
        index=idx,
    )
    cfg = _cfg(
        entry_order_type="limit",
        entry_limit_bps=200.0,
        slippage_bps=0.0,
        hold=9999,
        as_of=idx[0].date(),
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    limit_px = 100.0 * (1.0 - 200.0 / 10_000.0)
    assert out.trade.entry_price == pytest.approx(min(99.0, limit_px), abs=1e-6)


def test_tier2_limit_never_filled_returns_warning() -> None:
    """Limit entry that never trades leaves ``trade`` unset with a warning."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "close": [100.0, 101.5, 102.5, 103.5, 104.5],
            "volume": [1e6] * 5,
            "entry_day0": [1.0, 0.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 5,
        },
        index=idx,
    )
    cfg = _cfg(
        entry_order_type="limit",
        entry_limit_bps=200.0,
        slippage_bps=0.0,
        hold=9999,
        as_of=idx[0].date(),
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is None
    assert out.warning is not None and "never filled" in out.warning.lower()


def test_tier3_half_spread_slippage_model() -> None:
    """Half-spread model charges the quoted half-spread on each side."""
    idx = pd.bdate_range("2024-01-02", periods=4)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 110.0],
            "high": [101.0, 101.0, 111.0, 111.0],
            "low": [99.0, 99.0, 109.0, 109.0],
            "close": [100.0, 100.0, 110.0, 110.0],
            "volume": [10_000.0] * 4,
            "entry_day0": [1.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 4,
        },
        index=idx,
    )
    cfg = _cfg(
        slippage_bps=0.0,
        slippage_model=HalfSpreadSlippage(half_spread_bps=30.0),
        hold=1,
        as_of=idx[0].date(),
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    assert out.trade.entry_price == pytest.approx(100.30, abs=1e-6)
    assert out.trade.exit_price == pytest.approx(109.67, abs=1e-6)


def test_tier3_volume_impact_slippage_model() -> None:
    """Volume impact adverse fraction follows k * sigma * sqrt(shares / adv)."""
    m = VolumeImpactSlippage(k=0.1)
    adv = 10_000.0
    sigma = 0.02
    shares = 100.0
    frac = m.adverse_fraction("buy", shares, adv, sigma)
    assert frac == pytest.approx(0.1 * 0.02 * (shares / adv) ** 0.5, abs=1e-9)
    fill = apply_slippage(m, 100.0, "buy", shares=shares, adv=adv, sigma_daily=sigma)
    assert fill == pytest.approx(100.0 * (1.0 + 0.0002), abs=1e-6)


def test_tier3_composite_slippage_sums_components() -> None:
    """Composite model sums adverse fractions from nested models."""
    idx = pd.bdate_range("2024-01-02", periods=4)
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 110.0],
            "high": [101.0, 101.0, 111.0, 111.0],
            "low": [99.0, 99.0, 109.0, 109.0],
            "close": [100.0, 100.0, 110.0, 110.0],
            "volume": [10_000.0] * 4,
            "entry_day0": [1.0, 0.0, 0.0, 0.0],
            "dividend": [0.0] * 4,
        },
        index=idx,
    )
    cfg = _cfg(
        slippage_bps=0.0,
        slippage_model=CompositeSlippage(
            models=(FixedBpsSlippage(bps=5.0), HalfSpreadSlippage(half_spread_bps=5.0))
        ),
        hold=1,
        as_of=idx[0].date(),
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    assert out.trade.entry_price == pytest.approx(100.10, abs=1e-6)


def test_tier3_min_price_filter_blocks_ticker_d() -> None:
    """min_price above TICKER_D range yields zero trades."""
    panel = make_audit_panel()
    d = panel["TICKER_D"].drop(columns=["audit_signal"], errors="ignore")
    cfg = _cfg(
        entry_expr="close > 0",
        min_price=3.0,
        slippage_bps=0.0,
        as_of=d.index[30].date(),
        hold=20,
    )
    spy = d.copy()
    fetcher = StubPriceFetcher({"TICKER_D": d, "SPY": spy})
    res = run_backtest(cfg.model_copy(update={"tickers": ("TICKER_D",)}), fetcher)
    assert res.trades == []


def test_tier3_min_avg_dollar_volume_blocks_ticker_d() -> None:
    """ADV filter excludes micro-liquidity TICKER_D."""
    panel = make_audit_panel()
    d = panel["TICKER_D"].drop(columns=["audit_signal"], errors="ignore")
    cfg = _cfg(
        entry_expr="close > 0",
        min_avg_dollar_volume=50_000.0,
        avg_dollar_volume_window=20,
        slippage_bps=0.0,
        as_of=d.index[30].date(),
        hold=20,
    )
    spy = d.copy()
    fetcher = StubPriceFetcher({"TICKER_D": d, "SPY": spy})
    res = run_backtest(cfg.model_copy(update={"tickers": ("TICKER_D",)}), fetcher)
    assert res.trades == []


def test_tier3_allow_reentry_true_multiple_trades_ticker_e() -> None:
    """allow_reentry=True permits multiple round-trips on TICKER_E."""
    bars = make_audit_panel()["TICKER_E"].drop(
        columns=["audit_signal"], errors="ignore"
    )
    cfg = _cfg(
        entry_expr="rsi(close, 14) < 30",
        exit_expr="rsi(close, 14) > 70",
        hold=9999,
        top=1,
        allow_reentry=True,
        max_reentries=50,
        slippage_bps=0.0,
        commission_bps=0.0,
        as_of=bars.index[0].date(),
    )
    res = run_vbt(cfg, {"TICKER_E": bars}, parse(cfg.entry_expr), parse(cfg.exit_expr))
    e_trades = [t for t in res.trades if t.ticker == "TICKER_E"]
    assert len(e_trades) >= 2


def test_tier3_dividend_credit_splits_only() -> None:
    """Dividend column credits cash when price_adjustment is splits_only."""
    panel = make_audit_panel()
    a = panel["TICKER_A"].drop(columns=["audit_signal"], errors="ignore")
    cfg = _cfg(
        entry_expr="entry_day0 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        hold=9999,
        slippage_bps=0.0,
        commission_bps=0.0,
        price_adjustment="splits_only",
        as_of=a.index[0].date(),
    )
    a2 = a.copy()
    a2["entry_day0"] = 0.0
    a2.iloc[40, a2.columns.get_loc("entry_day0")] = 1.0
    spy = a2.copy()
    fetcher = StubPriceFetcher({"TICKER_A": a2, "SPY": spy})
    res = run_backtest(
        cfg.model_copy(update={"tickers": ("TICKER_A",), "as_of": a2.index[40].date()}),
        fetcher,
    )
    assert any(t.dividend_income > 0 for t in res.trades)


def test_tier3_sortino_exceeds_sharpe_on_downside_only_noise() -> None:
    """When downside volatility is tiny versus total volatility, Sortino exceeds Sharpe."""
    idx = pd.bdate_range("2024-01-02", periods=120)
    daily_ret = np.concatenate(
        [np.full(40, -0.001), np.full(40, 0.0005), np.full(40, 0.002)]
    )
    equity = pd.Series((1.0 + daily_ret).cumprod() * 100_000.0, index=idx, dtype=float)
    dret = equity.pct_change().dropna()
    assert _sortino(dret) > _sharpe(dret) + 1e-6


def test_tier3_calmar_matches_cagr_over_abs_drawdown() -> None:
    """Calmar = CAGR / |max_drawdown| when the equity path includes a real valley."""
    idx = pd.bdate_range("2024-01-02", periods=252 * 2)
    eq = np.empty(len(idx), dtype=float)
    eq[:126] = np.linspace(100_000.0, 70_000.0, 126)
    eq[126:] = np.linspace(70_000.0, 130_000.0, len(idx) - 126)
    equity = pd.Series(eq, index=idx, dtype=float)
    bench = equity.copy()
    m = compute_metrics(equity, bench, [], slot_count=1)
    cagr = m["cagr"]
    mdd = m["max_drawdown"]
    assert mdd < 0
    assert m["calmar"] == pytest.approx(cagr / abs(mdd), abs=1e-4)


def test_tier3_alpha_beta_benchmark_tracking_portfolio() -> None:
    """Identical equity and benchmark curves → beta≈1 and alpha≈0."""
    idx = pd.bdate_range("2024-01-02", periods=80)
    r = np.random.default_rng(0).normal(0.001, 0.002, len(idx))
    equity = pd.Series(100_000.0 * np.cumprod(1.0 + r), index=idx)
    bench = equity.copy()
    m = compute_metrics(equity, bench, [], slot_count=1)
    assert m["beta"] == pytest.approx(1.0, abs=1e-3)
    assert m["alpha_annual"] == pytest.approx(0.0, abs=1e-3)


def test_tier3_exposure_half_booked_days() -> None:
    """Exposure counts open slots vs calendar length."""
    idx = pd.bdate_range("2024-01-02", periods=40)
    equity = pd.Series(100_000.0, index=idx, dtype=float)
    bench = equity.copy()
    trades = [
        Trade(
            ticker="Q",
            rank=1,
            signal_date=idx[0].date(),
            entry_date=idx[0].date(),
            entry_price=10.0,
            exit_date=idx[19].date(),
            exit_price=11.0,
            exit_reason="time",
            shares=1.0,
            entry_cost=10.0,
            exit_value=11.0,
            pnl=1.0,
            return_pct=0.1,
        )
    ]
    m = compute_metrics(equity, bench, trades, slot_count=1)
    assert m["exposure"] == pytest.approx(20.0 / 40.0, abs=1e-4)


def test_tier3_invested_return_from_mock_trades() -> None:
    """invested_return aggregates realized PnL over deployed entry_cost."""
    idx = pd.bdate_range("2024-01-02", periods=5)
    equity = pd.Series(
        [100_000.0, 101_000.0, 101_000.0, 101_000.0, 101_000.0], index=idx
    )
    bench = equity.copy()
    trades = [
        Trade(
            ticker="Z",
            rank=1,
            signal_date=idx[0].date(),
            entry_date=idx[0].date(),
            entry_price=100.0,
            exit_date=idx[2].date(),
            exit_price=110.0,
            exit_reason="target",
            shares=100.0,
            entry_cost=10_000.0,
            exit_value=11_000.0,
            pnl=1_000.0,
            return_pct=0.10,
        )
    ]
    m = compute_metrics(equity, bench, trades, slot_count=1)
    assert m["invested_return"] == pytest.approx(0.10, abs=1e-6)
