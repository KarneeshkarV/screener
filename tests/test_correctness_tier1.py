"""Tier 1: shared features — VBT fast path vs core `simulate_ticker` + portfolio helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.core import simulate_ticker
from screener.backtester.engine import run_backtest
from screener.backtester.metrics import compute_metrics
from screener.backtester.models import BacktestConfig, Trade
from screener.backtester.pine import evaluate, parse
from screener.backtester.vbt_adapter import run_vbt

from tests.backtester_synthetic import run_core_portfolio_path
from tests.conftest import StubPriceFetcher, make_bars
from tests.correctness_fixtures import make_audit_panel


def _cfg(**overrides) -> BacktestConfig:
    base = dict(
        market="us",
        as_of=date(2023, 1, 2),
        hold=20,
        top=1,
        entry_expr="entry_day0 > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        allow_reentry=False,
        gap_fills=False,
        reinvest=False,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _assert_vbt_core_single(
    bars: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    entry_expr: str | None = None,
    exit_expr: str | None = None,
    signal_idx: int = 0,
) -> None:
    entry = cfg.entry_expr if entry_expr is None else entry_expr
    exit_ast = parse(exit_expr) if (exit_expr or cfg.exit_expr) else None
    if exit_ast is None and cfg.exit_expr:
        exit_ast = parse(cfg.exit_expr)
    core = simulate_ticker(bars, signal_idx, cfg, exit_ast=exit_ast)
    assert core.trade is not None, core.warning
    entry_ast = parse(entry)
    vbt = run_vbt(cfg, {"X": bars}, entry_ast, exit_ast)
    assert len(vbt.trades) == 1, vbt.warnings
    ct, vt = core.trade, vbt.trades[0]
    assert ct.exit_reason == vt.exit_reason
    assert ct.entry_price == pytest.approx(vt.entry_price, abs=1e-6)
    assert ct.exit_price == pytest.approx(vt.exit_price, abs=1e-6)
    assert ct.entry_date == vt.entry_date
    assert ct.exit_date == vt.exit_date


def test_tier1_moo_entry_fill_matches_next_open() -> None:
    """MOO entry fill equals signal_bar+1 open before slippage (TICKER_C bar 0)."""
    panel = make_audit_panel()
    bars = panel["TICKER_C"].drop(columns=["audit_signal"], errors="ignore")
    cfg = _cfg(entry_expr="entry_day0 > 0", slippage_bps=0.0, commission_bps=0.0)
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    assert out.trade.entry_price == pytest.approx(float(bars.iloc[1]["open"]), abs=1e-6)


def test_tier1_fixed_bps_slippage_entry() -> None:
    """20 bps slippage widens MOO buy: entry = open[1] * 1.002."""
    panel = make_audit_panel()
    bars = panel["TICKER_C"].drop(columns=["audit_signal"], errors="ignore")
    cfg = _cfg(entry_expr="entry_day0 > 0", slippage_bps=20.0)
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None
    ref = float(bars.iloc[1]["open"])
    assert out.trade.entry_price == pytest.approx(ref * 1.002, abs=1e-6)


def test_tier1_fixed_bps_slippage_exit_time() -> None:
    """20 bps slippage on sell: exit = ref_close * (1 - 0.002) for time exit."""
    bars = make_bars(
        n=10, spikes={0: {"close": 100.0, "open": 100.0, "high": 101.0, "low": 99.0}}
    )
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(
        entry_expr="entry_flag > 0",
        hold=2,
        slippage_bps=20.0,
        gap_fills=False,
    )
    out = simulate_ticker(bars, 0, cfg, exit_ast=None)
    assert out.trade is not None and out.trade.exit_reason == "time"
    ref = float(bars.loc[pd.Timestamp(out.trade.exit_date), "close"])
    assert out.trade.exit_price == pytest.approx(ref * (1.0 - 0.002), abs=1e-6)


def test_tier1_commission_round_trip_identity() -> None:
    """PnL matches exit proceeds minus entry cash with commission only."""
    bars = make_bars(
        n=8, spikes={0: {"close": 100.0, "open": 100.0, "high": 101.0, "low": 99.0}}
    )
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(
        entry_expr="entry_flag > 0",
        hold=3,
        slippage_bps=0.0,
        commission_bps=10.0,
    )
    res = run_vbt(cfg, {"Z": bars}, parse(cfg.entry_expr))
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.pnl == pytest.approx(t.exit_value - t.entry_cost, abs=1e-6)


@pytest.mark.parametrize(
    "name,overrides,spikes,exp_reason,exp_exit",
    [
        (
            "stop",
            {"stop_loss": 0.05},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 100.5, "low": 90.0, "close": 96.0},
            },
            "stop",
            95.0,
        ),
        (
            "target",
            {"take_profit": 0.10},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 115.0, "low": 99.5, "close": 108.0},
            },
            "target",
            110.0,
        ),
        (
            "trail",
            {"trailing_stop": 0.10},
            {
                1: {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0},
                2: {"open": 100.0, "high": 120.0, "low": 99.5, "close": 118.0},
                3: {"open": 118.0, "high": 119.0, "low": 107.0, "close": 109.0},
            },
            "trail",
            108.0,
        ),
        (
            "time",
            {"hold": 3},
            {},
            "time",
            None,
        ),
    ],
    ids=["stop_loss", "take_profit", "trailing_stop", "hold_cap"],
)
def test_tier1_vbt_core_risk_controls(
    name: str,
    overrides: dict,
    spikes: dict,
    exp_reason: str,
    exp_exit: float | None,
) -> None:
    """Stops, targets, trailing, and hold-cap: VBT ledger matches core simulate_ticker."""
    bars = make_bars(n=10, spikes=spikes)
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(entry_expr="entry_flag > 0", **overrides)
    _assert_vbt_core_single(bars, cfg, entry_expr="entry_flag > 0", signal_idx=0)
    res = run_vbt(cfg, {"X": bars}, parse("entry_flag > 0"))
    t = res.trades[0]
    assert t.exit_reason == exp_reason
    if exp_exit is not None:
        assert t.exit_price == pytest.approx(exp_exit, abs=1e-6)
    if exp_reason == "time":
        assert (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days >= 0
        entry_i = bars.index.get_loc(pd.Timestamp(t.entry_date))
        exit_i = bars.index.get_loc(pd.Timestamp(t.exit_date))
        assert exit_i - entry_i == 3


def test_tier1_exit_expression_both_engines() -> None:
    """Exit when Pine exit_expr first fires."""
    bars = make_bars(n=20)
    bars["entry_flag"] = 0.0
    bars.iloc[5, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(
        entry_expr="entry_flag > 0",
        exit_expr="close < sma(close, 3)",
        hold=9999,
    )
    exit_ast = parse(cfg.exit_expr)
    core = simulate_ticker(bars, 5, cfg, exit_ast=exit_ast)
    assert core.trade is not None and core.trade.exit_reason == "exit_expr"
    vbt = run_vbt(cfg, {"X": bars}, parse(cfg.entry_expr), exit_ast)
    assert vbt.trades[0].exit_reason == "exit_expr"
    assert vbt.trades[0].exit_date == core.trade.exit_date


def test_tier1_multi_slot_ranking_by_dollar_volume() -> None:
    """Three simultaneous signals ranked 1..3 by signal-bar dollar volume."""
    panel = {
        k: v.drop(columns=["audit_signal"], errors="ignore")
        for k, v in make_audit_panel().items()
        if k in ("TICKER_A", "TICKER_B", "TICKER_C")
    }
    sig_day = panel["TICKER_A"].index[60]
    b = panel["TICKER_B"]
    # engineer ranking A > B > C on the signal bar (post-hoc OHLC tweak)
    b.loc[sig_day, "close"] = 90.0
    b.loc[sig_day, "open"] = 89.5
    b.loc[sig_day, "high"] = 91.0
    b.loc[sig_day, "low"] = 88.5
    a, c = panel["TICKER_A"], panel["TICKER_C"]
    dv_a = float(a.loc[sig_day, "close"] * a.loc[sig_day, "volume"])
    dv_b = float(b.loc[sig_day, "close"] * b.loc[sig_day, "volume"])
    dv_c = float(c.loc[sig_day, "close"] * c.loc[sig_day, "volume"])
    assert dv_a > dv_b > dv_c

    entry_expr = "audit_signal > 0"
    for fr in panel.values():
        fr["audit_signal"] = 0.0
        fr.loc[sig_day, "audit_signal"] = 1.0

    cfg = _cfg(
        entry_expr=entry_expr,
        hold=9999,
        top=3,
        as_of=sig_day.date(),
        slippage_bps=0.0,
        commission_bps=0.0,
    )
    vbt = run_vbt(cfg, panel, parse(entry_expr))

    spy = a[["close"]].rename(columns={"close": "close"}).copy()
    spy["open"] = spy["close"]
    spy["high"] = spy["close"] + 0.01
    spy["low"] = spy["close"] - 0.01
    spy["volume"] = 1.0
    fetch_panel = {**panel, "SPY": spy}
    fetcher = StubPriceFetcher(fetch_panel)
    hist = run_backtest(
        cfg.model_copy(
            update={
                "tickers": tuple(panel.keys()),
                "benchmark": "SPY",
            }
        ),
        fetcher,
    )

    assert len(vbt.trades) == 3
    assert len(hist.trades) == 3
    by_t = {t.ticker: t for t in vbt.trades}
    bh = {t.ticker: t for t in hist.trades}
    assert (
        by_t["TICKER_A"].rank == 1
        and by_t["TICKER_B"].rank == 2
        and by_t["TICKER_C"].rank == 3
    )
    for k in panel:
        assert by_t[k].rank == bh[k].rank


def test_tier1_buy_hold_parity_ticker_c() -> None:
    """VBT vs core synthetic path on TICKER_C long hold."""
    panel = {
        "TICKER_C": make_audit_panel()["TICKER_C"].drop(
            columns=["audit_signal"], errors="ignore"
        )
    }
    cfg = _cfg(
        entry_expr="entry_day0 > 0",
        hold=9999,
        top=1,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
    )
    core = run_core_portfolio_path(cfg, panel)
    vbt = run_vbt(cfg, panel, parse(cfg.entry_expr))
    ct, vt = core.trades[0], vbt.trades[0]
    assert ct.entry_date == vt.entry_date and ct.exit_date == vt.exit_date
    assert ct.entry_price == pytest.approx(vt.entry_price, abs=1e-6)
    assert ct.exit_price == pytest.approx(vt.exit_price, abs=1e-6)
    assert abs(vbt.equity_curve.iloc[-1] / core.equity_curve.iloc[-1] - 1.0) <= 0.0005
    for m in ("sharpe", "cagr", "max_drawdown"):
        assert vbt.metrics[m] == pytest.approx(core.metrics[m], abs=1e-4)


def test_tier1_equity_curve_starts_at_initial_capital() -> None:
    """First equity point equals configured starting capital."""
    bars = make_bars(n=8)
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(entry_expr="entry_flag > 0", hold=2)
    res = run_vbt(cfg, {"X": bars}, parse(cfg.entry_expr))
    assert res.equity_curve.iloc[0] == pytest.approx(cfg.initial_capital, abs=1e-6)


def test_tier1_metrics_total_return_single_trade() -> None:
    """total_return equals (exit_value - entry_cost) / initial_capital for one trade."""
    bars = make_bars(n=8)
    bars["entry_flag"] = 0.0
    bars.iloc[0, bars.columns.get_loc("entry_flag")] = 1.0
    cfg = _cfg(
        entry_expr="entry_flag > 0", hold=2, slippage_bps=0.0, commission_bps=0.0
    )
    res = run_vbt(cfg, {"X": bars}, parse(cfg.entry_expr))
    t = res.trades[0]
    analytical = (t.exit_value - t.entry_cost) / cfg.initial_capital
    assert res.metrics["total_return"] == pytest.approx(analytical, abs=1e-4)


def test_tier1_metrics_hit_rate_two_one() -> None:
    """Two winning trades and one loser → hit_rate ≈ 2/3."""
    trades = [
        Trade(
            ticker="A",
            rank=1,
            signal_date=date(2024, 1, 1),
            entry_date=date(2024, 1, 2),
            entry_price=10.0,
            exit_date=date(2024, 1, 3),
            exit_price=11.0,
            exit_reason="target",
            shares=1.0,
            entry_cost=10.0,
            exit_value=11.0,
            pnl=1.0,
            return_pct=0.1,
        ),
        Trade(
            ticker="B",
            rank=1,
            signal_date=date(2024, 1, 1),
            entry_date=date(2024, 1, 2),
            entry_price=10.0,
            exit_date=date(2024, 1, 3),
            exit_price=12.0,
            exit_reason="target",
            shares=1.0,
            entry_cost=10.0,
            exit_value=12.0,
            pnl=2.0,
            return_pct=0.2,
        ),
        Trade(
            ticker="C",
            rank=1,
            signal_date=date(2024, 1, 1),
            entry_date=date(2024, 1, 2),
            entry_price=10.0,
            exit_date=date(2024, 1, 3),
            exit_price=9.0,
            exit_reason="stop",
            shares=1.0,
            entry_cost=10.0,
            exit_value=9.0,
            pnl=-1.0,
            return_pct=-0.1,
        ),
    ]
    idx = pd.bdate_range("2024-01-02", periods=5)
    equity = pd.Series(
        [100_000.0, 101_000.0, 103_000.0, 102_000.0, 102_500.0], index=idx
    )
    bench = equity.copy()
    m = compute_metrics(equity, bench, trades, slot_count=1)
    assert m["hit_rate"] == pytest.approx(2.0 / 3.0, abs=1e-4)


def test_tier1_allow_reentry_false_single_trade_ticker_e() -> None:
    """Oscillating RSI on TICKER_E: only first entry when allow_reentry=False."""
    bars = make_audit_panel()["TICKER_E"].drop(
        columns=["audit_signal"], errors="ignore"
    )
    rsi = evaluate(parse("rsi(close, 14)"), bars)
    first_os = rsi[rsi < 30].dropna().index[0]
    cfg = _cfg(
        entry_expr="rsi(close, 14) < 30",
        exit_expr="rsi(close, 14) > 70",
        hold=9999,
        top=1,
        allow_reentry=False,
        as_of=first_os.date(),
    )
    vbt = run_vbt(cfg, {"TICKER_E": bars}, parse(cfg.entry_expr), parse(cfg.exit_expr))
    e_trades = [t for t in vbt.trades if t.ticker == "TICKER_E"]
    assert len(e_trades) == 1

    spy = bars.copy()
    spy["ticker"] = "SPY"
    fetcher = StubPriceFetcher({"TICKER_E": bars, "SPY": bars})
    hist = run_backtest(
        cfg.model_copy(update={"tickers": ("TICKER_E",), "benchmark": "SPY"}),
        fetcher,
    )
    assert len([t for t in hist.trades if t.ticker == "TICKER_E"]) == 1
