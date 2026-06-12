"""Phase 4 — hand-computed trade mechanics correctness tests.

Every expected value is derived from formulas READ in the source, with the
derivation shown in comments.  No expected value was obtained by running the
engine and copying output.

Signal→entry offset (verified): signal_idx=3 → entry_idx=4 (next-bar open).

Gap-fill behaviour (verified from _resolve_stop_fill / _resolve_target_fill):
  stop:   open <= stop_ref  → fill at open  (gap_fills=True)
  target: open >= target_ref → fill at open  (gap_fills=True)
  Both default to fill at ref price when gap condition is False.

Slippage (_apply_slip with FixedBpsSlippage, verified from core.py lines 32-52):
  buy:  ref * (1 + bps/10_000)
  sell: ref * (1 - bps/10_000)

Portfolio.open (verified from portfolio.py lines 62-100):
  gross_per_share = entry_price * (1 + commission_bps/10_000)
  budget = min(slot_capital, cash)
  shares = budget / gross_per_share
  entry_cost = shares * entry_price + shares * entry_price * (commission_bps/10_000)
             = shares * entry_price * (1 + c)  [i.e. == budget by construction]

Portfolio.close (verified from portfolio.py lines 134-172):
  proceeds   = shares * exit_price
  commission = proceeds * (commission_bps/10_000)
  exit_value = proceeds - commission
  pnl        = exit_value - entry_cost
  return_pct = pnl / entry_cost
"""

from __future__ import annotations

from datetime import date

import pytest

from screener.backtester.core import simulate_ticker
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio

from tests.conftest import StubPriceFetcher
from tests.correctness.fixtures.explicit_bars import (
    bars_s1_buy_and_hold,
    bars_s2_stop_intrabar,
    bars_s3_target_intrabar,
    bars_s4_gap_down,
    bars_s5_gap_up,
    bars_s6_trailing_stop,
    bars_s7_partial_then_time,
    bars_s8_time_exit,
    bars_s9_commission_slippage,
    make_spy_bars,
)

TOL = 1e-6  # tight tolerance for all exact arithmetic assertions


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(
            2024, 1, 5
        ),  # bars index[3] = 2024-01-04 (Thu); index[4]=Fri 2024-01-05
        hold=100,  # far beyond bar count → eod exit unless overridden
        top=1,
        entry_expr="close > 0",  # always-true signal for run_backtest scenarios
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("TEST",),
        gap_fills=True,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
# Helper: drive a Portfolio for a single simulate_ticker outcome
# ---------------------------------------------------------------------------


def _drive_portfolio(
    trade_outcome,
    entry_price: float,
    exit_price: float,
    commission_bps: float = 0.0,
    initial_capital: float = 100_000.0,
) -> Portfolio:
    """Open and immediately close a single position using the trade outcome dates."""
    t = trade_outcome.trade
    port = Portfolio(initial_capital, slot_count=1)
    port.assign(t.ticker if t.ticker else "TEST", 1, t.signal_date)
    port.open(
        ticker="TEST",
        entry_date=t.entry_date,
        entry_price=entry_price,
        commission_bps=commission_bps,
    )
    port.close(
        ticker="TEST",
        exit_date=t.exit_date,
        exit_price=exit_price,
        reason=t.exit_reason,
        commission_bps=commission_bps,
    )
    return port


# ===========================================================================
# Scenario 1 — Buy-and-hold to EOD (force-close at last bar)
# ===========================================================================
# Bars: 10 bars; signal_idx=3, entry at bar4 open=100.0
# hold=100 (no time-limit hit in 10 bars) → eod exit at bar9 close=115.0
# No slippage, no commission.
#
# Layer A derived values:
#   entry_date  = bars.index[4].date()
#   entry_price = 100.0  (bar4 open, no slippage)
#   exit_date   = bars.index[9].date()  (last bar)
#   exit_price  = 115.0  (bar9 close, no slippage)
#   exit_reason = 'eod'
#
# Layer B derived values:
#   slot_capital = 100_000 / 1 = 100_000
#   budget       = min(100_000, 100_000) = 100_000
#   gross_per_share = 100.0 * (1 + 0) = 100.0
#   shares       = 100_000 / 100.0 = 1000.0
#   entry_cost   = 1000.0 * 100.0 = 100_000.0
#   exit_value   = 1000.0 * 115.0 = 115_000.0  (commission=0)
#   pnl          = 115_000.0 - 100_000.0 = 15_000.0
#   return_pct   = 15_000.0 / 100_000.0 = 0.15
#   final_equity = 100_000 - 100_000 + 115_000 = 115_000.0


class TestS1BuyAndHold:
    def setup_method(self):
        self.bars = bars_s1_buy_and_hold()
        self.cfg = _cfg(hold=100, slippage_bps=0.0, commission_bps=0.0)

    def test_layer_a_entry_exit_dates(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.entry_date == self.bars.index[4].date(), "signal+1 = entry bar 4"
        assert t.exit_date == self.bars.index[9].date(), "last bar is exit"
        assert t.exit_reason == "eod"

    def test_layer_a_fill_prices(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t.entry_price == pytest.approx(100.0, abs=TOL)
        assert t.exit_price == pytest.approx(115.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST", out.trade.exit_date, out.trade.exit_price, "eod", commission_bps=0.0
        )
        assert tr.shares == pytest.approx(1000.0, abs=TOL)
        assert tr.entry_cost == pytest.approx(100_000.0, abs=TOL)
        assert tr.exit_value == pytest.approx(115_000.0, abs=TOL)
        assert tr.pnl == pytest.approx(15_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(0.15, abs=TOL)
        assert port.cash() == pytest.approx(115_000.0, abs=TOL)


# ===========================================================================
# Scenario 2 — Stop hit intrabar → fill at stop_ref (open > stop_ref)
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; stop_loss=0.05
# stop_ref = entry_fill * (1 - 0.05) = 100.0 * 0.95 = 95.0
# bar5: open=99.0 > stop_ref=95.0 → NOT a gap-through; low=89.0 ≤ 95.0 → stop hit
# _resolve_stop_fill(99.0, 95.0, gap_fills=True): open=99>stop_ref=95 → fill=95.0
# No slippage → exit_fill = 95.0
#
# Layer B:
#   shares=1000, entry_cost=100_000, exit_value=95_000, pnl=-5_000, return_pct=-0.05


class TestS2StopIntrabar:
    def setup_method(self):
        self.bars = bars_s2_stop_intrabar()
        self.cfg = _cfg(hold=100, stop_loss=0.05, slippage_bps=0.0, commission_bps=0.0)

    def test_layer_a_exit_reason_and_date(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "stop"
        assert t.exit_date == self.bars.index[5].date()

    def test_layer_a_fill_prices(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t.entry_price == pytest.approx(100.0, abs=TOL)
        assert t.exit_price == pytest.approx(95.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "stop",
            commission_bps=0.0,
        )
        assert tr.shares == pytest.approx(1000.0, abs=TOL)
        assert tr.entry_cost == pytest.approx(100_000.0, abs=TOL)
        assert tr.exit_value == pytest.approx(95_000.0, abs=TOL)
        assert tr.pnl == pytest.approx(-5_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(-0.05, abs=TOL)


# ===========================================================================
# Scenario 3 — Target hit intrabar → fill at target_ref (open < target_ref)
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; take_profit=0.10
# target_ref = 100.0 * 1.10 = 110.0
# bar5: open=102.0 < target_ref=110.0; high=115.0 ≥ 110.0 → target hit
# _resolve_target_fill(102.0, 110.0, gap_fills=True): open=102<target=110 → fill=110.0
#
# Layer B:
#   shares=1000, entry_cost=100_000, exit_value=110_000, pnl=10_000, return_pct=0.10


class TestS3TargetIntrabar:
    def setup_method(self):
        self.bars = bars_s3_target_intrabar()
        self.cfg = _cfg(
            hold=100, take_profit=0.10, slippage_bps=0.0, commission_bps=0.0
        )

    def test_layer_a_exit_reason_and_date(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "target"
        assert t.exit_date == self.bars.index[5].date()

    def test_layer_a_fill_prices(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t.entry_price == pytest.approx(100.0, abs=TOL)
        assert t.exit_price == pytest.approx(110.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "target",
            commission_bps=0.0,
        )
        assert tr.shares == pytest.approx(1000.0, abs=TOL)
        assert tr.pnl == pytest.approx(10_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(0.10, abs=TOL)


# ===========================================================================
# Scenario 4 — Gap-DOWN through stop: gap_fills=True → fill at open
#              Companion with gap_fills=False → fill at stop_ref
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; stop_loss=0.05 → stop_ref=95.0
# bar5: open=90.0 ≤ stop_ref=95.0 → gap condition true
#   gap_fills=True:  _resolve_stop_fill(90.0, 95.0, True)  → 90.0  (fill at open)
#   gap_fills=False: _resolve_stop_fill(90.0, 95.0, False) → 95.0  (fill at stop_ref)
#
# gap_fills=True  Layer B: exit_value=90_000, pnl=-10_000, return_pct=-0.10
# gap_fills=False Layer B: exit_value=95_000, pnl=-5_000,  return_pct=-0.05


class TestS4GapDown:
    def setup_method(self):
        self.bars = bars_s4_gap_down()

    def test_gap_fills_true_fill_at_open(self):
        cfg = _cfg(hold=100, stop_loss=0.05, gap_fills=True, slippage_bps=0.0)
        out = simulate_ticker(self.bars, signal_idx=3, cfg=cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "stop"
        assert t.exit_date == self.bars.index[5].date()
        # fill at bar5 open=90.0 (gap fill)
        assert t.exit_price == pytest.approx(90.0, abs=TOL)

    def test_gap_fills_false_fill_at_stop_ref(self):
        cfg = _cfg(hold=100, stop_loss=0.05, gap_fills=False, slippage_bps=0.0)
        out = simulate_ticker(self.bars, signal_idx=3, cfg=cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "stop"
        # fill at stop_ref=95.0 despite open=90.0
        assert t.exit_price == pytest.approx(95.0, abs=TOL)

    def test_layer_b_gap_fills_true(self):
        cfg = _cfg(hold=100, stop_loss=0.05, gap_fills=True, slippage_bps=0.0)
        out = simulate_ticker(self.bars, signal_idx=3, cfg=cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "stop",
            commission_bps=0.0,
        )
        assert tr.pnl == pytest.approx(-10_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(-0.10, abs=TOL)

    def test_layer_b_gap_fills_false(self):
        cfg = _cfg(hold=100, stop_loss=0.05, gap_fills=False, slippage_bps=0.0)
        out = simulate_ticker(self.bars, signal_idx=3, cfg=cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "stop",
            commission_bps=0.0,
        )
        assert tr.pnl == pytest.approx(-5_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(-0.05, abs=TOL)


# ===========================================================================
# Scenario 5 — Gap-UP through target: gap_fills=True → fill at open
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; take_profit=0.10 → target_ref=110.0
# bar5: open=115.0 ≥ target_ref=110.0 → gap condition true
#   gap_fills=True: _resolve_target_fill(115.0, 110.0, True) → 115.0 (fill at open)
#
# Layer B: exit_value=115_000, pnl=15_000, return_pct=0.15


class TestS5GapUp:
    def setup_method(self):
        self.bars = bars_s5_gap_up()
        self.cfg = _cfg(hold=100, take_profit=0.10, gap_fills=True, slippage_bps=0.0)

    def test_layer_a_fill_at_open(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "target"
        assert t.exit_date == self.bars.index[5].date()
        # gap fill at open=115.0 (not at target_ref=110.0)
        assert t.exit_price == pytest.approx(115.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "target",
            commission_bps=0.0,
        )
        assert tr.pnl == pytest.approx(15_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(0.15, abs=TOL)


# ===========================================================================
# Scenario 6 — Trailing-stop ratchet (price rises then reverses)
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; trailing_stop=0.10
# Initial: peak = entry_fill = 100.0, initial_trail_ref = 90.0
#
# bar5 (i=5): Evaluation order in _check_exit_at_bar:
#   trail_ref = peak * (1 - 0.10) = 100.0 * 0.9 = 90.0
#   stop_hit=False, target_hit=False (no stop/target)
#   trail_hit: low=99.5 ≤ 90.0? No → no exit
#   THEN: high=120.0 > peak=100.0 → peak = 120.0
#
# bar6 (i=6): peak=120.0
#   trail_ref = 120.0 * 0.9 = 108.0
#   low=105.0 ≤ 108.0 → trail_hit=True
#   _resolve_stop_fill(bar6_open=119.0, trail_ref=108.0, gap_fills=True):
#     open=119.0 > trail_ref=108.0 → NOT a gap-through → fill at trail_ref=108.0
#
# Layer B: exit_value=108_000, pnl=8_000, return_pct=0.08


class TestS6TrailingStop:
    def setup_method(self):
        self.bars = bars_s6_trailing_stop()
        self.cfg = _cfg(
            hold=100, trailing_stop=0.10, slippage_bps=0.0, commission_bps=0.0
        )

    def test_layer_a_peak_and_exit(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "trail"
        assert t.exit_date == self.bars.index[6].date()

    def test_layer_a_fill_at_trail_ref(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        # peak lifted to 120 at bar5; trail_ref = 120*0.9 = 108.0
        # bar6 open=119 > trail_ref=108 → fill at trail_ref, not gap
        assert t.exit_price == pytest.approx(108.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "trail",
            commission_bps=0.0,
        )
        assert tr.shares == pytest.approx(1000.0, abs=TOL)
        assert tr.pnl == pytest.approx(8_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(0.08, abs=TOL)


# ===========================================================================
# Scenario 7 — Partial scale-out (50% at +10%) then time exit (via run_backtest)
# ===========================================================================
# as_of = bars.index[3].date() → signal_idx=3, entry_idx=4, open=100.0
# partial_exits=((0.10, 0.5),): partial target = entry_fill*1.10 = 110.0
# hold=5 → hold_limit_idx = 4 + 5 = 9
#
# _run_event_driven_sim calls _fire_partial_exits_at_bar before _check_exit_at_bar
#
# bar5 (i=5): high=115.0 ≥ partial_target=110.0; open=102.0 < 110.0
#   fill = _resolve_target_fill(102.0, 110.0, gap_fills=True) = 110.0  (open<target)
#   close_shares = 1000.0 * 0.5 = 500.0
#   pro_rata_cost = 100_000.0 * 0.5 = 50_000.0
#   exit_value_A = 500.0 * 110.0 = 55_000.0   (commission=0)
#   pnl_A = 55_000.0 - 50_000.0 = 5_000.0
#   Portfolio cash after partial = 0 + 55_000 = 55_000
#   Remaining: 500 shares, slot_capital = 50_000.0
#
# bar9 (i=9 ≥ hold_limit_idx=9): time exit at close=112.0
#   exit_value_B = 500.0 * 112.0 = 56_000.0
#   pnl_B = 56_000.0 - 50_000.0 = 6_000.0
#   Portfolio cash = 55_000 + 56_000 = 111_000.0
#
# Total pnl = 5_000 + 6_000 = 11_000


class TestS7PartialThenTime:
    def setup_method(self):
        self.bars = bars_s7_partial_then_time()
        # as_of must be bars.index[3]; first business day = 2024-01-01 (Mon), index[3]=2024-01-04
        self.as_of = self.bars.index[3].date()
        spy = make_spy_bars(len(self.bars))
        self.fetcher = StubPriceFetcher({"TEST": self.bars, "SPY": spy})
        self.cfg = _cfg(
            as_of=self.as_of,
            hold=5,
            partial_exits=((0.10, 0.5),),
            slippage_bps=0.0,
            commission_bps=0.0,
            gap_fills=True,
        )

    def test_layer_a_two_trades_emitted(self):
        result = run_backtest(self.cfg, self.fetcher)
        trades = result.trades
        assert len(trades) == 2, (
            f"expected 2 trades (partial + time), got {len(trades)}"
        )

    def test_layer_a_partial_exit_reason_and_date(self):
        result = run_backtest(self.cfg, self.fetcher)
        partial = next(t for t in result.trades if t.exit_reason == "target")
        assert partial.exit_date == self.bars.index[5].date()
        assert partial.exit_price == pytest.approx(110.0, abs=TOL)

    def test_layer_a_time_exit_reason_and_date(self):
        result = run_backtest(self.cfg, self.fetcher)
        time_t = next(t for t in result.trades if t.exit_reason == "time")
        assert time_t.exit_date == self.bars.index[9].date()
        assert time_t.exit_price == pytest.approx(112.0, abs=TOL)

    def test_layer_b_partial_pnl(self):
        result = run_backtest(self.cfg, self.fetcher)
        partial = next(t for t in result.trades if t.exit_reason == "target")
        assert partial.shares == pytest.approx(500.0, abs=TOL)
        assert partial.entry_cost == pytest.approx(50_000.0, abs=TOL)
        assert partial.exit_value == pytest.approx(55_000.0, abs=TOL)
        assert partial.pnl == pytest.approx(5_000.0, abs=TOL)

    def test_layer_b_time_pnl(self):
        result = run_backtest(self.cfg, self.fetcher)
        time_t = next(t for t in result.trades if t.exit_reason == "time")
        assert time_t.shares == pytest.approx(500.0, abs=TOL)
        assert time_t.entry_cost == pytest.approx(50_000.0, abs=TOL)
        assert time_t.exit_value == pytest.approx(56_000.0, abs=TOL)
        assert time_t.pnl == pytest.approx(6_000.0, abs=TOL)

    def test_layer_b_final_cash(self):
        result = run_backtest(self.cfg, self.fetcher)
        # Equity curve last value should reflect all cash received
        # final_equity = 100_000 - 100_000 (entry) + 55_000 (partial) + 56_000 (time) = 111_000
        assert result.equity_curve.iloc[-1] == pytest.approx(111_000.0, abs=TOL)


# ===========================================================================
# Scenario 8 — Time-based exit (max-hold-days)
# ===========================================================================
# signal_idx=3, entry at bar4 open=100.0; hold=3
# hold_limit_idx = 4 + 3 = 7
# Loop starts at i=5:
#   i=5: 5 >= 7? No  → continues
#   i=6: 6 >= 7? No  → continues
#   i=7: 7 >= 7? Yes → time exit at bar7 close=105.0
#
# Layer A: exit_reason='time', exit_date=bars.index[7], exit_price=105.0
# Layer B: shares=1000, pnl=5_000, return_pct=0.05


class TestS8TimeExit:
    def setup_method(self):
        self.bars = bars_s8_time_exit()
        self.cfg = _cfg(hold=3, slippage_bps=0.0, commission_bps=0.0)

    def test_layer_a_exit_reason_and_date(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        assert t.exit_reason == "time"
        assert t.exit_date == self.bars.index[7].date()

    def test_layer_a_fill_price(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        assert out.trade.exit_price == pytest.approx(105.0, abs=TOL)

    def test_layer_b_portfolio_mechanics(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "time",
            commission_bps=0.0,
        )
        assert tr.shares == pytest.approx(1000.0, abs=TOL)
        assert tr.entry_cost == pytest.approx(100_000.0, abs=TOL)
        assert tr.exit_value == pytest.approx(105_000.0, abs=TOL)
        assert tr.pnl == pytest.approx(5_000.0, abs=TOL)
        assert tr.return_pct == pytest.approx(0.05, abs=TOL)


# ===========================================================================
# Scenario 9 — Commission + slippage: verify exact shares / pnl
# ===========================================================================
# slippage_bps=10 (0.1%), commission_bps=0
# bar4 open = 100.5 → entry_ref = 100.5
#
# _apply_slip(100.5, 'buy', cfg):
#   entry_fill = 100.5 * (1 + 10/10_000) = 100.5 * 1.001 = 100.6005
#
# Portfolio.open(entry_price=100.6005, commission_bps=0):
#   gross_per_share = 100.6005 * (1 + 0) = 100.6005
#   budget = min(100_000, 100_000) = 100_000
#   shares = 100_000 / 100.6005 = 994.0308447771...
#   entry_cost = shares * 100.6005 = 100_000.0  (exactly by construction)
#
# Plan says shares ≈ 994.0308 (matches: 100_000 / (100.5 * 1.001) = 994.0308...)
#
# take_profit=0.185 → target_ref = 100.6005 * 1.185 = 119.2115925...
# bar5: open=119.4 ≥ target_ref=119.2116 → gap fill
#   _resolve_target_fill(119.4, 119.2116, gap_fills=True) → 119.4
#   exit_fill = _apply_slip(119.4, 'sell') = 119.4 * (1 - 10/10_000)
#             = 119.4 * 0.999 = 119.2806
#
# Portfolio.close(exit_price=119.2806, commission_bps=0):
#   proceeds   = 994.0308... * 119.2806 = 118_568.5956...
#   commission = 0
#   exit_value = 118_568.5956...
#   pnl        = 118_568.5956... - 100_000.0 = 18_568.5956...
#   return_pct = 18_568.5956... / 100_000.0 = 0.185685956...
#
# Plan's claimed values: shares=994.0308, pnl=18568.5956
# This test verifies our reading matches (and reports discrepancies if any).


class TestS9CommissionSlippage:
    # Exact hand-computed constants (derived above)
    ENTRY_REF = 100.5
    SLIPPAGE_BPS = 10.0
    ENTRY_FILL = 100.5 * 1.001  # = 100.6005
    SHARES = 100_000 / (100.5 * 1.001)  # = 994.030844...
    TAKE_PROFIT = 0.185
    # target_ref = ENTRY_FILL * 1.185
    GAP_OPEN = 119.4
    EXIT_FILL = 119.4 * 0.999  # = 119.2806
    EXIT_VALUE = SHARES * (119.4 * 0.999)
    PNL = EXIT_VALUE - 100_000.0
    RETURN_PCT = PNL / 100_000.0

    def setup_method(self):
        self.bars = bars_s9_commission_slippage()
        self.cfg = _cfg(
            hold=100,
            take_profit=self.TAKE_PROFIT,
            slippage_bps=self.SLIPPAGE_BPS,
            commission_bps=0.0,
            gap_fills=True,
        )

    def test_layer_a_entry_fill_with_slippage(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t is not None
        # entry_fill = 100.5 * 1.001 = 100.6005
        assert t.entry_price == pytest.approx(self.ENTRY_FILL, abs=TOL)

    def test_layer_a_gap_fill_target(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        t = out.trade
        assert t.exit_reason == "target"
        assert t.exit_date == self.bars.index[5].date()
        # exit_fill = 119.4 * 0.999 = 119.2806 (gap fill at open, then sell slippage)
        assert t.exit_price == pytest.approx(self.EXIT_FILL, abs=TOL)

    def test_layer_b_shares_match_plan(self):
        """Verify shares = 100_000 / (100.5 * 1.001) ≈ 994.0308 as plan states."""
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        pos = port.get_position("TEST")
        assert pos is not None
        # Plan's claimed value is 994.0308; our exact value is 994.030844...
        assert pos.shares == pytest.approx(self.SHARES, abs=TOL)
        # Confirm it's approximately 994.0308 (matches plan's claim to 4dp)
        assert abs(pos.shares - 994.0308) < 0.0001

    def test_layer_b_pnl_match_plan(self):
        """Verify pnl ≈ 18568.5956 as stated in the plan."""
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "target",
            commission_bps=0.0,
        )
        # Exact derived value
        assert tr.pnl == pytest.approx(self.PNL, abs=TOL)
        # Plan's claim: 18568.5956 — verify within 4 decimal places
        assert abs(tr.pnl - 18568.5956) < 0.001

    def test_layer_b_entry_cost_exact(self):
        out = simulate_ticker(self.bars, signal_idx=3, cfg=self.cfg)
        port = Portfolio(100_000.0, slot_count=1)
        port.open(
            "TEST", out.trade.entry_date, out.trade.entry_price, commission_bps=0.0
        )
        tr = port.close(
            "TEST",
            out.trade.exit_date,
            out.trade.exit_price,
            "target",
            commission_bps=0.0,
        )
        # entry_cost = budget = 100_000 (no commission)
        assert tr.entry_cost == pytest.approx(100_000.0, abs=TOL)
