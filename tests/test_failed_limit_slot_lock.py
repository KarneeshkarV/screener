"""Regressions: triggered limit orders that fail to open must free their slot.

Previously ``fill_pending_entries`` skipped deleting a failed order whenever
``cfg.entry_order_type == "limit"``, and the India pre-queue whole-share checks
skipped limit orders altogether. A triggered limit order that failed (e.g.
whole-share unaffordable in India, ``exit_ast`` raising ``PineError``, or
``intraday_only`` skipping a session-last bar) permanently locked its slot
until the ticker's history ended and starved lower-ranked candidates / reserves.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from screener.backtester.book import fill_pending_entries, open_book
from screener.backtester.core import PendingEntryOrder, _RunCaches
from screener.backtester.historical import run_backtest
from screener.backtester.models import BacktestConfig
from screener.backtester.pine import PineError, parse
from screener.backtester.rolling_simulation import (
    prepare_rolling_backtest,
    run_prepared_rolling_backtest,
    run_rolling_backtest,
)
from tests.conftest import StubPriceFetcher


def _bars(
    index: pd.DatetimeIndex,
    *,
    close: float | np.ndarray,
    volume: float,
    low: float | np.ndarray | None = None,
    open_: float | np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(index)
    close_arr = (
        np.full(n, float(close))
        if np.isscalar(close)
        else np.asarray(close, dtype=float)
    )
    open_arr = (
        close_arr.copy()
        if open_ is None
        else (
            np.full(n, float(open_))
            if np.isscalar(open_)
            else np.asarray(open_, dtype=float)
        )
    )
    low_arr = (
        close_arr * 0.90
        if low is None
        else (
            np.full(n, float(low)) if np.isscalar(low) else np.asarray(low, dtype=float)
        )
    )
    high_arr = np.maximum(open_arr, close_arr) + 1.0
    return pd.DataFrame(
        {
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": np.full(n, float(volume)),
        },
        index=index,
    )


def test_rolling_india_limit_unaffordable_rank1_fills_rank2() -> None:
    """An unaffordable India rank-1 limit candidate must not starve rank-2."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    panel = {
        "UNAFF.NS": _bars(dates, close=20_000.0, low=18_000.0, volume=100_000.0),
        "AFF.NS": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[-1].date(),
        benchmark="^NSEI",
        tickers=("UNAFF", "AFF"),
        entry_expr="close > 0",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(panel),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )

    assert result.trades, "expected rank-2 affordable ticker to fill"
    assert {trade.ticker for trade in result.trades} == {"AFF"}


def test_rolling_india_limit_borderline_affordable_at_limit_price_fills() -> None:
    """Pre-queue affordability uses the causal limit price, not the signal close."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    # Close is 10_200 (> 10_000 budget), but 5% limit price is 9_690 (<= 10_000).
    panel = {
        "BORDER.NS": _bars(dates, close=10_200.0, low=9_500.0, volume=100_000.0),
        "CHEAP.NS": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[-1].date(),
        benchmark="^NSEI",
        tickers=("BORDER", "CHEAP"),
        entry_expr="close > 0",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(panel),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )

    assert result.trades
    assert result.trades[0].ticker == "BORDER"
    assert result.trades[0].shares == 1.0


def test_rolling_india_affordable_untouched_limit_still_waits_causally() -> None:
    """Causality preserved: an affordable untouched India limit still waits."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    panel = {
        "AAA.NS": _bars(dates, close=100.0, low=95.0, volume=100_000.0),
        "BBB.NS": _bars(dates, close=100.0, low=87.0, volume=10_000.0),
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[-1].date(),
        benchmark="^NSEI",
        tickers=("AAA", "BBB"),
        entry_expr="close > 50",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=1_000.0,
        entry_order_type="limit",
        entry_limit_bps=1200.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(panel),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )

    assert result.trades == []


def test_rolling_india_limit_becomes_unaffordable_at_fill_frees_slot() -> None:
    """A limit affordable at queue time that fails affordability at fill frees its slot."""
    dates = pd.bdate_range("2024-01-02", periods=10)

    # LOSSY (rank 1): fills Slot 0 on Day 1 at 95, gaps down to 10 on Day 2 and
    # exits on stop_loss, cutting realized equity from 20_000 to ~11_075
    # (per-slot budget ~5_537). Slot 0 is then occupied by CHEAP1 for hold=20.
    lossy_close = np.full(len(dates), 100.0)
    lossy_close[2:] = 10.0
    lossy_low = np.full(len(dates), 90.0)
    lossy_low[2:] = 10.0

    # MID (rank 2): close=9_000 -> limit=8_550 (affordable on Day 0 when slot budget
    # is 10_000). Does not touch 8_550 until Day 3, after LOSSY's loss reduced the
    # per-slot budget below 8_550.
    mid_low = np.full(len(dates), 8_900.0)
    mid_low[3:] = 8_000.0

    # CHEAP1 (rank 3) takes Slot 0 after LOSSY exits; CHEAP2 (rank 4) can ONLY
    # fill if MID's failed triggered limit actually frees Slot 1 on Day 3.
    panel = {
        "LOSSY.NS": _bars(dates, close=lossy_close, low=lossy_low, volume=5_000_000.0),
        "MID.NS": _bars(dates, close=9_000.0, low=mid_low, volume=20_000.0),
        "CHEAP1.NS": _bars(dates, close=100.0, low=90.0, volume=15_000.0),
        "CHEAP2.NS": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=10_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[-1].date(),
        benchmark="^NSEI",
        tickers=("LOSSY", "MID", "CHEAP1", "CHEAP2"),
        entry_expr="close >= 50",
        exit_expr=None,
        hold=20,
        stop_loss=0.5,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=2,
        initial_capital=20_000.0,
        compounding=True,
        sizing_rule="equal_slot",
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(panel),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )

    traded = {trade.ticker for trade in result.trades}
    assert "MID" not in traded
    assert {"CHEAP1", "CHEAP2"}.issubset(traded)
    cheap2 = next(t for t in result.trades if t.ticker == "CHEAP2")
    assert cheap2.entry_date == dates[4].date()


def test_rolling_limit_exit_ast_pine_error_at_fill_frees_slot(monkeypatch) -> None:
    """When exit_ast raises PineError on a triggered limit, the slot is freed for rank-2."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    panel = {
        "AAA": _bars(dates, close=100.0, low=90.0, volume=100_000.0),
        "BBB": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
        "SPY": _bars(dates, close=100.0, low=99.0, volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=dates[-1].date(),
        benchmark="SPY",
        tickers=("AAA", "BBB"),
        entry_expr="close > 0",
        exit_expr="close < 0",
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    # Clear prewarmed exit_signals so AAA is genuinely queued as a pending limit
    # order on Day 0 and hits PineError inside _make_slot_state when triggered.
    prepared = prepare_rolling_backtest(
        cfg,
        StubPriceFetcher(panel),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )
    prepared = replace(prepared, exit_signals={})

    import screener.backtester.core as core_mod

    real_evaluate = core_mod.evaluate

    def failing_evaluate(node, bars):
        if float(bars["volume"].iloc[0]) == 100_000.0:
            raise PineError("broken exit for AAA")
        return real_evaluate(node, bars)

    monkeypatch.setattr(core_mod, "evaluate", failing_evaluate)

    result = run_prepared_rolling_backtest(prepared, cfg)

    assert result.trades, "rank-2 BBB should trade once AAA's failed limit frees slot 0"
    assert {trade.ticker for trade in result.trades} == {"BBB"}
    assert any(
        "AAA: exit eval failed: broken exit for AAA" in w for w in result.warnings
    )


def test_rolling_limit_prewarmed_exit_ast_failure_does_not_starve_rank2() -> None:
    """Prewarmed exit_ast PineError on rank-1 must warn and let rank-2 trade."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    aaa = _bars(dates, close=100.0, low=90.0, volume=100_000.0)
    bbb = _bars(dates, close=100.0, low=90.0, volume=10_000.0)
    bbb["custom_flag"] = 0.0
    spy = _bars(dates, close=100.0, low=99.0, volume=1_000_000.0)

    cfg = BacktestConfig(
        market="us",
        as_of=dates[-1].date(),
        benchmark="SPY",
        tickers=("AAA", "BBB"),
        entry_expr="close > 0",
        exit_expr="custom_flag > 0",
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": aaa, "BBB": bbb, "SPY": spy}),
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
    )

    assert result.trades
    assert {trade.ticker for trade in result.trades} == {"BBB"}
    exit_warnings = [
        w for w in result.warnings if w.startswith("AAA: exit eval failed:")
    ]
    assert len(exit_warnings) == 1


def test_historical_india_limit_unaffordable_active_and_reserve_promotes_affordable() -> (
    None
):
    """Unaffordable India active and reserve limit orders must not starve later reserves."""
    n = 30
    dates = pd.bdate_range("2024-01-01", periods=n)
    as_of_i = 20

    # Dollar volume ranks UNAFFACT > UNAFFRES > AFFRES.
    # UNAFFACT has untouched lows so it would deadlock forever if queued;
    # UNAFFRES has touched lows and sits ahead of AFFRES in reserve_queue.
    panel = {
        "UNAFFACT.NS": _bars(dates, close=20_000.0, low=19_900.0, volume=50_000.0),
        "UNAFFRES.NS": _bars(dates, close=15_000.0, low=13_000.0, volume=20_000.0),
        "AFFRES.NS": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=10_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[as_of_i].date(),
        benchmark="^NSEI",
        tickers=("UNAFFACT", "UNAFFRES", "AFFRES"),
        entry_expr="close > 0",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        top=1,
        strategy_name=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        reinvest=True,
        compounding=False,
        reserve_multiple=3,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_backtest(cfg, StubPriceFetcher(panel))

    assert result.trades, "affordable reserve must fill instead of deadlocking"
    assert {trade.ticker for trade in result.trades} == {"AFFRES"}


def test_historical_india_limit_triggered_reserve_failure_demotes_to_back() -> None:
    """A reserve limit that fails affordability at trigger is demoted to the back of the queue."""
    n = 30
    dates = pd.bdate_range("2024-01-01", periods=n)
    as_of_i = 15

    # Active 1 (FAST, rank 1): fills Slot 0 on Day as_of+1 at 95, hits take_profit
    # on Day as_of+2 while realized equity is > 20_000 (per-slot budget > 10_000).
    fast_close = np.full(n, 100.0)
    fast_close[as_of_i + 2 :] = 120.0
    fast_low = np.full(n, 90.0)
    fast_low[as_of_i + 2 :] = 115.0
    fast = _bars(dates, close=fast_close, low=fast_low, volume=5_000_000.0)

    # Active 2 (LOSSY, rank 2): fills Slot 1 on Day as_of+2 at 95, gaps down to 10
    # on Day as_of+3 and exits on stop_loss, cutting per-slot budget below 8_550.
    lossy_close = np.full(n, 100.0)
    lossy_close[as_of_i + 3 :] = 10.0
    lossy_low = np.full(n, 90.0)
    lossy_low[as_of_i + 1] = 99.0
    lossy_low[as_of_i + 3 :] = 10.0
    lossy = _bars(dates, close=lossy_close, low=lossy_low, volume=4_000_000.0)

    # Reserve 1 (MIDRES, rank 3): close=9_000 -> limit=8_550. Affordable when FAST
    # exits Slot 0 on Day as_of+2, so MIDRES is queued into Slot 0. Does not touch
    # 8_550 until Day as_of+4, AFTER LOSSY's loss shrank per-slot budget below 8_550.
    mid_low = np.full(n, 8_900.0)
    mid_low[as_of_i + 4 :] = 8_000.0
    mid_res = _bars(dates, close=9_000.0, low=mid_low, volume=30_000.0)

    # Reserve 2 (AFF1, rank 4) takes Slot 1 when LOSSY exits (and holds for 20 bars);
    # Reserve 3 (AFF2, rank 5) can ONLY fill if MIDRES is dropped from Slot 0 and
    # demoted to the back of reserve_queue on Day as_of+4.
    aff_1 = _bars(dates, close=100.0, low=90.0, volume=20_000.0)
    aff_2 = _bars(dates, close=100.0, low=90.0, volume=10_000.0)

    panel = {
        "FAST.NS": fast,
        "LOSSY.NS": lossy,
        "MIDRES.NS": mid_res,
        "AFF1.NS": aff_1,
        "AFF2.NS": aff_2,
        "^NSEI": _bars(dates, close=100.0, low=99.0, volume=10_000_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[as_of_i].date(),
        benchmark="^NSEI",
        tickers=("FAST", "LOSSY", "MIDRES", "AFF1", "AFF2"),
        entry_expr="close >= 50",
        exit_expr=None,
        hold=20,
        stop_loss=0.5,
        take_profit=0.1,
        trailing_stop=None,
        top=2,
        strategy_name=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=20_000.0,
        reinvest=True,
        compounding=True,
        reserve_multiple=3,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )

    result = run_backtest(cfg, StubPriceFetcher(panel))

    traded = {trade.ticker for trade in result.trades}
    assert "MIDRES" not in traded
    assert {"AFF1", "AFF2"}.issubset(traded)
    aff2 = next(t for t in result.trades if t.ticker == "AFF2")
    assert aff2.entry_date == dates[as_of_i + 5].date()


def test_fill_pending_entries_drops_failed_triggered_limit_but_keeps_untouched() -> (
    None
):
    """Direct unit check on fill_pending_entries limit failure vs untouched wait."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    bars_by_ticker = {
        "UNTOUCHED": _bars(dates, close=100.0, low=98.0, volume=10_000.0),
        "UNAFF": _bars(dates, close=20_000.0, low=18_000.0, volume=10_000.0),
        "BADEXIT": _bars(dates, close=100.0, low=90.0, volume=10_000.0),
    }
    cfg = BacktestConfig(
        market="india",
        as_of=dates[-1].date(),
        benchmark="^NSEI",
        tickers=("UNTOUCHED", "UNAFF", "BADEXIT"),
        entry_expr="close > 0",
        exit_expr="missing_col > 0",
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=3,
        initial_capital=30_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )
    book = open_book(cfg)
    pending = {
        0: PendingEntryOrder("UNTOUCHED", 0, 1, dates[0].date()),
        1: PendingEntryOrder("UNAFF", 0, 2, dates[0].date()),
        2: PendingEntryOrder("BADEXIT", 0, 3, dates[0].date()),
    }
    warnings: list[str] = []

    filled = fill_pending_entries(
        day=dates[1],
        at_close=False,
        pending_orders=pending,
        bars_by_ticker=bars_by_ticker,
        cfg=cfg,
        portfolio=book.portfolio,
        slot_states=book.slot_states,
        slot_bars=book.slot_bars,
        fill_model=book.fill_model,
        caches=_RunCaches(),
        exit_ast=parse("missing_col > 0"),
        warnings=warnings,
    )

    assert filled == []
    # Untouched limit stays pending; both triggered failures are dropped.
    assert list(pending.keys()) == [0]
    assert pending[0].ticker == "UNTOUCHED"
    assert any("BADEXIT: exit eval failed:" in w for w in warnings)


def test_fill_pending_entries_drops_intraday_only_session_last_limit() -> None:
    """A limit touched on an intraday session-last bar is dropped instead of locking."""
    stamps = pd.DatetimeIndex(
        [
            "2024-01-02 14:30:00",
            "2024-01-02 15:30:00",
            "2024-01-03 09:30:00",
        ]
    )
    frame = _bars(stamps, close=100.0, low=90.0, volume=10_000.0)
    cfg = BacktestConfig(
        market="us",
        interval="1h",
        intraday_only=True,
        as_of=stamps[-1].date(),
        benchmark="SPY",
        tickers=("AAA",),
        entry_expr="close > 0",
        exit_expr=None,
        hold=2,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=1,
        initial_capital=10_000.0,
        entry_order_type="limit",
        entry_limit_bps=500.0,
        min_price=0.0,
        min_avg_dollar_volume=0.0,
    )
    book = open_book(cfg)
    pending = {
        0: PendingEntryOrder("AAA", 0, 1, stamps[0].to_pydatetime()),
    }
    warnings: list[str] = []

    filled = fill_pending_entries(
        day=stamps[1],
        at_close=False,
        pending_orders=pending,
        bars_by_ticker={"AAA": frame},
        cfg=cfg,
        portfolio=book.portfolio,
        slot_states=book.slot_states,
        slot_bars=book.slot_bars,
        fill_model=book.fill_model,
        caches=_RunCaches(market_index=stamps),
        exit_ast=None,
        warnings=warnings,
    )

    assert filled == []
    assert pending == {}
    assert any("session-last bar (intraday-only)" in w for w in warnings)
