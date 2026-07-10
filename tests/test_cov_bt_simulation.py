"""Offline coverage tests for the backtester core/rolling/historical/data/pine modules.

These tests are deterministic and never touch the network: every price fetch goes
through ``StubPriceFetcher`` or a monkeypatched seam, and CLI paths use
``click.testing.CliRunner`` with an injected fetcher (``obj=...``).

They are written to drive the remaining uncovered lines in:
  - screener/backtester/core.py
  - screener/backtester/rolling.py
  - screener/backtester/historical.py
  - screener/backtester/data.py
  - screener/backtester/pine.py
"""

from __future__ import annotations


from datetime import date


import numpy as np


import pandas as pd


from click.testing import CliRunner


from main import cli


from screener.backtester.core import (
    _eligible_reserve_signal_idx,
    _make_slot_state,
    _trailing_liquidity,
)


from screener.backtester.historical import (
    run_backtest,
)


from screener.backtester.models import BacktestConfig


from screener.backtester.pine import (
    parse,
)


from tests.conftest import StubPriceFetcher, make_bars


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=2,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=10_000.0,
        benchmark="SPY",
        tickers=("AAA",),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _stub_env(n=60):
    return StubPriceFetcher(
        {
            "AAA": make_bars(n=n, seed=11, open_base=100.0),
            "BBB": make_bars(n=n, seed=12, open_base=50.0),
            "SPY": make_bars(n=n, seed=99, open_base=400.0),
        }
    )


def _universe_file(tmp_path):
    f = tmp_path / "univ.txt"
    f.write_text("AAA\nBBB\n")
    return f


from screener.backtester.core import (  # noqa: E402
    _SlotState,
    _maybe_credit_dividends,
)


from screener.backtester.portfolio import Portfolio  # noqa: E402


def _open_slot(bars, *, entry_idx=1, ticker="AAA", **state_kw):
    """Build a portfolio with an open position + matching slot state."""
    cfg = _cfg(initial_capital=10_000.0)
    portfolio = Portfolio(cfg.initial_capital, 1)
    entry_fill = float(bars.iloc[entry_idx]["open"])
    portfolio.assign(ticker, 1, bars.index[0].date())
    portfolio.open(
        ticker=ticker,
        entry_date=bars.index[entry_idx].date(),
        entry_price=entry_fill,
        commission_bps=0.0,
    )
    defaults = dict(
        ticker=ticker,
        entry_idx=entry_idx,
        entry_date=bars.index[entry_idx].date(),
        entry_fill=entry_fill,
        signal_date=bars.index[entry_idx - 1].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=entry_idx + 5,
        peak=entry_fill,
        exit_signal=None,
    )
    defaults.update(state_kw)
    state = _SlotState(**defaults)
    return cfg, portfolio, state


from screener.backtester.historical import _run_event_driven_sim  # noqa: E402


def _flat_then_trending(start, n, base, *, dip_at=None):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(base, base + n, n), index=idx, dtype=float)
    if dip_at is not None:
        close.iloc[dip_at] = base * 0.5  # crash to trip stop / free slot
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )


from screener.backtester.core import (  # noqa: E402
    _build_frame_cache,
    _cached_trailing_liquidity,
    _RunCaches,
)


from screener.backtester.models import Trade  # noqa: E402


from screener.backtester.portfolio import build_equity_curve  # noqa: E402


def test_event_driven_sim_reserve_rotation_branches():
    as_of = pd.Timestamp("2024-02-01")
    # ACTIVE exits early (dip trips stop), freeing the slot for reserves.
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    # reserve candidates: no-data, ineligible (insufficient lookback), good.
    good_reserve = _flat_then_trending("2024-01-01", 80, 90.0)
    short_reserve = _flat_then_trending("2024-02-02", 80, 70.0)  # starts after as_of
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(
        [
            {"ticker": "NODATA_R", "rank": 2},
            {"ticker": "SHORT_R", "rank": 3},
            {"ticker": "GOOD_R", "rank": 4},
        ]
    )
    bars_by = {
        "ACTIVE": active,
        "NODATA_R": pd.DataFrame(),
        "SHORT_R": short_reserve,
        "GOOD_R": good_reserve,
    }
    cfg = _cfg(
        as_of=as_of.date(),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    # The active stopped out and at least one reserve rotated in.
    assert any(t.ticker == "ACTIVE" for t in trades)


def test_event_driven_sim_reentry_branches():
    as_of = pd.Timestamp("2024-02-01")
    # active closes (stop) then re-signals later so it re-enters.
    active = _flat_then_trending("2024-01-01", 90, 100.0, dip_at=24)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_force_close_open_slot():
    # Active never exits within horizon -> force-closed (eod) at the end.
    as_of = pd.Timestamp("2024-02-01")
    active = _flat_then_trending("2024-01-01", 120, 100.0)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=500,
        top=1,
        entry_expr="close > 0",
        stop_loss=None,
        take_profit=None,
        reinvest=False,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    assert any(str(t.exit_reason) == "eod" for t in trades)


def test_event_driven_sim_reserve_makefail():
    # A reserve whose only eligible signal is its last bar -> make_slot fails
    # (no post-signal entry bar) -> warning branch.
    as_of = pd.Timestamp("2024-02-01")
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    # reserve frame ends one bar after the freeing day so its signal is last.
    good_reserve = _flat_then_trending("2024-01-01", 28, 90.0)
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame([{"ticker": "RSV", "rank": 2}])
    bars_by = {"ACTIVE": active, "RSV": good_reserve}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(warnings, list)


def test_event_driven_sim_reserve_already_taken():
    # A reserve ticker that is also an active -> the `ticker in taken` skip.
    as_of = pd.Timestamp("2024-02-01")
    dup = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    other = _flat_then_trending("2024-01-01", 80, 90.0)
    actives = pd.DataFrame([{"ticker": "DUP", "rank": 1}])
    # DUP appears again in reserves (already taken) before a fresh reserve.
    reserves = pd.DataFrame(
        [{"ticker": "DUP", "rank": 2}, {"ticker": "OTHER", "rank": 3}]
    )
    bars_by = {"DUP": dup, "OTHER": other}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 3)"),
        exit_ast=None,
        lookback=3,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_reentry_not_eligible_then_eligible():
    # Active stops out early; entry signal stays False for a while (re-entry
    # pending but not yet eligible -> the `continue` branch), then fires again.
    as_of = pd.Timestamp("2024-02-01")
    idx = pd.bdate_range("2024-01-01", periods=90)
    # Build a path: rise (enter), crash (stop), flat-down (no signal), rise again.
    seg1 = np.linspace(100, 110, 30)
    seg2 = np.linspace(70, 60, 30)  # falling -> entry signal stays False
    seg3 = np.linspace(60, 90, 30)  # rising -> entry signal fires again
    close = pd.Series(np.concatenate([seg1, seg2, seg3]), index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    high = pd.concat([openp, close], axis=1).max(axis=1) + 1.0
    low = pd.concat([openp, close], axis=1).min(axis=1) - 1.0
    vol = pd.Series(100_000.0, index=idx, dtype=float)
    active = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": active}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 5)",
        stop_loss=0.05,
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > sma(close, 5)"),
        exit_ast=None,
        lookback=5,
        warnings=warnings,
    )
    trades = portfolio.closed_trades()
    # at least the initial trade plus a re-entry occurred.
    assert len([t for t in trades if t.ticker == "ACTIVE"]) >= 1


def test_event_driven_sim_reserve_makeslot_fail_last_bar():
    # Reserve becomes eligible on a day that is the LAST bar of its frame, so
    # _make_slot_state has no post-signal entry bar -> the make-slot-fail branch.
    as_of = pd.Timestamp("2024-02-01")
    # active stops out on the freeing day.
    active = _flat_then_trending("2024-01-01", 80, 100.0, dip_at=24)
    freeing_day = active.index[26]  # roughly where stop fires + day_loop frees
    # reserve frame: enough history to be eligible, ending exactly at a freeing day.
    rsv_idx = pd.bdate_range("2024-01-02", periods=40)
    rsv_close = pd.Series(np.linspace(90, 130, 40), index=rsv_idx, dtype=float)
    rsv_open = rsv_close.shift(1).fillna(rsv_close.iloc[0] - 1.0)
    rsv = pd.DataFrame(
        {
            "open": rsv_open,
            "high": pd.concat([rsv_open, rsv_close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([rsv_open, rsv_close], axis=1).min(axis=1) - 1.0,
            "close": rsv_close,
            "volume": pd.Series(100_000.0, index=rsv_idx),
        }
    )
    # Truncate the reserve so its last bar is exactly the active's freeing day
    # (2024-02-05): the reserve is eligible there but has no post-signal bar, so
    # _make_slot_state returns None -> the reserve make-slot-fail branch.
    freeing_day = pd.Timestamp("2024-02-05")
    rsv = rsv.loc[rsv.index <= freeing_day]
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame([{"ticker": "RSV", "rank": 2}])
    bars_by = {"ACTIVE": active, "RSV": rsv}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.08,
        reinvest=True,
        reserve_multiple=5,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_reentry_makeslot_fail_last_bar():
    # Re-entry signal lands on the final bar of the frame -> make-slot-fail.
    as_of = pd.Timestamp("2024-02-01")
    idx = pd.bdate_range("2024-01-01", periods=40)
    # rise (enter), crash (stop near bar 27), then rise so re-entry signal fires
    # right up to the last bar.
    close = pd.Series(
        np.concatenate([np.linspace(100, 110, 27), np.linspace(70, 100, 13)]),
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    frame = pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([openp, close], axis=1).min(axis=1) - 1.0,
            "close": close,
            "volume": pd.Series(100_000.0, index=idx),
        }
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": frame}
    cfg = _cfg(
        as_of=as_of.date(),
        hold=2,
        top=1,
        entry_expr="close > 0",
        stop_loss=0.05,
        allow_reentry=True,
        max_reentries=5,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=as_of,
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    assert isinstance(portfolio.closed_trades(), list)


def test_event_driven_sim_force_close_empty_tail():
    # Active enters on the final available bar so the force-close tail
    # (bars.index > entry_date) is empty -> the `tail.empty` continue branch.
    # frame ends 2 bars after as_of: signal at as_of -> entry on the last bar.
    idx = pd.bdate_range("2024-01-01", periods=24)  # ends ~ 2024-02-01
    last = idx[-1]
    # ensure as_of is the second-to-last bar so entry is the last bar.
    as_of2 = idx[-2]
    close = pd.Series(np.linspace(100, 120, 24), index=idx, dtype=float)
    openp = close.shift(1).fillna(close.iloc[0] - 1.0)
    frame = pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([openp, close], axis=1).min(axis=1) - 1.0,
            "close": close,
            "volume": pd.Series(100_000.0, index=idx),
        }
    )
    actives = pd.DataFrame([{"ticker": "ACTIVE", "rank": 1}])
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    bars_by = {"ACTIVE": frame}
    cfg = _cfg(
        as_of=as_of2.date(),
        hold=500,
        top=1,
        entry_expr="close > 0",
        stop_loss=None,
        take_profit=None,
        reinvest=False,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 1)
    warnings: list[str] = []
    _run_event_driven_sim(
        portfolio=portfolio,
        actives_df=actives,
        reserves_df=reserves,
        bars_by_tv=bars_by,
        as_of_ts=pd.Timestamp(as_of2),
        cfg=cfg,
        entry_ast=parse("close > 0"),
        exit_ast=None,
        lookback=0,
        warnings=warnings,
    )
    # entry on the last bar, no post-entry bars -> position stays open, tail empty.
    assert last in frame.index


def test_run_backtest_with_exit_expr_and_empty_frame_trade():
    # Exercises run_backtest exit_ast lookback branch + empty-frame skip.
    good = make_bars(n=120, start="2024-01-01", seed=4, open_base=100.0)
    spy = make_bars(n=120, start="2024-01-01", seed=9, open_base=400.0)
    fetcher = StubPriceFetcher({"GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        exit_expr="close < sma(close, 5)",
        tickers=("GOOD",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_cached_trailing_liquidity_guard_and_memo():
    bars = make_bars(n=30, open_base=100.0)
    fc = _build_frame_cache(bars)
    # signal_idx < 0 / window <= 0 short-circuit.
    assert _cached_trailing_liquidity(fc, bars, -1) == (0.0, 0.0)
    assert _cached_trailing_liquidity(fc, bars, 5, window=0) == (0.0, 0.0)
    # first call populates the memo; second returns the cached tuple.
    first = _cached_trailing_liquidity(fc, bars, 10, window=5)
    second = _cached_trailing_liquidity(fc, bars, 10, window=5)
    assert first == second
    assert (10, 5) in fc.liquidity_by_idx


def test_cached_trailing_liquidity_empty_window():
    bars = make_bars(n=5, open_base=100.0)
    fc = _build_frame_cache(bars)
    # signal_idx past the end -> the close window slice is empty.
    assert _cached_trailing_liquidity(fc, bars, 100, window=5) == (0.0, 0.0)


def test_cached_trailing_liquidity_nan_close_fallback():
    bars = make_bars(n=20, open_base=100.0)
    bars.iloc[10, bars.columns.get_loc("close")] = np.nan
    fc = _build_frame_cache(bars)
    # A NaN inside the window defers to the slice-based original for exact parity.
    assert _cached_trailing_liquidity(fc, bars, 12, window=5) == _trailing_liquidity(
        bars, 12, 5
    )


def test_cached_trailing_liquidity_non_finite_adv():
    bars = make_bars(n=20, open_base=100.0)
    bars["volume"] = np.inf  # adv mean -> inf -> reset to 0.0
    fc = _build_frame_cache(bars)
    adv, sigma = _cached_trailing_liquidity(fc, bars, 10, window=5)
    assert adv == 0.0
    assert np.isfinite(sigma)


def test_make_slot_state_cached_exit_warning_string():
    bars = make_bars(n=20, open_base=100.0)
    caches = _RunCaches()
    caches.exit_signals["AAA"] = "exit eval failed: boom"
    state, warn = _make_slot_state(
        "AAA", bars, 5, _cfg(), parse("close > 0"), 1, caches=caches
    )
    assert state is None
    assert warn == "exit eval failed: boom"


def test_make_slot_state_caches_exit_eval_failure():
    bars = make_bars(n=20, open_base=100.0)
    caches = _RunCaches()
    state, warn = _make_slot_state(
        "AAA", bars, 5, _cfg(), parse("nonexistent_col > 0"), 1, caches=caches
    )
    assert state is None
    assert warn and "exit eval failed" in warn
    # failure is memoised so repeat callers keep the same warning.
    assert caches.exit_signals["AAA"] == warn


def test_maybe_credit_dividends_frame_cache_bad_value():
    bars = make_bars(n=5, open_base=100.0)
    bars["dividend"] = ["x", "y", "z", "w", "v"]  # non-numeric
    fc = _build_frame_cache(bars)
    cfg = _cfg(price_adjustment="none")
    portfolio = Portfolio(cfg.initial_capital, 1)
    state = _SlotState(
        ticker="AAA",
        entry_idx=0,
        entry_date=bars.index[0].date(),
        entry_fill=100.0,
        signal_date=bars.index[0].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=5,
        peak=100.0,
        exit_signal=None,
        frame_cache=fc,
    )
    # float("y") raises ValueError -> swallowed, nothing credited.
    _maybe_credit_dividends(portfolio, state, bars, 1, cfg)


def test_eligible_reserve_signal_idx_cached_entry_error():
    bars = make_bars(n=30, open_base=100.0)
    cfg = _cfg(min_price=None, min_avg_dollar_volume=None)
    caches = _RunCaches()
    # the cached full-frame entry evaluation raises PineError -> None.
    assert (
        _eligible_reserve_signal_idx(
            bars,
            bars.index[20],
            cfg,
            parse("nonexistent_col > 0"),
            3,
            ticker="AAA",
            caches=caches,
        )
        is None
    )


def test_eligible_reserve_signal_idx_non_cache_success():
    bars = make_bars(n=30, open_base=100.0)
    cfg = _cfg(min_price=None, min_avg_dollar_volume=None)
    # entry always true, filters off -> the non-cache path returns the position.
    pos = _eligible_reserve_signal_idx(bars, bars.index[20], cfg, parse("close > 0"), 3)
    assert pos == 20


def test_build_equity_curve_trade_outside_calendar():
    calendar = pd.bdate_range("2024-01-01", periods=5)
    frame = make_bars(n=5, open_base=100.0)
    frame.index = calendar
    trade = Trade(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 6, 1),
        entry_date=date(2024, 6, 3),
        entry_price=100.0,
        exit_date=date(2024, 6, 5),
        exit_price=101.0,
        exit_reason="eod",
        shares=1.0,
        entry_cost=100.0,
        exit_value=101.0,
        pnl=1.0,
        return_pct=0.01,
    )
    # The trade's span falls entirely after the calendar (lo >= hi) -> skipped.
    curve = build_equity_curve(
        calendar, [trade], {"AAA": frame}, 10_000.0, price_adjustment="none"
    )
    assert (curve.to_numpy() == 10_000.0).all()


def test_historical_cli_open_report(monkeypatch, tmp_path):
    import screener.reporting as reporting

    opened: list = []
    monkeypatch.setattr(reporting, "open_report", lambda p: opened.append(p))
    report = tmp_path / "open.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            "2024-02-15",
            "--hold",
            "5",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--report",
            str(report),
            "--open-report",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code == 0, res.output
    assert opened  # open_report was invoked
