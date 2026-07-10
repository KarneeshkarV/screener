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


import pytest


from click.testing import CliRunner


from main import cli


from screener.backtester import data


from screener.backtester.fills import FillModel


from screener.backtester.historical import (
    run_backtest,
)


from screener.backtester.models import BacktestConfig


from screener.backtester.pine import (
    PineError,
    evaluate,
    parse,
)


from screener.backtester.rolling import (
    _build_rolling_candidate_matrices,
    run_rolling_backtest,
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
    _close_slot_at_day,
    _fire_partial_exits_at_bar,
    _force_close_open_slots,
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


def test_close_slot_fully_closed_by_partial():
    bars = make_bars(n=20, open_base=100.0)
    # craft a target that the next bar's high definitely exceeds, fraction 1.0.
    cfg, portfolio, state = _open_slot(
        bars,
        entry_idx=1,
        partial_targets=(0.0,),  # any positive high triggers
        partial_fractions=(1.0,),
        stop_ref=None,
        target_ref=None,
    )
    state.partial_fired = [False]
    slot_states = {0: state}
    fm = FillModel(cfg)
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[2],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    # whole position scaled out -> slot freed via the position-None branch.
    assert out is True
    assert slot_states[0] is None


def test_close_slot_duplicate_index_returns_false():
    bars = make_bars(n=6)
    # duplicate a timestamp so get_loc returns a slice/array.
    dup_idx = bars.index.tolist()
    dup_idx[3] = dup_idx[2]
    bars.index = pd.DatetimeIndex(dup_idx)
    cfg, portfolio, state = _open_slot(bars, entry_idx=0)
    slot_states = {0: state}
    fm = FillModel(cfg)
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[2],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False


def test_fire_partial_exits_no_position():
    bars = make_bars(n=10, open_base=100.0)
    cfg = _cfg()
    portfolio = Portfolio(cfg.initial_capital, 1)
    state = _SlotState(
        ticker="AAA",
        entry_idx=1,
        entry_date=bars.index[1].date(),
        entry_fill=100.0,
        signal_date=bars.index[0].date(),
        rank=1,
        stop_ref=None,
        target_ref=None,
        hold_limit_idx=6,
        peak=100.0,
        exit_signal=None,
        partial_targets=(0.05,),
        partial_fractions=(0.5,),
        partial_fired=[False],
    )
    # no open position -> early return (pos is None).
    _fire_partial_exits_at_bar(state, bars, 2, cfg, portfolio, FillModel(cfg))
    assert state.partial_fired == [False]


def test_fire_partial_exits_no_targets():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars)
    # no partial targets -> immediate return.
    _fire_partial_exits_at_bar(state, bars, 2, cfg, portfolio, FillModel(cfg))


def test_maybe_credit_dividends_paths():
    cfg_none = _cfg(price_adjustment="none")
    bars = make_bars(n=5)
    portfolio = Portfolio(cfg_none.initial_capital, 1)
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
    )
    # no dividend column -> early return.
    _maybe_credit_dividends(portfolio, state, bars, 1, cfg_none)

    # full adjustment -> early return even with dividend column.
    bars_div = make_bars(n=5)
    bars_div["dividend"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    _maybe_credit_dividends(
        portfolio, state, bars_div, 1, _cfg(price_adjustment="full")
    )

    # non-numeric dividend -> ValueError swallowed.
    bars_bad = make_bars(n=5)
    bars_bad["dividend"] = ["x", "y", "z", "w", "v"]
    _maybe_credit_dividends(portfolio, state, bars_bad, 1, cfg_none)

    # zero/neg dividend -> not credited.
    bars_zero = make_bars(n=5)
    bars_zero["dividend"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    _maybe_credit_dividends(portfolio, state, bars_zero, 1, cfg_none)


def test_force_close_open_slots_empty_tail():
    bars = make_bars(n=10, start="2024-01-01")
    cfg, portfolio, state = _open_slot(bars, entry_idx=2)
    slot_states = {0: state, 1: None}
    slot_bars = {0: bars}
    # end_ts before entry_date -> tail empty -> skip (continue).
    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=pd.Timestamp("2020-01-01"),
        fill_model=FillModel(cfg),
    )
    assert slot_states[0] is state  # not closed


def test_force_close_open_slots_closes():
    bars = make_bars(n=10, start="2024-01-01")
    cfg, portfolio, state = _open_slot(bars, entry_idx=2)
    slot_states = {0: state}
    slot_bars = {0: bars}
    _force_close_open_slots(
        slot_states=slot_states,
        slot_bars=slot_bars,
        cfg=cfg,
        portfolio=portfolio,
        end_ts=bars.index[-1],
        fill_model=FillModel(cfg),
    )
    assert slot_states[0] is None
    assert portfolio.closed_trades()


def test_pine_unary_minus_eval():
    bars = make_bars(n=5)
    out = evaluate(parse("-close + close"), bars)
    assert out.iloc[0] == pytest.approx(0.0)


def test_pine_series_from_name_missing_series_direct():
    from screener.backtester.pine import _series_from_name

    bars = make_bars(n=5).drop(columns=["open"])
    with pytest.raises(PineError):
        _series_from_name("open", bars)


def test_rolling_with_filters_and_exit_expr():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, seed=1, open_base=100.0),
            "SPY": make_bars(n=60, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=1,
        tickers=("AAA",),
        exit_expr="close < sma(close, 3)",
        min_price=1.0,
        min_avg_dollar_volume=1.0,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=date(2024, 1, 15), end_date=date(2024, 3, 1)
    )
    assert isinstance(result.trades, list)


def test_build_matrices_filter_mat_present():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bars = make_bars(n=20, open_base=100.0)
    bars.index = idx
    bars_by = {"AAA": bars}
    entry_sig = {"AAA": pd.Series(True, index=idx)}
    filter_sig = {"AAA": pd.Series(True, index=idx)}
    mats = _build_rolling_candidate_matrices(
        bars_by, entry_sig, filter_sig, list(idx), lookback_required=3
    )
    assert mats.filter_mat is not None


def test_run_backtest_active_no_data_and_no_history():
    # AAA has no data; BBB has data only after as_of (no history at as_of).
    aaa_empty = pd.DataFrame()
    bbb = make_bars(n=30, start="2024-06-01", open_base=80.0)
    spy = make_bars(n=120, start="2024-01-01", open_base=400.0)
    # GOOD ticker selectable at as_of.
    good = make_bars(n=120, start="2024-01-01", open_base=100.0)
    fetcher = StubPriceFetcher({"AAA": aaa_empty, "BBB": bbb, "GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=3,
        entry_expr="close > 0",
        tickers=("GOOD", "AAA", "BBB"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_run_backtest_reentry_full_flow():
    # Long horizon ticker that re-enters after closing.
    good = make_bars(n=150, start="2024-01-01", seed=7, open_base=100.0)
    spy = make_bars(n=150, start="2024-01-01", seed=9, open_base=400.0)
    fetcher = StubPriceFetcher({"GOOD": good, "SPY": spy})
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=2,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("GOOD",),
        allow_reentry=True,
        max_reentries=3,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_rolling_dashboard_path(monkeypatch, tmp_path):
    import screener.backtester.dashboard as dash

    monkeypatch.setattr(
        dash, "render_dashboard", lambda result, d: tmp_path / "dash.html"
    )
    monkeypatch.setattr(dash, "serve_dashboard", lambda d, p: None)
    fetcher = _stub_env()
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--hold",
            "5",
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--dashboard",
            "--dashboard-dir",
            str(tmp_path),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Dashboard:" in res.output


def test_rolling_candidate_make_slot_fails_on_last_bar():
    # A ticker whose data ends exactly at the window end: a signal on the final
    # in-window bar has no post-signal entry bar -> _make_slot_state returns None
    # inside _simulate_day (the candidate make-slot-fail branch).
    idx = pd.bdate_range("2024-01-02", periods=25)
    aaa = make_bars(n=25, seed=3, open_base=100.0)
    aaa.index = idx
    spy = make_bars(n=25, seed=9, open_base=400.0)
    spy.index = idx
    fetcher = StubPriceFetcher({"AAA": aaa, "SPY": spy})
    cfg = _cfg(
        as_of=idx[-1].date(),
        hold=3,
        top=1,
        tickers=("AAA",),
        entry_expr="close > 0",  # fires every bar, including the last
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=idx[0].date(), end_date=idx[-1].date()
    )
    assert isinstance(result.trades, list)


def test_rolling_with_empty_panel_ticker():
    # One ticker has data, another resolves to an empty frame in the panel.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, seed=1, open_base=100.0),
            "EMPTY": pd.DataFrame(),
            "SPY": make_bars(n=60, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=4,
        top=2,
        tickers=("AAA", "EMPTY"),
        entry_expr="close > sma(close, 3)",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg, fetcher, start_date=date(2024, 1, 15), end_date=date(2024, 3, 1)
    )
    assert isinstance(result.trades, list)


def test_normalize_frame_with_adj_close_and_actions():
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [1000, 1100],
            "Adj Close": [10.5, 11.5],
            "Dividends": [0.0, 0.5],
            "Stock Splits": [0.0, 2.0],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    out = data._normalize_frame(raw)
    assert "adj_close" in out.columns
    assert "dividend" in out.columns
    assert "split_factor" in out.columns


def test_normalize_frame_dividend_alias():
    raw = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
            "dividend": [0.25],
        },
        index=pd.bdate_range("2024-01-01", periods=1),
    )
    out = data._normalize_frame(raw)
    assert out["dividend"].iloc[0] == pytest.approx(0.25)


def test_yfinance_fetcher_partial_cache_extends_forward(tmp_path, monkeypatch):
    import yfinance as yf

    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, batch_size=50)
    # seed cache with an early window.
    early = pd.DataFrame(
        {
            "Open": np.arange(10.0, 20.0),
            "High": np.arange(11.0, 21.0),
            "Low": np.arange(9.0, 19.0),
            "Close": np.arange(10.5, 20.5),
            "Volume": np.arange(1000, 1010),
        },
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    data._save_cache("AAA", data._normalize_frame(early), tmp_path)

    def fake_download(target, **kwargs):
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(30.0, 35.0),
                "High": np.arange(31.0, 36.0),
                "Low": np.arange(29.0, 34.0),
                "Close": np.arange(30.5, 35.5),
                "Volume": np.arange(2000, 2005),
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    # request a window that extends past the cached max -> forward fetch branch.
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 3, 1))
    assert "AAA" in out


def test_yfinance_fetcher_partial_cache_extends_backward(tmp_path, monkeypatch):
    import yfinance as yf

    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, batch_size=50)
    # seed cache with a LATE window (covers the recent end, misses early start).
    late = pd.DataFrame(
        {
            "Open": np.arange(10.0, 20.0),
            "High": np.arange(11.0, 21.0),
            "Low": np.arange(9.0, 19.0),
            "Close": np.arange(10.5, 20.5),
            "Volume": np.arange(1000, 1010),
        },
        index=pd.bdate_range("2024-02-15", periods=10),
    )
    data._save_cache("AAA", data._normalize_frame(late), tmp_path)

    def fake_download(target, **kwargs):
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(30.0, 35.0),
                "High": np.arange(31.0, 36.0),
                "Low": np.arange(29.0, 34.0),
                "Close": np.arange(30.5, 35.5),
                "Volume": np.arange(2000, 2005),
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    # request a window starting well before cache min -> backward fetch branch.
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 2, 28))
    assert "AAA" in out


def test_yfinance_fetcher_auto_adjust_false_actions(tmp_path, monkeypatch):
    import yfinance as yf

    captured = {}

    def fake_download(target, **kwargs):
        captured.update(kwargs)
        idx = pd.bdate_range(kwargs["start"], periods=5)
        return pd.DataFrame(
            {
                "Open": np.arange(10.0, 15.0),
                "High": np.arange(11.0, 16.0),
                "Low": np.arange(9.0, 14.0),
                "Close": np.arange(10.5, 15.5),
                "Volume": np.arange(1000, 1005),
                "Dividends": [0.0] * 5,
                "Stock Splits": [0.0] * 5,
            },
            index=idx,
        )

    monkeypatch.setattr(yf, "download", fake_download)
    fetcher = data.YFinancePriceFetcher(
        cache_dir=tmp_path, auto_adjust=False, batch_size=50
    )
    fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 15))
    assert captured.get("actions") is True


def test_fmp_fetcher_requires_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError):
        data.FMPPriceFetcher()


def test_fmp_fetcher_empty_payload(monkeypatch, tmp_path):
    import requests

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(requests, "Session", lambda: FakeSession())
    fetcher = data.FMPPriceFetcher(api_key="k", cache_dir=tmp_path)
    out = fetcher.fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 15))
    assert out["AAA"].empty


def test_configure_yfinance_uses_only_supported_api(monkeypatch):
    import sys
    import types

    data._YFINANCE_CONFIGURED = False
    fake_yf = types.ModuleType("yfinance")
    fake_yf.set_tz_cache_location = lambda loc: None
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    data._configure_yfinance()
    assert data._YFINANCE_CONFIGURED is True


def test_configure_yfinance_swap_failure(monkeypatch):
    import sys
    import types

    data._YFINANCE_CONFIGURED = False
    fake_yf = types.ModuleType("yfinance")

    def boom(loc):
        raise RuntimeError("nope")

    fake_yf.set_tz_cache_location = boom
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    monkeypatch.setattr("os.path.isdir", lambda p: True)
    data._configure_yfinance()
    assert data._YFINANCE_CONFIGURED is True


def test_configure_yfinance_already_configured():
    data._YFINANCE_CONFIGURED = True
    data._configure_yfinance()  # short-circuit


def test_event_driven_sim_active_edge_branches():
    as_of = pd.Timestamp("2024-02-01")
    # ACTIVE rows: no-data, no-history, make-slot-fail, good.
    good = _flat_then_trending("2024-01-01", 60, 100.0)
    no_history = _flat_then_trending("2024-03-01", 30, 80.0)  # all after as_of
    actives = pd.DataFrame(
        [
            {"ticker": "GOOD", "rank": 1},
            {"ticker": "NODATA", "rank": 2},
            {"ticker": "NOHIST", "rank": 3},
            {"ticker": "MAKEFAIL", "rank": 4},
        ]
    )
    reserves = pd.DataFrame(columns=["ticker", "rank"])
    # MAKEFAIL frame ends exactly at as_of so signal bar is last -> no entry bar.
    makefail_frame = _flat_then_trending("2023-12-01", 44, 50.0)
    bars_by = {
        "GOOD": good,
        "NODATA": pd.DataFrame(),
        "NOHIST": no_history,
        "MAKEFAIL": makefail_frame,
    }
    cfg = _cfg(
        as_of=as_of.date(),
        hold=3,
        top=4,
        entry_expr="close > 0",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    portfolio = Portfolio(cfg.initial_capital, 4)
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
    assert any("no data during sim" in w for w in warnings)
    assert any("no history at as_of" in w for w in warnings)
