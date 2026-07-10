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


from types import SimpleNamespace


import numpy as np


import pandas as pd


import pytest


from click.testing import CliRunner


from main import cli


from screener.backtester import rolling


from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)


from screener.backtester.core import (
    _precompute_entry_signals,
)


from screener.backtester.fills import FillModel


from screener.backtester.historical import (
    _benchmark_series_from_panel,
    run_backtest,
    select_candidates,
)


from screener.backtester.models import BacktestConfig


from screener.backtester.pine import (
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


def test_precompute_entry_signals_eval_failure():
    bars_by = {"AAA": make_bars(n=10), "EMPTY": pd.DataFrame()}
    warns: list[str] = []
    out = _precompute_entry_signals(bars_by, parse("nonexistent_col > 0"), warns)
    assert out == {}
    assert any("entry eval failed" in w for w in warns)


def test_build_rolling_candidate_matrices_membership_and_regime():
    idx = pd.bdate_range("2024-01-01", periods=20)
    bars = make_bars(n=20, open_base=100.0)
    bars.index = idx
    bars_by = {"AAA": bars}
    entry_sig = {"AAA": pd.Series(True, index=idx)}
    master = list(idx)
    # membership_added suppresses early signals; regime_allowed gates days.
    regime_allowed = pd.Series([False] * 5 + [True] * 15, index=idx)
    mats = _build_rolling_candidate_matrices(
        bars_by,
        entry_sig,
        {},
        master,
        lookback_required=3,
        membership_added={"AAA": idx[10].date()},
        regime_allowed=regime_allowed,
    )
    # before date-added -> suppressed.
    assert not bool(mats.signal_mat.iloc[5]["AAA"])
    # after both gates -> allowed.
    assert bool(mats.signal_mat.iloc[15]["AAA"])
    assert mats.filter_mat is None


def test_select_candidates_warnings():
    bars_by = {
        "EMPTY": pd.DataFrame(),
        "SHORT": make_bars(n=2),
        "GOOD": make_bars(n=30, open_base=100.0),
    }
    entry = parse("close > sma(close, 3)")
    df, warns = select_candidates(
        bars_by,
        entry,
        pd.Timestamp("2024-02-20"),
        2,
        3,
        _cfg(min_price=None, min_avg_dollar_volume=None),
    )
    assert any("no data" in w for w in warns)
    assert any("insufficient lookback" in w for w in warns)


def test_select_candidates_filtered_count():
    bars_by = {"AAA": make_bars(n=30, open_base=100.0)}
    entry = parse("close > 0")
    df, warns = select_candidates(
        bars_by,
        entry,
        pd.Timestamp("2024-02-20"),
        2,
        0,
        _cfg(min_price=1e9),
    )
    assert any("filtered" in w for w in warns)
    assert df.empty


def test_select_candidates_eval_failure_warning():
    bars_by = {"AAA": make_bars(n=30)}
    df, warns = select_candidates(
        bars_by, parse("nonexistent_col > 0"), pd.Timestamp("2024-02-20"), 2, 0
    )
    assert any("entry eval failed" in w for w in warns)


def test_select_candidates_no_signal():
    bars_by = {"AAA": make_bars(n=30, open_base=100.0)}
    df, warns = select_candidates(
        bars_by, parse("close > 1000000000"), pd.Timestamp("2024-02-20"), 2, 3
    )
    assert df.empty


def test_select_candidates_ranks_roles():
    bars_by = {
        "AAA": make_bars(n=30, open_base=100.0),
        "BBB": make_bars(n=30, open_base=50.0),
        "CCC": make_bars(n=30, open_base=30.0),
    }
    df, _ = select_candidates(
        bars_by,
        parse("close > 0"),
        pd.Timestamp("2024-02-20"),
        1,
        0,
        _cfg(reserve_multiple=3),
    )
    assert "rank" in df.columns and "role" in df.columns
    assert (df["role"] == "active").sum() == 1
    assert (df["role"] == "reserve").sum() >= 1


def test_benchmark_series_from_panel_empty_and_present():
    assert _benchmark_series_from_panel({}, "SPY").empty
    assert _benchmark_series_from_panel({"SPY": pd.DataFrame()}, "SPY").empty
    s = _benchmark_series_from_panel({"SPY": make_bars(n=5)}, "SPY")
    assert not s.empty and s.name == "SPY"


def test_run_backtest_empty_selection():
    # entry never fires -> empty selection branch.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=60, open_base=100.0),
            "SPY": make_bars(n=60, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 15),
        entry_expr="close > 1000000000",
        tickers=("AAA",),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert result.trades == []
    assert result.selection.empty


def test_run_backtest_with_reserve_rotation():
    # Build a universe with an active that exits early and reserves to rotate in.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=80, seed=1, open_base=100.0),
            "BBB": make_bars(n=80, seed=2, open_base=80.0),
            "CCC": make_bars(n=80, seed=3, open_base=60.0),
            "SPY": make_bars(n=80, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("AAA", "BBB", "CCC"),
        reserve_multiple=3,
        reinvest=True,
        stop_loss=0.03,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_run_backtest_allow_reentry():
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=80, seed=5, open_base=100.0),
            "SPY": make_bars(n=80, seed=9, open_base=400.0),
        }
    )
    cfg = _cfg(
        as_of=date(2024, 2, 1),
        hold=3,
        top=1,
        entry_expr="close > sma(close, 3)",
        tickers=("AAA",),
        allow_reentry=True,
        max_reentries=2,
        reinvest=True,
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    assert isinstance(result.trades, list)


def test_run_rolling_backtest_end_before_start():
    fetcher = StubPriceFetcher({"AAA": make_bars(n=10), "SPY": make_bars(n=10)})
    with pytest.raises(ValueError):
        run_rolling_backtest(
            _cfg(), fetcher, start_date=date(2024, 5, 1), end_date=date(2024, 1, 1)
        )


def test_run_rolling_backtest_no_trading_days():
    # Price data exists only OUTSIDE the requested window -> early_result path.
    fetcher = StubPriceFetcher(
        {
            "AAA": make_bars(n=10, start="2024-01-01"),
            "SPY": make_bars(n=10, start="2024-01-01"),
        }
    )
    result = run_rolling_backtest(
        _cfg(tickers=("AAA",), min_price=None, min_avg_dollar_volume=None),
        fetcher,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 2, 1),
    )
    assert any("no trading days" in w for w in result.warnings)
    assert result.trades == []


def test_run_rolling_backtest_with_regime_and_splits():
    data_dict = {
        "AAA": make_bars(n=60, seed=1, open_base=100.0),
        "SPY": make_bars(n=60, seed=9, open_base=400.0),
    }
    # add split_factor so splits_only path runs.
    aaa = data_dict["AAA"]
    aaa["split_factor"] = 1.0
    aaa["dividend"] = 0.0
    cfg = _cfg(
        as_of=date(2024, 3, 1),
        hold=3,
        top=1,
        tickers=("AAA",),
        regime_filter=("uptrend",),
        price_adjustment="splits_only",
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data_dict),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 3, 1),
    )
    assert isinstance(result.trades, list)


def test_run_rolling_backtest_with_membership_added():
    idx = pd.bdate_range("2024-01-01", periods=60)
    aaa = make_bars(n=60, seed=1, open_base=100.0)
    aaa.index = idx
    spy = make_bars(n=60, seed=9, open_base=400.0)
    spy.index = idx
    cfg = _cfg(
        as_of=idx[-1].date(),
        hold=3,
        top=1,
        tickers=("AAA",),
        membership_added=(("AAA", idx[30].date()),),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher({"AAA": aaa, "SPY": spy}),
        start_date=idx[5].date(),
        end_date=idx[-1].date(),
    )
    # entries suppressed before date-added => trades only have signal dates >= added.
    for t in result.trades:
        assert t.signal_date >= idx[30].date()


def test_cli_common_helpers():
    # slippage models
    for m in ["fixed", "half-spread", "vol-impact", "composite"]:
        assert build_slippage_model(m, 1.0, 2.0, 0.1) is not None
    # partial exits
    assert parse_partial_exits(()) == ()
    assert parse_partial_exits(("0.05:0.5",)) == ((0.05, 0.5),)
    import click

    with pytest.raises(click.UsageError):
        parse_partial_exits(("bad",))
    # min filters: 0 disables.
    assert resolve_min_filters("us", 0.0, 0.0) == (None, None)
    assert resolve_min_filters("us", None, None) == (1.0, 1000.0)
    # strategy exprs.
    e, x = resolve_strategy_exprs(None, "close > 0", None)
    assert e == "close > 0"
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs(None, None, None)
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs("does_not_exist", None, None)


def test_historical_cli_csv_and_report(tmp_path):
    fetcher = _stub_env()
    report = tmp_path / "report.html"
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
            "--top",
            "2",
            "--entry",
            "close > sma(close, 3)",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
            "--csv",
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert report.exists()


def test_historical_cli_report_no_csv(tmp_path):
    fetcher = _stub_env()
    report = tmp_path / "r.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--universe-file",
            str(_universe_file(tmp_path)),
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
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Report:" in res.output


def test_historical_cli_no_universe_error():
    res = CliRunner().invoke(
        cli,
        ["backtest-historical", "--as-of", "2024-02-15", "--entry", "close > 0"],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "No universe provided" in res.output


def test_historical_cli_strategy_shortcut():
    fetcher = _stub_env()
    res = CliRunner().invoke(
        cli,
        [
            "backtest-historical",
            "--tickers",
            "AAA,BBB",
            "--as-of",
            "2024-02-15",
            "--strategy",
            "breakout",
            "--hold",
            "5",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_csv():
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
            "2024-02-20",
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
            "--csv",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_default_window_and_report(tmp_path):
    fetcher = _stub_env(n=400)
    report = tmp_path / "roll.html"
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA,BBB",
            "--years",
            "1",
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
            "--report",
            str(report),
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert report.exists()


def test_rolling_cli_point_in_time_requires_universe():
    # --point-in-time with --tickers is a usage error.
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--tickers",
            "AAA",
            "--entry",
            "close > 0",
            "--point-in-time",
        ],
        obj=_stub_env(),
    )
    assert res.exit_code != 0
    assert "point-in-time" in res.output


def test_rolling_cli_universe_path(monkeypatch):
    fetcher = _stub_env()

    loaded = SimpleNamespace(
        symbols=("AAA", "BBB"),
        name="sp500",
        source="test",
        cached_path="/tmp/x",
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)

    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--universe",
            "sp500",
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
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output
    assert "Universe:" in res.output


def test_rolling_cli_point_in_time_universe(monkeypatch):
    fetcher = _stub_env()
    loaded = SimpleNamespace(
        symbols=("AAA", "BBB"),
        name="sp500",
        source="test",
        cached_path="/tmp/x",
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)
    monkeypatch.setattr(
        rolling,
        "load_sp500_membership",
        lambda **k: {"AAA": date(2010, 1, 1), "BBB": None},
    )
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "--universe",
            "sp500",
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
            "--point-in-time",
            "--min-price",
            "0",
            "--min-avg-dollar-volume",
            "0",
        ],
        obj=fetcher,
    )
    assert res.exit_code == 0, res.output


def test_rolling_cli_point_in_time_non_sp500(monkeypatch):
    fetcher = _stub_env()
    loaded = SimpleNamespace(
        symbols=("AAA",), name="nifty50", source="test", cached_path="/tmp/x"
    )
    monkeypatch.setattr(rolling, "load_current_universe", lambda *a, **k: loaded)
    res = CliRunner().invoke(
        cli,
        [
            "backtest-rolling",
            "-m",
            "india",
            "--universe",
            "nifty50",
            "--start",
            "2024-01-15",
            "--end",
            "2024-02-15",
            "--entry",
            "close > 0",
            "--point-in-time",
        ],
        obj=fetcher,
    )
    assert res.exit_code != 0
    assert "sp500" in res.output


def test_close_slot_day_not_in_bars():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars)
    slot_states = {0: state}
    fm = FillModel(cfg)
    # a day not present in the index -> returns False.
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=pd.Timestamp("2050-01-01"),
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False


def test_close_slot_before_entry_bar():
    bars = make_bars(n=10)
    cfg, portfolio, state = _open_slot(bars, entry_idx=3)
    slot_states = {0: state}
    fm = FillModel(cfg)
    # day == entry bar (i < entry_idx+1) -> skip.
    out = _close_slot_at_day(
        slot_id=0,
        state=state,
        bars=bars,
        day=bars.index[3],
        cfg=cfg,
        portfolio=portfolio,
        slot_states=slot_states,
        fill_model=fm,
    )
    assert out is False
