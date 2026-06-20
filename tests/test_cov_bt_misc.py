"""Offline coverage tests for assorted backtester modules.

These tests drive the remaining uncovered lines in the backtester support
modules (cli_common, portfolio, dashboard, tearsheet, fills, slippage,
display, vivek_equity, lab, pine_runner). Everything is deterministic and
offline: provider/fetcher seams are stubbed and CLI paths use CliRunner.
"""

from __future__ import annotations

import http.client
import json
import threading
from datetime import date
from http.server import ThreadingHTTPServer

import click
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.backtester import lab
from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.dashboard import (
    _empty_panel,
    _num,
    _pct,
    _table_html,
    dashboard_frames,
    render_dashboard,
    serve_dashboard,
)
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.fills import FillModel
from screener.backtester.lab import LabHandler
from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
    Trade,
)
from screener.backtester.portfolio import Portfolio, build_equity_curve
from screener.backtester.slippage import (
    CompositeSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
    apply_slippage,
)
from screener.backtester.tearsheet import (
    _empty_section,
    _heatmap_cell,
    _monthly_heatmap_html,
    _winners_losers_frames,
    render_tearsheet,
)
from screener.backtester.vivek_equity import prepare_vivek_equity_tool_frame

from tests.conftest import StubPriceFetcher, make_bars


# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------


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
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("AAA", "BBB"),
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _trade(**overrides) -> Trade:
    defaults = dict(
        ticker="AAA",
        rank=1,
        signal_date=date(2024, 1, 5),
        entry_date=date(2024, 1, 8),
        entry_price=100.0,
        exit_date=date(2024, 1, 12),
        exit_price=110.0,
        exit_reason="target",
        shares=10.0,
        entry_cost=1000.0,
        exit_value=1100.0,
        pnl=100.0,
        return_pct=0.10,
    )
    defaults.update(overrides)
    return Trade(**defaults)


# ---------------------------------------------------------------------------
# cli_common.py
# ---------------------------------------------------------------------------


def test_resolve_strategy_exprs_named_strategy_fills_exprs():
    entry, exit_ = resolve_strategy_exprs("ema_trend", None, None)
    assert entry  # filled from the strategy
    # explicit exprs override
    e2, x2 = resolve_strategy_exprs("ema_trend", "custom_entry", "custom_exit")
    assert e2 == "custom_entry"
    assert x2 == "custom_exit"


def test_resolve_strategy_exprs_unknown_strategy_raises_usage_error():
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs("not_a_real_strategy", None, None)


def test_resolve_strategy_exprs_missing_entry_raises():
    with pytest.raises(click.UsageError):
        resolve_strategy_exprs(None, None, None)


def test_build_slippage_model_each_branch():
    assert isinstance(build_slippage_model("fixed", 5.0, 1.0, 0.1), FixedBpsSlippage)
    assert isinstance(
        build_slippage_model("half-spread", 5.0, 1.0, 0.1), HalfSpreadSlippage
    )
    assert isinstance(
        build_slippage_model("vol-impact", 5.0, 1.0, 0.1), VolumeImpactSlippage
    )
    composite = build_slippage_model("composite", 5.0, 1.0, 0.1)
    assert isinstance(composite, CompositeSlippage)
    assert len(composite.models) == 3


def test_parse_partial_exits_valid_and_invalid():
    assert parse_partial_exits(None) == ()
    assert parse_partial_exits(["1.0:0.5", "2.0:0.5"]) == ((1.0, 0.5), (2.0, 0.5))
    with pytest.raises(click.UsageError):
        parse_partial_exits(["bad-token"])


def test_resolve_min_filters_defaults_and_zero_disables():
    price, adv = resolve_min_filters("us", None, None)
    assert price == 1.0 and adv == 1_000.0
    # zero sentinel disables both filters
    price0, adv0 = resolve_min_filters("us", 0, 0)
    assert price0 is None and adv0 is None
    # explicit non-zero passthrough
    price1, adv1 = resolve_min_filters("us", 2.5, 5_000.0)
    assert price1 == 2.5 and adv1 == 5_000.0


# ---------------------------------------------------------------------------
# slippage.py
# ---------------------------------------------------------------------------


class _NegativeSlippage:
    def adverse_fraction(self, side, shares, adv, sigma_daily):
        return -0.5


def test_apply_slippage_clamps_negative_fraction():
    # Negative adverse fraction is clamped to 0 -> price unchanged.
    assert apply_slippage(_NegativeSlippage(), 100.0, "buy") == 100.0
    assert apply_slippage(_NegativeSlippage(), 100.0, "sell") == 100.0


# ---------------------------------------------------------------------------
# fills.py
# ---------------------------------------------------------------------------


def _bars(n: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000.0] * n,
        },
        index=idx,
    )


def test_fillmodel_unknown_entry_order_type_returns_warning():
    cfg = _cfg().model_copy(update={"entry_order_type": "moo"})
    # Force an unknown order type past validation via private attribute bypass:
    # construct the model and monkeypatch cfg through object.__setattr__.
    object.__setattr__(cfg, "entry_order_type", "weird")
    model = FillModel(cfg)
    idx, price, warn = model.entry_price(_bars(), 0)
    assert idx is None and price is None
    assert "unknown entry_order_type" in warn


def test_fillmodel_legacy_slippage_factor_path():
    # slippage_model=None forces the legacy fixed-bps factor branch in _apply_slip.
    cfg = _cfg(slippage_bps=100.0).model_copy(update={"slippage_model": None})
    model = FillModel(cfg)
    idx, price, warn = model.entry_price(_bars(), 0)
    assert warn is None
    # buy-side: ref open (101.0) * (1 + 100bps) = 101.0 * 1.01
    assert price == pytest.approx(101.0 * 1.01)


# ---------------------------------------------------------------------------
# vivek_equity.py
# ---------------------------------------------------------------------------


def test_prepare_vivek_equity_empty_frame_returns_empty():
    assert prepare_vivek_equity_tool_frame(pd.DataFrame()).empty
    assert prepare_vivek_equity_tool_frame(None).empty


def test_prepare_vivek_equity_exercises_close_transition():
    # Build a series that goes up (buy), then ema crosses while still in trend
    # (close condition), driving the `prev != 0.0 and close_` branch.
    idx = pd.bdate_range("2024-01-01", periods=200)
    close = np.r_[
        np.full(45, 100.0),
        np.linspace(101.0, 200.0, 60),  # strong uptrend -> buy (dir=1, ema1>ema2)
        np.linspace(200.0, 175.0, 20),  # pullback: ema1 dips below ema2, price>trend
        np.linspace(176.0, 230.0, 75),  # recover
    ]
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) + 0.5
    low = np.minimum(openp, close) - 0.5
    bars = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    out = prepare_vivek_equity_tool_frame(bars)
    # The close column should fire at least once over this path.
    assert (out["vivek_equity_close"] > 0).any()


# ---------------------------------------------------------------------------
# portfolio.py
# ---------------------------------------------------------------------------


def test_portfolio_rejects_nonpositive_slots():
    with pytest.raises(ValueError):
        Portfolio(100_000.0, 0)


def test_portfolio_update_peak_and_get_position_no_open():
    pf = Portfolio(100_000.0, 2)
    # No open position: update_peak is a no-op, get_position returns None.
    pf.update_peak("AAA", 200.0)
    assert pf.get_position("AAA") is None


def test_portfolio_update_peak_raises_peak():
    pf = Portfolio(100_000.0, 2)
    pf.open("AAA", date(2024, 1, 1), 100.0, commission_bps=0.0)
    pf.update_peak("AAA", 150.0)
    assert pf.get_position("AAA").peak_price == 150.0


def test_portfolio_credit_dividends_paths():
    pf = Portfolio(100_000.0, 2)
    # non-positive per-share dividend -> early return 0.0
    assert pf.credit_dividends("AAA", 0.0) == 0.0
    pos = pf.open("AAA", date(2024, 1, 1), 100.0, commission_bps=0.0)
    cash_before = pf.cash()
    total = pf.credit_dividends("AAA", 1.0)
    assert total == pytest.approx(pos.shares * 1.0)
    assert pf.cash() == pytest.approx(cash_before + total)
    # Unrelated ticker -> no credit (continue branch).
    assert pf.credit_dividends("ZZZ", 1.0) == 0.0


def test_portfolio_close_unknown_ticker_raises():
    pf = Portfolio(100_000.0, 2)
    with pytest.raises(KeyError):
        pf.close("AAA", date(2024, 1, 2), 110.0, "time", 0.0)


def test_portfolio_partial_close_validation_and_full_fraction():
    pf = Portfolio(100_000.0, 2)
    pf.open("AAA", date(2024, 1, 1), 100.0, commission_bps=0.0)
    with pytest.raises(ValueError):
        pf.partial_close("AAA", date(2024, 1, 2), 110.0, "target", 0.0, 0.0)
    # fraction >= 1.0 delegates to close()
    trade = pf.partial_close("AAA", date(2024, 1, 2), 110.0, "target", 1.0, 0.0)
    assert trade.shares == pytest.approx(pf.closed_trades()[0].shares)


def test_portfolio_partial_close_no_open_raises():
    pf = Portfolio(100_000.0, 2)
    with pytest.raises(KeyError):
        pf.partial_close("AAA", date(2024, 1, 2), 110.0, "target", 0.5, 0.0)


def test_portfolio_open_tickers_lists_unique():
    pf = Portfolio(100_000.0, 3)
    pf.open("AAA", date(2024, 1, 1), 100.0, commission_bps=0.0)
    pf.open("BBB", date(2024, 1, 1), 50.0, commission_bps=0.0)
    assert set(pf.open_tickers()) == {"AAA", "BBB"}


def test_build_equity_curve_dividends_and_missing_bar():
    calendar = pd.bdate_range("2024-01-01", periods=10)
    # Frame with an ex-dividend in the holding window AND a NaN close on one day
    # to exercise the dividend-credit, missing-bar carry-forward branches.
    frame = pd.DataFrame(
        {
            "close": [
                100.0,
                101.0,
                float("nan"),
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "dividend": [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=calendar,
    )
    trade = _trade(
        ticker="AAA",
        entry_date=calendar[1].date(),
        exit_date=calendar[6].date(),
        shares=10.0,
        entry_cost=1000.0,
        exit_value=1100.0,
    )
    curve = build_equity_curve(
        calendar,
        [trade],
        {"AAA": frame},
        initial_capital=100_000.0,
        price_adjustment="splits_only",
    )
    assert not curve.isna().any()
    assert len(curve) == len(calendar)


def test_build_equity_curve_missing_frame_uses_entry_price():
    calendar = pd.bdate_range("2024-01-01", periods=6)
    trade = _trade(
        ticker="AAA",
        entry_date=calendar[1].date(),
        exit_date=calendar[4].date(),
        entry_price=100.0,
        shares=10.0,
        entry_cost=1000.0,
        exit_value=1100.0,
    )
    # No price panel entry for AAA -> mark at entry_price.
    curve = build_equity_curve(
        calendar, [trade], {}, initial_capital=100_000.0, price_adjustment="full"
    )
    assert not curve.isna().any()


def test_build_equity_curve_day_absent_from_frame_index():
    # Frame is non-empty but missing a calendar day inside the holding window
    # (`day not in frame.index` -> price=NaN -> carry-forward last valid close).
    calendar = pd.bdate_range("2024-01-01", periods=6)
    # Drop calendar[3] from the frame's index entirely.
    frame_idx = calendar.delete(3)
    frame = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 104.0, 105.0]},
        index=frame_idx,
    )
    trade = _trade(
        ticker="AAA",
        entry_date=calendar[1].date(),
        exit_date=calendar[5].date(),
        entry_price=100.0,
        shares=10.0,
        entry_cost=1000.0,
        exit_value=1100.0,
    )
    curve = build_equity_curve(
        calendar,
        [trade],
        {"AAA": frame},
        initial_capital=100_000.0,
        price_adjustment="full",
    )
    assert not curve.isna().any()
    assert len(curve) == len(calendar)


# ---------------------------------------------------------------------------
# display.py
# ---------------------------------------------------------------------------


def _result(**overrides) -> BacktestResult:
    idx = pd.bdate_range("2024-01-01", periods=45)
    equity = pd.Series([100_000 + i * 400 for i in range(len(idx))], index=idx)
    benchmark = pd.Series([100 + i * 0.2 for i in range(len(idx))], index=idx)
    defaults = dict(
        config=_cfg(strategy_name="ema_trend"),
        trades=[_trade(), _trade(ticker="BBB", rank=2, return_pct=-0.04, pnl=-40.0)],
        equity_curve=equity,
        benchmark_curve=benchmark,
        metrics={
            "total_return": 0.176,
            "benchmark_return": 0.088,
            "max_drawdown": -0.02,
            "sharpe": 1.2,
            "trade_count": 2,
            "unique_tickers": 2,
            "hit_rate": 0.5,
        },
        warnings=["sample warning"],
        selection=pd.DataFrame(
            [{"ticker": "AAA", "signal_date": date(2024, 1, 5), "rank": 1}]
        ),
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def test_print_backtest_with_trades_and_warnings(capsys):
    print_backtest(_result())
    out = capsys.readouterr().out
    assert "warning" in out
    assert "Trade Ledger" in out


def test_print_backtest_no_trades(capsys):
    print_backtest(_result(trades=[]))
    out = capsys.readouterr().out
    assert "No trades" in out


def test_print_ledger_csv_empty_and_full(capsys):
    print_ledger_csv(_result(trades=[]))
    empty_out = capsys.readouterr().out
    assert "ticker" in empty_out
    print_ledger_csv(_result())
    full_out = capsys.readouterr().out
    assert "AAA" in full_out


# ---------------------------------------------------------------------------
# dashboard.py
# ---------------------------------------------------------------------------


def test_dashboard_helpers_non_numeric_and_empty():
    assert _pct("n/a") == "n/a"
    assert _num("x") == "x"
    assert "title" in _empty_panel("pid", "title", "msg")
    assert _table_html(pd.DataFrame(), "tid") == '<p class="empty">No rows.</p>'


def test_dashboard_frames_zero_first_values():
    # Equity / benchmark first values of 0 -> NaN strategy/benchmark returns.
    idx = pd.bdate_range("2024-01-01", periods=40)
    equity = pd.Series([0.0] * len(idx), index=idx)
    benchmark = pd.Series([0.0] * len(idx), index=idx)
    result = _result(equity_curve=equity, benchmark_curve=benchmark)
    frames = dashboard_frames(result)
    assert frames["curves"]["strategy_return"].isna().all()
    assert frames["curves"]["benchmark_return"].isna().all()


def test_render_dashboard_all_empty_sections(tmp_path):
    # Empty curves, monthly, trades, selection -> exercises every empty branch.
    empty = pd.Series(dtype=float)
    result = _result(
        trades=[],
        equity_curve=empty,
        benchmark_curve=empty,
        selection=pd.DataFrame(),
    )
    path = render_dashboard(result, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'id="performance-chart"' in html
    assert "No equity curve data." in html
    assert "No monthly returns." in html
    assert "No trades." in html
    assert "No selected signals." in html


def test_serve_dashboard_starts_and_stops(tmp_path, monkeypatch):
    served = {}

    class _FakeServer:
        def __init__(self, addr, handler):
            served["addr"] = addr

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def serve_forever(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "screener.backtester.dashboard._ReusableThreadingHTTPServer", _FakeServer
    )
    serve_dashboard(tmp_path, 0)
    assert served["addr"][0] == "127.0.0.1"


# ---------------------------------------------------------------------------
# tearsheet.py
# ---------------------------------------------------------------------------


def test_tearsheet_helpers():
    assert "title" in _empty_section("sid", "title", "msg")
    assert _heatmap_cell(float("nan")) == '<td class="hm-empty"></td>'
    # negative value -> red color branch
    assert "185,28,28" in _heatmap_cell(-0.05)
    # positive value -> teal color branch
    assert "15,118,110" in _heatmap_cell(0.05)
    assert (
        _monthly_heatmap_html(pd.DataFrame())
        == '<p class="empty">No monthly returns.</p>'
    )


def test_winners_losers_frames_formats_pnl():
    trades = pd.DataFrame(
        [
            {"ticker": "AAA", "return_pct": 0.10, "pnl": 100.0},
            {"ticker": "BBB", "return_pct": -0.05, "pnl": -50.0},
        ]
    )
    winners, losers = _winners_losers_frames(trades)
    assert winners.iloc[0]["pnl"] == "100.00"
    assert "%" in winners.iloc[0]["return_pct"]


def test_render_tearsheet_all_empty(tmp_path):
    empty = pd.Series(dtype=float)
    result = _result(
        trades=[],
        equity_curve=empty,
        benchmark_curve=empty,
        selection=pd.DataFrame(),
        warnings=[],
    )
    out = tmp_path / "tearsheet.html"
    path = render_tearsheet(result, out)
    html = path.read_text(encoding="utf-8")
    assert "No equity curve data." in html
    assert "No monthly returns." in html
    assert "No trades." in html
    assert "No warnings." in html


def test_render_tearsheet_config_rows_truncation(tmp_path):
    many = tuple(f"T{i}" for i in range(25))
    result = _result(
        config=_cfg(
            tickers=many,
            membership_added=tuple((f"T{i}", date(2024, 1, 1)) for i in range(3)),
        ),
    )
    out = tmp_path / "ts2.html"
    path = render_tearsheet(result, out)
    html = path.read_text(encoding="utf-8")
    assert "25 tickers" in html
    assert "3 dated symbols" in html


# ---------------------------------------------------------------------------
# lab.py
# ---------------------------------------------------------------------------


def test_lab_json_default():
    assert lab._json_default(date(2024, 1, 1)) == "2024-01-01"
    assert lab._json_default(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
    assert lab._json_default(42) == 42


def test_lab_html_renders():
    html = lab._lab_html()
    assert "Backtest Strategy Lab" in html


def _start_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_lab_get_index_returns_html():
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/", headers={"Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        assert resp.status == 200
        assert b"Backtest Strategy Lab" in resp.read()
    finally:
        server.shutdown()
        server.server_close()


def test_lab_get_unknown_path_404():
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/nope", headers={"Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        assert resp.status == 404
    finally:
        server.shutdown()
        server.server_close()


def test_lab_post_unknown_path_404():
    server, port = _start_server()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        payload = b"{}"
        conn.request(
            "POST",
            "/api/other",
            body=payload,
            headers={
                "Host": f"127.0.0.1:{port}",
                "Content-Length": str(len(payload)),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 404
    finally:
        server.shutdown()
        server.server_close()


def _post_run(port, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    body = json.dumps(payload).encode()
    conn.request(
        "POST",
        "/api/run",
        body=body,
        headers={
            "Host": f"127.0.0.1:{port}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    return conn.getresponse()


def test_lab_post_unknown_universe_400():
    server, port = _start_server()
    try:
        resp = _post_run(
            port,
            {
                "market": "us",
                "strategies": ["ema_trend"],
                "universe": "bogus",
                "start": "2024-01-01",
                "end": "2024-02-01",
            },
        )
        assert resp.status == 400
        assert "Unknown universe" in resp.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_lab_post_unknown_compare_universe_400():
    server, port = _start_server()
    try:
        resp = _post_run(
            port,
            {
                "market": "us",
                "strategies": ["ema_trend"],
                "tickers": "AAA",
                "compare_universe": "bogus",
                "start": "2024-01-01",
                "end": "2024-02-01",
            },
        )
        assert resp.status == 400
        assert "comparison universe" in resp.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_lab_post_run_success(monkeypatch):
    bars_a = make_bars(n=80, seed=21, open_base=100.0)
    spy = make_bars(n=80, seed=23, open_base=400.0)
    fetcher = StubPriceFetcher({"AAA": bars_a, "SPY": spy})
    monkeypatch.setattr(lab, "build_price_fetcher", lambda auto_adjust=True: fetcher)
    server, port = _start_server()
    try:
        resp = _post_run(
            port,
            {
                "market": "us",
                "strategies": ["ema_trend"],
                "tickers": "AAA",
                "start": bars_a.index[20].date().isoformat(),
                "end": bars_a.index[60].date().isoformat(),
                "hold": 5,
                "top": 1,
                "initial_capital": 100_000,
            },
        )
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "results" in data
    finally:
        server.shutdown()
        server.server_close()


def test_backtest_lab_command_serves_then_stops(monkeypatch):
    """Drive the click command; stub the server so serve_forever returns fast."""
    started = {}

    class _FakeServer:
        def __init__(self, addr, handler):
            started["addr"] = addr

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            started["closed"] = True

    monkeypatch.setattr(lab, "ThreadingHTTPServer", _FakeServer)
    runner = CliRunner()
    result = runner.invoke(lab.backtest_lab, ["--host", "127.0.0.1", "--port", "0"])
    assert result.exit_code == 0
    assert started["closed"] is True


# ---------------------------------------------------------------------------
# pine_runner.py (thin shim)
# ---------------------------------------------------------------------------


def test_pine_runner_reexports():
    from screener.backtester import pine_runner

    assert hasattr(pine_runner, "main")
    assert hasattr(pine_runner, "STRATEGIES")
    assert "main" in pine_runner.__all__
