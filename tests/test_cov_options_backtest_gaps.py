"""Coverage for options structures / position backtester / CLI edge paths."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import click
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli as _root_cli  # noqa: F401 - import-order guard
from screener.options import cli as options_cli
from screener.options import position_backtest as pb
from screener.options import structures as st
from screener.options.bt_models import LegFill, OptionsBacktestConfig
from tests.conftest import StubPriceFetcher
from tests.test_options_position_backtest import (
    _WEEKDAY,
    _call_put_pair,
    _chain,
    _contract,
    _run,
    _synthetic_loader,
    _underlying_bars,
)


def _fetcher(start: date, end: date, **kwargs) -> StubPriceFetcher:
    return StubPriceFetcher({"RELIANCE.NS": _underlying_bars(start, end, **kwargs)})


def _cfg(start: date, end: date, **overrides) -> OptionsBacktestConfig:
    values = {
        "tickers": ("RELIANCE",),
        "start": start,
        "end": end,
        "structure": "long_call",
        "entry_expr": "true",
        "exit_dte": 1,
    }
    values.update(overrides)
    return OptionsBacktestConfig(**values)


# ---------------------------------------------------------------------------
# structures.py


def test_select_expiry_edge_rules():
    as_of = date(2026, 7, 10)
    empty = _chain(_contract(expiry=date(2026, 7, 1)), day=as_of)
    assert st.select_expiry(empty, "front", as_of) is None

    near = date(2026, 7, 17)
    far = date(2026, 7, 24)
    chain = _chain(
        _contract(expiry=near),
        _contract(expiry=far, symbol="B"),
        day=as_of,
    )
    assert st.select_expiry(chain, "dte:notanint", as_of) is None
    # No expiry with DTE >= 90 → fall back to the farthest.
    assert st.select_expiry(chain, "dte:90", as_of) == far
    with pytest.raises(ValueError, match="unknown expiry rule"):
        st.select_expiry(chain, "quarterly", as_of)


def test_nearest_strike_and_bs_delta_guards(monkeypatch):
    assert st._nearest_strike([], 100.0) is None

    as_of = date(2026, 7, 10)
    priced = _contract(strike=1270.0, expiry=date(2026, 7, 28))
    unpriced = _contract(
        strike=1270.0, expiry=date(2026, 7, 28), bid=None, ask=None, last=None
    )
    assert st._bs_delta(unpriced, spot=1275.0, as_of=as_of) is None  # no price
    assert st._bs_delta(priced, spot=0.0, as_of=as_of) is None  # bad spot
    # Same-day expiry → dte < 1.
    assert st._bs_delta(priced, spot=1275.0, as_of=date(2026, 7, 28)) is None
    # IV unsolvable: call priced above spot has no BS solution.
    absurd = _contract(
        strike=1270.0,
        expiry=date(2026, 7, 28),
        iv=None,
        bid=None,
        ask=None,
        last=5000.0,
    )
    assert st._bs_delta(absurd, spot=1275.0, as_of=as_of) is None
    # Greeks failure propagates as None.
    monkeypatch.setattr(st, "black_scholes_greeks", lambda *a, **k: None)
    assert st._bs_delta(priced, spot=1275.0, as_of=as_of) is None


def test_select_strike_spot_fallback_and_bad_rules():
    expiry = date(2026, 7, 28)
    chain = _chain(
        _contract(strike=1250.0, expiry=expiry),
        _contract(strike=1270.0, expiry=expiry, symbol="B"),
        _contract(strike=1290.0, expiry=expiry, symbol="C"),
        spot=None,
    )
    # No spot → median strike proxy picks the middle strike for ATM.
    picked = st.select_strike(chain, expiry, "call", "atm")
    assert picked is not None and picked.strike == 1270.0
    assert st.select_strike(chain, expiry, "call", "moneyness:bad") is None
    assert st.select_strike(chain, expiry, "call", "delta:bad") is None
    with pytest.raises(ValueError, match="unknown strike rule"):
        st.select_strike(chain, expiry, "call", "gamma:0.5")


def test_select_strike_delta_moneyness_fallback():
    expiry = date(2026, 7, 28)
    chain = _chain(
        _contract(strike=1270.0, expiry=expiry),
        _contract(strike=1340.0, expiry=expiry, symbol="B"),
        _contract(right="put", strike=1270.0, expiry=expiry, symbol="C"),
        _contract(right="put", strike=1150.0, expiry=expiry, symbol="D"),
        spot=1275.0,
    )
    # as_of on expiry day → every BS delta is None → moneyness fallback.
    call = st.select_strike(chain, expiry, "call", "delta:0.50", as_of=expiry)
    assert call is not None and call.strike == 1340.0  # spot * 1.05
    put = st.select_strike(chain, expiry, "put", "delta:0.20", as_of=expiry)
    assert put is not None and put.strike == 1150.0  # spot * 0.90


# ---------------------------------------------------------------------------
# position_backtest.py helpers


def test_mark_price_settle_fallback_and_find_contract():
    bare = _contract(bid=None, ask=None, last=None, settle=7.5)
    assert pb._mark_price(bare) == pytest.approx(7.5)
    dead = _contract(bid=None, ask=None, last=None, settle=None)
    assert pb._mark_price(dead) is None
    chain = _chain(_contract())
    assert (
        pb._find_contract(chain, right="put", strike=999.0, expiry=date(2026, 7, 28))
        is None
    )


def test_slip_signed_intrinsic_and_return_helpers():
    assert pb._slip_price(10.0, -1, 0.01, opening=True) == pytest.approx(9.9)
    assert pb._slip_price(10.0, -1, 0.01, opening=False) == pytest.approx(10.1)
    leg = LegFill(
        right="call",
        strike=100.0,
        expiry=date(2026, 7, 28),
        side=1,
        lots=1,
        lot_size=1.0,
        entry_price=10.0,
        exit_price=None,
    )
    # Missing exit price legs are skipped.
    assert pb._signed_premium([leg], use_exit=True) == 0.0
    assert pb._intrinsic("put", 100.0, 90.0) == pytest.approx(10.0)
    assert pb._position_return_pct(10.0, 12.0, 0.0) == 0.0


def test_resolve_entry_expression_criteria():
    ok = _cfg(date(2026, 7, 6), date(2026, 7, 7), screen_criterion="high_iv_rank")
    assert pb._resolve_entry_expression(ok) == "iv_rank >= 80"
    bad = _cfg(date(2026, 7, 6), date(2026, 7, 7), screen_criterion="bogus")
    with pytest.raises(ValueError, match="unknown screen criterion"):
        pb._resolve_entry_expression(bad)


def test_default_chain_loader_delegates(monkeypatch):
    seen = {}

    def fake_load(day, symbols=None, refresh=False):
        seen["args"] = (day, symbols, refresh)
        return {}

    monkeypatch.setattr(pb, "load_bhavcopy_chains", fake_load)
    out = pb._default_chain_loader(date(2026, 7, 7), {"RELIANCE"}, refresh=True)
    assert out == {}
    assert seen["args"] == (date(2026, 7, 7), {"RELIANCE"}, True)


def test_liquidity_min_volume():
    cfg = _cfg(date(2026, 7, 6), date(2026, 7, 7), min_volume=1_000_000.0)
    assert not pb._liquidity_ok(_contract(volume=10.0), cfg)


def test_open_structure_warning_paths(monkeypatch):
    cfg = _cfg(date(2026, 7, 6), date(2026, 7, 9))
    structure = st.build_structure("long_call")
    warnings: list[str] = []
    # All expiries in the past → no expiry.
    stale = _chain(_contract(expiry=date(2026, 7, 1)), day=date(2026, 7, 8))
    assert (
        pb._open_structure(
            stale, structure, as_of=date(2026, 7, 8), cfg=cfg, warnings=warnings
        )
        is None
    )
    assert any("no expiry" in w for w in warnings)

    # Chain has only puts → straddle call leg has no strike.
    puts_only = _chain(
        _contract(right="put", expiry=date(2026, 7, 28)), day=date(2026, 7, 8)
    )
    warnings.clear()
    assert (
        pb._open_structure(
            puts_only,
            st.build_structure("straddle"),
            as_of=date(2026, 7, 8),
            cfg=cfg,
            warnings=warnings,
        )
        is None
    )
    assert any("no strike" in w for w in warnings)

    # Selected contract without any usable price → "no price" warning.
    priceless = _contract(bid=None, ask=None, last=None, settle=None)
    chain = _chain(_contract(), day=date(2026, 7, 8))
    monkeypatch.setattr(pb, "select_strike", lambda *a, **k: priceless)
    warnings.clear()
    assert (
        pb._open_structure(
            chain, structure, as_of=date(2026, 7, 8), cfg=cfg, warnings=warnings
        )
        is None
    )
    assert any("no price" in w for w in warnings)


def test_mark_legs_last_resort_entry_price():
    leg = LegFill(
        right="call",
        strike=1270.0,
        expiry=date(2026, 7, 28),
        side=1,
        lots=1,
        lot_size=500.0,
        entry_price=10.0,
    )
    pos = pb._OpenPosition(
        symbol="RELIANCE",
        structure="long_call",
        signal_date=date(2026, 7, 6),
        entry_date=date(2026, 7, 7),
        legs=[leg],
        entry_premium=5000.0,
        gross_premium=5000.0,
        entry_costs=0.0,
    )
    marks, carry = pb._mark_legs(
        pos, None, as_of=date(2026, 7, 8), underlying_close=None, carry={}
    )
    assert marks == [10.0]
    assert carry == {}


def test_check_exit_expiry_without_spot():
    leg = LegFill(
        right="call",
        strike=1270.0,
        expiry=date(2026, 7, 8),
        side=1,
        lots=1,
        lot_size=500.0,
        entry_price=10.0,
    )
    pos = pb._OpenPosition(
        symbol="RELIANCE",
        structure="long_call",
        signal_date=date(2026, 7, 6),
        entry_date=date(2026, 7, 7),
        legs=[leg],
        entry_premium=5000.0,
        gross_premium=5000.0,
        entry_costs=0.0,
    )
    cfg = _cfg(date(2026, 7, 6), date(2026, 7, 9))
    decision = pb._check_exit(
        pos,
        [10.0],
        day=date(2026, 7, 8),
        underlying_close=None,
        exit_signal=False,
        cfg=cfg,
    )
    assert decision == ("expiry", True)


# ---------------------------------------------------------------------------
# position_backtest.py run paths


def test_run_validations():
    with pytest.raises(ValueError, match="end must be on or after start"):
        _run(_cfg(date(2026, 7, 8), date(2026, 7, 7)))
    us = OptionsBacktestConfig.model_construct(
        market="us",
        tickers=("AAPL",),
        start=date(2026, 7, 6),
        end=date(2026, 7, 7),
        structure="long_call",
        entry_expr="true",
    )
    with pytest.raises(ValueError, match="market=india"):
        _run(us)
    with pytest.raises(ValueError, match="invalid entry expression"):
        _run(_cfg(date(2026, 7, 6), date(2026, 7, 7), entry_expr="(("))
    with pytest.raises(ValueError, match="invalid exit expression"):
        _run(_cfg(date(2026, 7, 6), date(2026, 7, 7), exit_expr="(("))
    with pytest.raises(ValueError, match="at least one ticker"):
        _run(_cfg(date(2026, 7, 6), date(2026, 7, 7), tickers=(" ",)))


def test_exit_expr_signal_exit():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=11.0, put_last=4.0),
    }
    cfg = _cfg(d0, d2, exit_expr="close > 0", exit_dte=0)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d2),
    )
    assert result.trades
    assert result.trades[0].exit_reason == "exit_expr"


def test_max_hold_time_exit():
    d0 = date(2026, 7, 6)
    end = date(2026, 7, 10)
    schedule = {
        day: _call_put_pair(day, call_last=10.0, put_last=5.0)
        for day in (date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 9), end)
    }
    cfg = _cfg(d0, end, max_hold=2, exit_dte=0)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, end),
    )
    assert result.trades
    assert result.trades[0].exit_reason == "time"


def test_entry_and_exit_eval_failures():
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    schedule = {d1: _call_put_pair(d1, call_last=10.0, put_last=5.0)}
    cfg = _cfg(d0, d1, entry_expr="mystery_field > 1")
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d1),
    )
    assert result.trades == []
    assert any("entry eval failed" in w for w in result.warnings)

    cfg = _cfg(d0, d1, exit_expr="mystery_field > 1")
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d1),
    )
    assert any("exit eval failed" in w for w in result.warnings)


def test_screen_criterion_with_empty_panel():
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    schedule = {d1: _call_put_pair(d1, call_last=10.0, put_last=5.0)}
    cfg = _cfg(d0, d1, entry_expr="", screen_criterion="high_iv_rank")
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d1),
        panel=pd.DataFrame(),
    )
    # No panel coverage → criterion never fires → no trades, no crash.
    assert result.trades == []


def test_no_price_history_warning():
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    schedule = {d1: _call_put_pair(d1, call_last=10.0, put_last=5.0)}
    cfg = _cfg(d0, d1, tickers=("RELIANCE", "TCS"))
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d1),  # no TCS bars
    )
    assert any("no underlying price history" in w for w in result.warnings)


def test_loader_exception_records_warning():
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)

    def broken(day, symbols):
        raise RuntimeError("archive down")

    cfg = _cfg(d0, d1)
    result = _run(cfg, chain_loader=broken, price_fetcher=_fetcher(d0, d1))
    assert result.trades == []
    assert any("chain load failed" in w for w in result.warnings)


def test_pending_entry_waits_for_chain():
    d0, d2 = date(2026, 7, 6), date(2026, 7, 8)
    # No chain on the intermediate session → pending entry carries to d2.
    schedule = {d2: _call_put_pair(d2, call_last=10.0, put_last=5.0)}
    cfg = _cfg(d0, d2, exit_dte=0)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, d2),
    )
    assert result.trades
    assert result.trades[0].entry_date == d2


def test_expiry_intrinsic_without_chain():
    expiry = date(2026, 7, 9)
    d0, d1 = date(2026, 7, 7), date(2026, 7, 8)
    # Entry d1; expiry day has no archive chain → intrinsic vs underlying close.
    schedule = {d1: _call_put_pair(d1, call_last=10.0, put_last=5.0, expiry=expiry)}
    cfg = _cfg(d0, expiry, exit_dte=0)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=_fetcher(d0, expiry, start_px=1275.0, end_px=1290.0),
    )
    assert result.trades
    assert result.trades[0].exit_reason == "expiry"


def test_empty_range_weekend_returns_empty():
    saturday = date(2026, 7, 11)
    cfg = _cfg(saturday, saturday)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader({}),
        price_fetcher=_fetcher(saturday - timedelta(days=5), saturday),
    )
    assert result.trades == []
    assert result.equity_curve.empty


def test_intraday_indexed_bars_normalize_lookup():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    bars = _underlying_bars(d0, d2)
    bars.index = bars.index + pd.Timedelta(hours=15, minutes=30)
    # Drop the final session so d2 has no bar → close lookup misses.
    bars = bars[bars.index.normalize() < pd.Timestamp(d2)]
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    cfg = _cfg(d0, d2, target_pct=25.0)
    result = _run(
        cfg,
        chain_loader=_synthetic_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )
    assert result.trades
    assert result.trades[0].exit_reason == "target"


# ---------------------------------------------------------------------------
# options CLI backtest paths


def _weekday_calendar(monkeypatch):
    monkeypatch.setattr(pb, "is_trading_day", _WEEKDAY)


def _base_args(start: date, end: date, *extra: str) -> list[str]:
    return [
        "backtest",
        "--tickers",
        "RELIANCE",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--structure",
        "long_call",
        "--entry",
        "true",
        *extra,
    ]


def test_cli_csv_output_with_and_without_trades(monkeypatch):
    _weekday_calendar(monkeypatch)
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    obj = {
        "chain_loader": _synthetic_loader(schedule),
        "price_fetcher": _fetcher(d0, d2),
    }
    res = CliRunner().invoke(
        options_cli.options,
        _base_args(d0, d2, "--target-pct", "20", "--csv"),
        obj=obj,
    )
    assert res.exit_code == 0, res.output
    assert "RELIANCE" in res.output and "target" in res.output

    res = CliRunner().invoke(
        options_cli.options,
        [*_base_args(d0, d2, "--csv")[:-3], "--entry", "close > 999999", "--csv"],
        obj=obj,
    )
    assert res.exit_code == 0, res.output
    assert res.output.startswith("symbol,structure,entry_date")


def test_cli_warning_flood_and_no_trades(monkeypatch):
    _weekday_calendar(monkeypatch)
    start, end = date(2026, 6, 1), date(2026, 7, 3)  # >20 trading days

    def broken(day, symbols):
        raise RuntimeError("archive down")

    res = CliRunner().invoke(
        options_cli.options,
        _base_args(start, end),
        obj={
            "chain_loader": broken,
            "price_fetcher": _fetcher(start, end),
        },
    )
    assert res.exit_code == 0, res.output
    assert "more warnings" in res.output
    assert "No option trades taken" in res.output


def test_cli_default_fetcher_and_loader(monkeypatch):
    _weekday_calendar(monkeypatch)
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    schedule = {d1: _call_put_pair(d1, call_last=10.0, put_last=5.0)}
    monkeypatch.setattr(options_cli, "get_price_fetcher", lambda obj: _fetcher(d0, d1))
    monkeypatch.setattr(
        pb,
        "load_bhavcopy_chains",
        lambda day, symbols=None, refresh=False: (
            {"RELIANCE": schedule[day]} if day in schedule else {}
        ),
    )
    res = CliRunner().invoke(options_cli.options, _base_args(d0, d1))
    assert res.exit_code == 0, res.output
    assert "Options Position Backtest" in res.output


def test_cli_market_and_ticker_guards():
    ctx = click.Context(options_cli.backtest)
    with ctx, pytest.raises(click.UsageError, match="only -m india"):
        options_cli.backtest.callback(
            "us",
            "AAPL",
            datetime(2026, 7, 6),
            datetime(2026, 7, 8),
            "long_call",
            "atm",
            "front",
            0.05,
            1,
            "true",
            None,
            None,
            None,
            None,
            1,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            False,
        )
    res = CliRunner().invoke(
        options_cli.options,
        _base_args(date(2026, 7, 6), date(2026, 7, 8))[:2]
        + [" "]
        + _base_args(date(2026, 7, 6), date(2026, 7, 8))[3:],
    )
    assert res.exit_code != 0
    assert "at least one symbol" in res.output


def test_cli_config_and_run_errors(monkeypatch):
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)

    def boom(**kwargs):
        raise ValueError("bad config")

    monkeypatch.setattr(options_cli, "OptionsBacktestConfig", boom)
    res = CliRunner().invoke(options_cli.options, _base_args(d0, d1))
    assert res.exit_code != 0
    assert "bad config" in res.output

    monkeypatch.undo()
    monkeypatch.setattr(pb, "is_trading_day", _WEEKDAY)
    res = CliRunner().invoke(
        options_cli.options,
        _base_args(d0, d1, "--screen-criterion", "bogus"),
        obj={
            "chain_loader": _synthetic_loader({}),
            "price_fetcher": _fetcher(d0, d1),
        },
    )
    assert res.exit_code != 0
    assert "unknown screen criterion" in res.output
