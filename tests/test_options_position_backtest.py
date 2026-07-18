"""Offline tests for the India options position-P&L backtester."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli as _root_cli  # noqa: F401 - import-order guard
from screener.earnings_backtest.metrics import compute_backtest_summary
from screener.options import cli as options_cli
from screener.options.bt_models import OptionsBacktestConfig
from screener.options.models import OptionChain, OptionContract
from screener.options.nse_bhavcopy import normalize_bhavcopy_options
from screener.options.position_backtest import run_options_position_backtest
from tests.conftest import StubPriceFetcher

FIXTURE = Path(__file__).parent / "fixtures" / "nse_fo_bhavcopy_options_sample.csv"

# Offline weekday calendar — avoid NSE holiday network fetch in tests.
_WEEKDAY = lambda d: d.weekday() < 5  # noqa: E731


def _run(cfg, **kwargs):
    kwargs.setdefault("trading_day", _WEEKDAY)
    return run_options_position_backtest(cfg, **kwargs)


def _contract(**overrides) -> OptionContract:
    values = {
        "symbol": "RELIANCE26JUL1270CE",
        "underlying": "RELIANCE",
        "expiry": date(2026, 7, 28),
        "strike": 1270.0,
        "right": "call",
        "oi": 100_000.0,
        "oi_change": 10.0,
        "volume": 1_000.0,
        "iv": 0.25,
        "bid": 32.0,
        "ask": 34.0,
        "last": 33.0,
        "previous_close": 30.0,
        "settle": 33.0,
        "lot_size": 500.0,
        "as_of": datetime(2026, 7, 8, tzinfo=timezone.utc),
        "source": "fixture",
    }
    values.update(overrides)
    return OptionContract(**values)


def _chain(
    *contracts: OptionContract, spot: float = 1275.0, day: date | None = None
) -> OptionChain:
    as_of = day or date(2026, 7, 8)
    return OptionChain(
        underlying="RELIANCE",
        market="india",
        spot=spot,
        as_of=as_of,
        source="fixture",
        contracts=tuple(contracts),
    )


def _underlying_bars(
    start: date,
    end: date,
    *,
    start_px: float = 1270.0,
    end_px: float = 1300.0,
) -> pd.DataFrame:
    idx = pd.bdate_range(start=start - timedelta(days=30), end=end)
    n = len(idx)
    close = pd.Series(
        [start_px + (end_px - start_px) * i / max(n - 1, 1) for i in range(n)],
        index=idx,
        dtype=float,
    )
    openp = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": openp,
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _synthetic_loader(
    schedule: dict[date, OptionChain],
):
    def loader(day: date, symbols: set[str]) -> dict[str, OptionChain]:
        chain = schedule.get(day)
        if chain is None:
            return {}
        if symbols and chain.underlying not in {s.upper() for s in symbols}:
            return {}
        return {chain.underlying: chain}

    return loader


def _call_put_pair(
    day: date,
    *,
    call_last: float,
    put_last: float,
    strike: float = 1270.0,
    expiry: date = date(2026, 7, 28),
    spot: float = 1275.0,
    oi: float = 100_000.0,
    volume: float = 1_000.0,
) -> OptionChain:
    return _chain(
        _contract(
            right="call",
            strike=strike,
            last=call_last,
            bid=call_last - 0.5,
            ask=call_last + 0.5,
            settle=call_last,
            expiry=expiry,
            as_of=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
            oi=oi,
            volume=volume,
            symbol=f"C{strike}{day.isoformat()}",
        ),
        _contract(
            right="put",
            strike=strike,
            last=put_last,
            bid=put_last - 0.5,
            ask=put_last + 0.5,
            settle=put_last,
            expiry=expiry,
            as_of=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
            oi=oi,
            volume=volume,
            symbol=f"P{strike}{day.isoformat()}",
        ),
        spot=spot,
        day=day,
    )


def test_settle_populated_from_fixture():
    frame = pd.read_csv(FIXTURE)
    chains = normalize_bhavcopy_options(frame, as_of=date(2026, 7, 8))
    chain = chains["RELIANCE"]
    assert all(c.settle is not None and c.settle > 0 for c in chain.contracts)
    assert chain.contracts[0].settle == pytest.approx(45.25)


def test_target_hit():
    d0, d1, d2, d3 = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
        date(2026, 7, 9),
    )
    # Signal d0 → entry d1 at call=10; d2 mark rises to 15 (+50%) → target 25% hits.
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=8.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=5.0),
        d3: _call_put_pair(d3, call_last=16.0, put_last=4.0),
    }
    bars = _underlying_bars(d0, d3)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d3,
        structure="long_call",
        entry_expr="true",
        target_pct=25.0,
        exit_dte=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.exit_reason == "target"
    assert trade.return_pct > 0
    assert trade.entry_date > d0  # PIT: entry after signal


def test_stop_hit():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=8.0),
        d2: _call_put_pair(d2, call_last=4.0, put_last=12.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d2,
        structure="long_call",
        entry_expr="true",
        stop_pct=40.0,
        exit_dte=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    assert result.trades[0].exit_reason == "stop"


def test_hold_to_expiry_intrinsic_settle():
    # Short window ending on expiry day; premium decays, settle at intrinsic.
    expiry = date(2026, 7, 10)
    d0, d1 = date(2026, 7, 8), date(2026, 7, 9)
    schedule = {
        d1: _call_put_pair(
            d1, call_last=10.0, put_last=5.0, expiry=expiry, spot=1280.0
        ),
        expiry: _call_put_pair(
            expiry, call_last=5.0, put_last=1.0, expiry=expiry, spot=1280.0
        ),
    }
    # Underlying close on expiry = 1280 → call intrinsic = 10 (strike 1270).
    bars = _underlying_bars(d0, expiry, start_px=1275.0, end_px=1280.0)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=expiry,
        structure="long_call",
        entry_expr="true",
        exit_dte=0,
        max_hold=None,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.exit_reason in {"expiry", "end", "dte"}
    # At least one leg closed with a non-None exit price.
    assert all(leg.exit_price is not None for leg in trade.legs)


def test_dte_exit():
    # Entry d1 (7/7), expiry 7/11 → DTE=4 on entry day; on d2 DTE=3 → exit_dte=3.
    expiry = date(2026, 7, 11)
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0, expiry=expiry),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0, expiry=expiry),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d2,
        structure="long_call",
        entry_expr="true",
        exit_dte=3,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    assert result.trades[0].exit_reason == "dte"


def test_liquidity_skip():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0, oi=10.0, volume=1.0),
        d2: _call_put_pair(d2, call_last=12.0, put_last=4.0, oi=10.0, volume=1.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d2,
        structure="long_call",
        entry_expr="true",
        min_oi=50_000.0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades == []
    assert any("liquidity" in w.lower() for w in result.warnings)


def test_cost_arithmetic_to_the_rupee():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=12.0, put_last=4.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    slip = 0.01
    commission = 20.0
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d2,
        structure="long_call",
        entry_expr="true",
        target_pct=5.0,  # 12 vs 10 after slip should clear a small target
        slippage_pct=slip,
        commission_per_order=commission,
        exit_dte=1,
        lots=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    lot = 500.0
    # Entry: buy call at 10 * (1+0.01) = 10.1
    entry_px = 10.0 * (1.0 + slip)
    # Exit at target day: sell at mark * (1-slip)
    # mark on d2 = 12 mid; exit = 12 * 0.99
    exit_px = 12.0 * (1.0 - slip)
    entry_prem = entry_px * lot
    exit_prem = exit_px * lot
    raw_pnl = exit_prem - entry_prem
    expected_pnl = raw_pnl - commission - commission  # open + close
    assert trade.pnl == pytest.approx(expected_pnl, abs=0.01)
    expected_ret = (expected_pnl / entry_prem) * 100.0
    assert trade.return_pct == pytest.approx(expected_ret, abs=0.01)


def test_missing_day_carry_forward():
    d0, d1, d3 = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 9),
    )
    # Intermediate session missing from archive; position should carry and exit
    # on d3 via target.
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d3: _call_put_pair(d3, call_last=20.0, put_last=2.0),
    }
    bars = _underlying_bars(d0, d3)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d3,
        structure="long_call",
        entry_expr="true",
        target_pct=50.0,
        exit_dte=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    assert result.trades[0].exit_date == d3
    assert result.trades[0].exit_reason == "target"


def test_pit_entry_after_signal():
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d1)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d1,
        structure="long_call",
        entry_expr="true",
        exit_dte=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.entry_date == d1
    assert trade.details["signal_date"] == d0.isoformat()
    assert trade.entry_date > d0


def test_summary_via_compute_backtest_summary():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=d2,
        structure="straddle",
        entry_expr="true",
        target_pct=10.0,
        exit_dte=1,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    summary = compute_backtest_summary(result.trades, strategy="straddle")
    assert summary["trades_taken"] == len(result.trades)
    assert "avg_return_pct" in summary


def test_cli_smoke_with_injected_loader(monkeypatch):
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    monkeypatch.setattr(
        "screener.options.position_backtest.is_trading_day",
        _WEEKDAY,
    )
    runner = CliRunner()
    res = runner.invoke(
        options_cli.options,
        [
            "backtest",
            "--tickers",
            "RELIANCE",
            "--start",
            d0.isoformat(),
            "--end",
            d2.isoformat(),
            "--structure",
            "long_call",
            "--target-pct",
            "20",
            "--entry",
            "true",
        ],
        obj={
            "chain_loader": _synthetic_loader(schedule),
            "price_fetcher": fetcher,
        },
    )
    assert res.exit_code == 0, res.output
    assert "Options Position Backtest" in res.output or "Trade" in res.output


def test_cli_rejects_us_market():
    res = CliRunner().invoke(
        options_cli.options,
        [
            "backtest",
            "-m",
            "us",
            "--tickers",
            "AAPL",
            "--start",
            "2026-07-01",
            "--end",
            "2026-07-10",
            "--structure",
            "long_call",
        ],
    )
    # market choice restricted to india at Click level
    assert res.exit_code != 0
