"""Offline tests for the intraday snapshot-driven options backtester (4.2/4.3).

An in-memory history provider feeds hand-built intraday ``OptionChain``
snapshots; no store, no network. Covers point-in-time entry/exit at snapshot
timestamps, target/stop/exit-time/session-end reasons, the Phase 4.3 equity
hedge, margin tracking, and the default contract-store provider path.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone

from screener.options import contract_store
from screener.options.intraday_backtest import (
    IntradayOptionsBacktestConfig,
    run_intraday_options_backtest,
)
from screener.options.models import OptionChain, OptionContract

DAY = date(2026, 7, 8)  # Wednesday
EXPIRY = date(2026, 7, 17)


def _utc(h: int, m: int) -> datetime:
    return datetime(2026, 7, 8, h, m, tzinfo=timezone.utc)


def _call(as_of: datetime, *, last: float, strike: float = 100.0) -> OptionContract:
    return OptionContract(
        symbol=f"SPY{strike:g}C{as_of:%H%M}",
        underlying="SPY",
        expiry=EXPIRY,
        strike=strike,
        right="call",
        oi=5_000.0,
        oi_change=0.0,
        volume=1_000.0,
        iv=0.25,
        bid=last - 0.5,
        ask=last + 0.5,
        last=last,
        previous_close=last,
        settle=last,
        lot_size=100.0,
        as_of=as_of,
        source="fixture",
    )


def _put(as_of: datetime, *, last: float, strike: float = 100.0) -> OptionContract:
    return OptionContract(
        symbol=f"SPY{strike:g}P{as_of:%H%M}",
        underlying="SPY",
        expiry=EXPIRY,
        strike=strike,
        right="put",
        oi=5_000.0,
        oi_change=0.0,
        volume=1_000.0,
        iv=0.25,
        bid=last - 0.5,
        ask=last + 0.5,
        last=last,
        previous_close=last,
        settle=last,
        lot_size=100.0,
        as_of=as_of,
        source="fixture",
    )


def _chain(as_of: datetime, *contracts: OptionContract, spot: float) -> OptionChain:
    return OptionChain(
        underlying="SPY",
        market="us",
        spot=spot,
        as_of=as_of,
        source="fixture",
        contracts=tuple(contracts),
    )


class _StubProvider:
    def __init__(self, mapping: dict[tuple[str, date], list[OptionChain]]) -> None:
        self._m = mapping

    def chains(self, underlying: str, day: date) -> list[OptionChain]:
        return list(self._m.get((underlying.upper(), day), []))

    def contract_bars(
        self, *args: object, **kwargs: object
    ) -> object:  # pragma: no cover
        raise NotImplementedError


def _cfg(**overrides: object) -> IntradayOptionsBacktestConfig:
    values: dict[str, object] = dict(
        tickers=("SPY",), start=DAY, end=DAY, market="us", structure="long_call"
    )
    values.update(overrides)
    return IntradayOptionsBacktestConfig(**values)  # type: ignore[arg-type]


def _rising_call_session() -> _StubProvider:
    # 09:30 / 10:30 / 11:30 ET (13:30 / 14:30 / 15:30 UTC), call last 10 → 12 → 15.
    chains = [
        _chain(_utc(13, 30), _call(_utc(13, 30), last=10.0), spot=100.0),
        _chain(_utc(14, 30), _call(_utc(14, 30), last=12.0), spot=102.0),
        _chain(_utc(15, 30), _call(_utc(15, 30), last=15.0), spot=105.0),
    ]
    return _StubProvider({("SPY", DAY): chains})


def test_default_flattens_at_session_end():
    result = run_intraday_options_backtest(_cfg(), _rising_call_session())
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "session_end"
    # Entered at the first snapshot (last=10), exited at the last (last=15).
    assert trade.legs[0].entry_price == 10.0
    assert trade.legs[0].exit_price == 15.0
    assert not result.equity_curve.empty


def test_target_exits_intraday_before_session_end():
    # +20% at the 10:30 snapshot (12 vs 10) trips a 15% target there.
    result = run_intraday_options_backtest(
        _cfg(target_pct=15.0), _rising_call_session()
    )
    trade = result.trades[0]
    assert trade.exit_reason == "target"
    assert trade.legs[0].exit_price == 12.0
    assert trade.details["exit_ts"] == _utc(14, 30).isoformat()


def test_stop_exits_intraday():
    chains = [
        _chain(_utc(13, 30), _call(_utc(13, 30), last=10.0), spot=100.0),
        _chain(_utc(14, 30), _call(_utc(14, 30), last=7.0), spot=97.0),
        _chain(_utc(15, 30), _call(_utc(15, 30), last=6.0), spot=96.0),
    ]
    provider = _StubProvider({("SPY", DAY): chains})
    result = run_intraday_options_backtest(_cfg(stop_pct=20.0), provider)
    trade = result.trades[0]
    assert trade.exit_reason == "stop"  # -30% at 10:30 trips the 20% stop
    assert trade.legs[0].exit_price == 7.0


def test_entry_time_and_exit_time_gate_the_session():
    # entry_time 10:30 ET → skip the 09:30 snapshot; exit_time 11:00 ET → the
    # 11:30 snapshot trips "time".
    result = run_intraday_options_backtest(
        _cfg(entry_time=time(10, 30), exit_time=time(11, 0)),
        _rising_call_session(),
    )
    trade = result.trades[0]
    assert trade.legs[0].entry_price == 12.0  # entered at the 10:30 snapshot
    assert trade.exit_reason == "time"
    assert trade.legs[0].exit_price == 15.0


def test_equity_hedge_pnl_nets_into_the_trade():
    # Long 50 underlying units alongside the call; spot 100 → 105 over the day.
    result = run_intraday_options_backtest(
        _cfg(equity_hedge_qty=50.0), _rising_call_session()
    )
    trade = result.trades[0]
    assert trade.details["equity_hedge_qty"] == 50.0
    assert trade.details["equity_hedge_pnl"] == 50.0 * (105.0 - 100.0)  # 250
    # Option leg P&L is (15-10)*100 = 500; hedge adds 250 → 750 gross of costs.
    assert trade.pnl == 500.0 + 250.0


def test_margin_curve_tracked_for_short_put():
    chains = [
        _chain(_utc(13, 30), _put(_utc(13, 30), last=5.0), spot=100.0),
        _chain(_utc(14, 30), _put(_utc(14, 30), last=5.0), spot=100.0),
    ]
    provider = _StubProvider({("SPY", DAY): chains})
    result = run_intraday_options_backtest(
        _cfg(structure="short_put", margin_model="regt"), provider
    )
    assert result.peak_margin > 0.0
    assert not result.margin_curve.empty


def _five_rising_snapshots() -> _StubProvider:
    chains = [
        _chain(_utc(13, 30), _call(_utc(13, 30), last=10.0), spot=100.0),
        _chain(_utc(14, 0), _call(_utc(14, 0), last=13.0), spot=103.0),
        _chain(_utc(14, 30), _call(_utc(14, 30), last=16.0), spot=106.0),
        _chain(_utc(15, 0), _call(_utc(15, 0), last=20.0), spot=110.0),
        _chain(_utc(15, 30), _call(_utc(15, 30), last=25.0), spot=115.0),
    ]
    return _StubProvider({("SPY", DAY): chains})


def test_single_entry_per_session_by_default():
    # After a target exit the engine must NOT re-enter the same session.
    result = run_intraday_options_backtest(
        _cfg(target_pct=15.0), _five_rising_snapshots()
    )
    assert len(result.trades) == 1
    assert result.trades[0].legs[0].entry_price == 10.0


def test_allow_reentry_reopens_after_exit():
    result = run_intraday_options_backtest(
        _cfg(target_pct=15.0, allow_reentry=True), _five_rising_snapshots()
    )
    assert len(result.trades) >= 2  # re-enters after each intraday target exit


def test_no_snapshots_yields_empty_result():
    result = run_intraday_options_backtest(_cfg(), _StubProvider({}))
    assert result.trades == []
    assert result.equity_curve.empty


def test_cli_intraday_backtest_smoke():
    from click.testing import CliRunner

    from screener.options import cli as options_cli

    res = CliRunner().invoke(
        options_cli.options,
        [
            "intraday-backtest",
            "--tickers",
            "SPY",
            "--start",
            DAY.isoformat(),
            "--end",
            DAY.isoformat(),
            "--market",
            "us",
            "--structure",
            "long_call",
            "--target-pct",
            "15",
        ],
        obj={"provider": _rising_call_session()},
    )
    assert res.exit_code == 0, res.output
    assert "Trades Taken" in res.output


def test_defaults_to_contract_store_provider(tmp_path):
    from screener.cache import reset_cache_area_paths, set_cache_area_path

    set_cache_area_path("contracts", tmp_path)
    try:
        for ts, last, spot in [
            (_utc(13, 30), 10.0, 100.0),
            (_utc(15, 30), 15.0, 105.0),
        ]:
            contract_store.append_snapshot(
                _chain(ts, _call(ts, last=last), spot=spot), market="us"
            )
        # provider=None → ContractStoreHistoryProvider reads what we just wrote.
        result = run_intraday_options_backtest(_cfg())
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "session_end"
    finally:
        reset_cache_area_paths()
