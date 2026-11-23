"""Offline regressions for options expiry settlement and entry-day churn.

Covers two defects:

* Settlement on expiry day used the (possibly split/bonus adjusted) yfinance
  underlying close, overwriting the unadjusted bhavcopy settle/spot for every
  leg in the structure.
* A structure opened on a ``DTE == 1`` session was immediately closed on the
  same bar by the default ``exit_dte = 1`` or by entry slippage crossing a
  stop, then re-queued for the next session in a daily churn loop.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from screener.options.bt_models import LegSpec, OptionsBacktestConfig, StructureSpec
from screener.options.models import OptionChain, OptionContract
from screener.options.position_backtest import run_options_position_backtest
from tests.conftest import StubPriceFetcher


# Offline weekday calendar — avoid the NSE holiday network fetch in tests.
def _WEEKDAY(d):
    return d.weekday() < 5


def _run(cfg, **kwargs):
    kwargs.setdefault("trading_day", _WEEKDAY)
    return run_options_position_backtest(cfg, **kwargs)


def _contract(
    day: date,
    *,
    right: str,
    expiry: date,
    last: float,
    settle: float | None,
    strike: float = 1270.0,
) -> OptionContract:
    return OptionContract(
        symbol=f"RELIANCE{expiry:%d%b}{strike:g}{right[0].upper()}",
        underlying="RELIANCE",
        expiry=expiry,
        strike=strike,
        right=right,
        oi=100_000.0,
        volume=1_000.0,
        iv=0.25,
        bid=None,
        ask=None,
        last=last,
        previous_close=last,
        settle=settle,
        lot_size=500.0,
        as_of=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        source="fixture",
    )


def _chain(
    day: date,
    *contracts: OptionContract,
    spot: float | None = 1275.0,
) -> OptionChain:
    return OptionChain(
        underlying="RELIANCE",
        market="india",
        spot=spot,
        as_of=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        source="fixture",
        contracts=tuple(contracts),
    )


def _loader(schedule: dict[date, OptionChain]):
    def loader(day: date, symbols: set[str]) -> dict[str, OptionChain]:
        chain = schedule.get(day)
        if chain is None:
            return {}
        if symbols and chain.underlying not in {s.upper() for s in symbols}:
            return {}
        return {chain.underlying: chain}

    return loader


def _bars(
    start: date,
    end: date,
    *,
    expiry_day: date,
    expiry_close: float,
    other_close: float = 1275.0,
) -> pd.DataFrame:
    """Weekday bars with an explicit (optionally split-adjusted) expiry close."""
    idx = pd.bdate_range(start=start - timedelta(days=30), end=end)
    close = pd.Series(other_close, index=idx, dtype=float)
    close.loc[pd.Timestamp(expiry_day)] = expiry_close
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def _single_leg_cfg(
    start: date,
    end: date,
    *,
    stop_pct: float | None = None,
    slippage_pct: float = 0.0,
    exit_dte: int = 1,
) -> OptionsBacktestConfig:
    return OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=start,
        end=end,
        structure="long_call",
        entry_expr="true",
        stop_pct=stop_pct,
        slippage_pct=slippage_pct,
        exit_dte=exit_dte,
        max_hold=None,
    )


def test_expiry_settlement_prefers_bhavcopy_settle_over_adjusted_close():
    # A 1:2 split halves the adjusted close (640) while NSE strikes stay
    # nominal at 1270. The bhavcopy settle (12) is authoritative.
    expiry, d0, d1 = date(2026, 7, 10), date(2026, 7, 8), date(2026, 7, 9)
    schedule = {
        d1: _chain(
            d1,
            _contract(d1, right="call", expiry=expiry, last=10.0, settle=10.0),
        ),
        expiry: _chain(
            expiry,
            _contract(expiry, right="call", expiry=expiry, last=11.0, settle=12.0),
            spot=1280.0,
        ),
    }
    bars = _bars(d0, expiry, expiry_day=expiry, expiry_close=640.0)
    cfg = _single_leg_cfg(d0, expiry)
    result = _run(
        cfg,
        chain_loader=_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )

    trade = result.trades[0]
    assert (trade.entry_date, trade.exit_date) == (d1, expiry)
    assert trade.exit_reason == "expiry"
    assert trade.legs[0].exit_price == pytest.approx(12.0)
    # (12 - 10) * 500, no costs/slippage configured.
    assert trade.pnl == pytest.approx(1_000.0)
    assert "exit_mark" not in trade.details


def test_expiry_settlement_falls_back_to_chain_spot_before_adjusted_close():
    # No settle price: intrinsic against today's unadjusted chain spot (1280
    # → 10), not against the split-adjusted close (640 → 0).
    expiry, d0, d1 = date(2026, 7, 10), date(2026, 7, 8), date(2026, 7, 9)
    schedule = {
        d1: _chain(
            d1,
            _contract(d1, right="call", expiry=expiry, last=10.0, settle=10.0),
        ),
        expiry: _chain(
            expiry,
            _contract(expiry, right="call", expiry=expiry, last=11.0, settle=None),
            spot=1280.0,
        ),
    }
    bars = _bars(d0, expiry, expiry_day=expiry, expiry_close=640.0)
    cfg = _single_leg_cfg(d0, expiry)
    result = _run(
        cfg,
        chain_loader=_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )

    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.legs[0].exit_price == pytest.approx(10.0)
    assert trade.pnl == pytest.approx(0.0)


def test_expiry_settlement_uses_underlying_close_only_as_last_resort():
    # No settle and no chain spot: intrinsic against the underlying close
    # (1290 → 20), not the contract's last traded quote (11).
    expiry, d0, d1 = date(2026, 7, 10), date(2026, 7, 8), date(2026, 7, 9)
    schedule = {
        d1: _chain(
            d1,
            _contract(d1, right="call", expiry=expiry, last=10.0, settle=10.0),
        ),
        expiry: _chain(
            expiry,
            _contract(expiry, right="call", expiry=expiry, last=11.0, settle=None),
            spot=None,
        ),
    }
    bars = _bars(d0, expiry, expiry_day=expiry, expiry_close=1290.0)
    cfg = _single_leg_cfg(d0, expiry)
    result = _run(
        cfg,
        chain_loader=_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )

    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.legs[0].exit_price == pytest.approx(20.0)
    assert "exit_mark" not in trade.details


def test_multi_expiry_only_the_expiring_leg_settles(monkeypatch):
    # A diagonal whose near leg expires on d2 while the far leg still has a
    # month to run. Only the near leg settles; the far leg keeps its MTM mark
    # from today's chain instead of being collapsed to intrinsic.
    near, far = date(2026, 7, 8), date(2026, 7, 30)
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    spec = StructureSpec(
        name="diagonal",
        legs=(
            LegSpec(
                right="call", side=1, lots=1, strike_rule="atm", expiry_rule="front"
            ),
            LegSpec(
                right="call", side=1, lots=1, strike_rule="atm", expiry_rule="next"
            ),
        ),
    )
    monkeypatch.setattr(
        "screener.options.position_backtest.build_structure",
        lambda *_args, **_kwargs: spec,
    )
    schedule = {
        d1: _chain(
            d1,
            _contract(d1, right="call", expiry=near, last=10.0, settle=10.0),
            _contract(d1, right="call", expiry=far, last=40.0, settle=40.0),
        ),
        near: _chain(
            near,
            _contract(near, right="call", expiry=near, last=11.0, settle=12.0),
            # Fresh quote 40, official settle 41: a non-expiring leg must use
            # the MTM quote, not the settle.
            _contract(near, right="call", expiry=far, last=40.0, settle=41.0),
            spot=1300.0,
        ),
    }
    bars = _bars(d0, near, expiry_day=near, expiry_close=1300.0)
    cfg = OptionsBacktestConfig(
        tickers=("RELIANCE",),
        start=d0,
        end=near,
        structure="diagonal",
        entry_expr="true",
        exit_dte=1,
    )
    result = _run(
        cfg,
        chain_loader=_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )

    trade = result.trades[0]
    by_expiry = {leg.expiry: leg for leg in trade.legs}
    assert trade.exit_reason == "expiry"
    assert by_expiry[near].exit_price == pytest.approx(12.0)
    assert by_expiry[far].exit_price == pytest.approx(40.0)
    # Near leg 12 vs 10 and far leg 40 vs 40, 500 shares each.
    assert trade.pnl == pytest.approx(1_000.0)


def test_dte_one_entry_is_not_closed_on_its_own_bar():
    # Signal d0 → entry d1 with expiry d2 (DTE == 1) under the default
    # exit_dte=1. Entry slippage alone pushes the mark through the 2% stop.
    # The position must not round-trip on d1; it settles on d2.
    expiry, d0, d1 = date(2026, 7, 8), date(2026, 7, 6), date(2026, 7, 7)
    schedule = {
        d1: _chain(
            d1,
            _contract(d1, right="call", expiry=expiry, last=10.0, settle=10.0),
            spot=1280.0,
        ),
        expiry: _chain(
            expiry,
            _contract(expiry, right="call", expiry=expiry, last=5.0, settle=5.0),
            spot=1280.0,
        ),
    }
    bars = _bars(d0, expiry, expiry_day=expiry, expiry_close=1280.0)
    cfg = _single_leg_cfg(d0, expiry, stop_pct=2.0, slippage_pct=0.05, exit_dte=1)
    result = _run(
        cfg,
        chain_loader=_loader(schedule),
        price_fetcher=StubPriceFetcher({"RELIANCE.NS": bars}),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == d1
    assert trade.exit_date == expiry
    assert trade.exit_date != trade.entry_date
    assert trade.exit_reason == "expiry"
    assert trade.details["hold_days"] == 1
