"""Offline tests for the Phase 4.1 options-backtest realism upgrades.

Covers configurable fill models, short-option margin (SPAN-like / Reg-T) with
portfolio utilisation tracking, explicit expiry settlement + assignment
metadata, and config-driven DTE/delta rolls. All values are hand-computed and
no network is touched.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from screener.options.bt_models import LegFill, OptionsBacktestConfig
from screener.options.models import OptionChain
from screener.options.position_backtest import (
    compute_regt_margin,
    compute_span_margin,
)
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


def _base_cfg(**overrides) -> OptionsBacktestConfig:
    values = dict(
        tickers=("RELIANCE",),
        start=date(2026, 7, 6),
        end=date(2026, 7, 8),
        structure="long_call",
        entry_expr="true",
    )
    values.update(overrides)
    return OptionsBacktestConfig(**values)


def _leg(**overrides) -> LegFill:
    values = dict(
        right="put",
        strike=1270.0,
        expiry=date(2026, 7, 28),
        side=-1,
        lots=1,
        lot_size=500.0,
        entry_price=5.0,
        entry_iv=0.25,
    )
    values.update(overrides)
    return LegFill(**values)


# --------------------------------------------------------------------------
# Fill models
# --------------------------------------------------------------------------


def test_cross_fill_buys_ask_sells_bid():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    # call_last=10 → bid 9.5 / ask 10.5; on d2 last=15 → bid 14.5 / ask 15.5.
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(end=d2, structure="long_call", fill_model="cross", target_pct=20.0)
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    # Long call: buy at the ask on entry, sell at the bid on exit.
    assert trade.legs[0].entry_price == 10.5
    assert trade.legs[0].exit_price == 14.5
    assert trade.exit_reason == "target"


def test_cross_fill_slippage_bps():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        end=d2,
        structure="long_call",
        fill_model="cross",
        slippage_bps=100.0,  # 1%
        target_pct=20.0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    trade = result.trades[0]
    # ask 10.5 * (1 + 100bps) = 10.605.
    assert trade.legs[0].entry_price == 10.605


def _quoteless_chain(day: date, *, last: float, settle: float) -> OptionChain:
    contract = _contract(
        right="call",
        strike=1270.0,
        bid=None,
        ask=None,
        last=last,
        settle=settle,
        expiry=date(2026, 7, 28),
        as_of=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
        symbol=f"C1270{day.isoformat()}",
    )
    return _chain(contract, spot=1275.0, day=day)


def test_illiquid_spread_proxied_from_settle_dispersion():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    # No quotes: mid = last = 10, dispersion = |settle - last| = 1 → half 0.5,
    # so the proxied ask is 10.5 and the long entry buys there.
    schedule = {
        d1: _quoteless_chain(d1, last=10.0, settle=11.0),
        d2: _quoteless_chain(d2, last=10.0, settle=11.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(end=d2, structure="long_call", fill_model="cross")
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    assert result.trades[0].legs[0].entry_price == 10.5


# --------------------------------------------------------------------------
# Margin models
# --------------------------------------------------------------------------


def test_regt_margin_short_put_hand_computed():
    cfg = _base_cfg()
    leg = _leg(side=-1, right="put", strike=1270.0, entry_price=5.0)
    # OTM amount = spot - strike = 5; per-unit = max(0.20*1275 - 5, 0.10*1270)
    #            = max(250, 127) = 250; qty = 500; premium = 5*500 = 2500.
    margin = compute_regt_margin([leg], 1275.0, cfg)
    assert margin == 250.0 * 500 + 2500.0  # 127500


def test_regt_margin_long_leg_is_premium_only():
    cfg = _base_cfg()
    leg = _leg(side=1, right="call", strike=1270.0, entry_price=10.0)
    assert compute_regt_margin([leg], 1275.0, cfg) == 10.0 * 500


def test_regt_margin_uses_current_marks_not_entry():
    """Short-option premium spike must raise the Reg-T requirement."""
    cfg = _base_cfg()
    leg = _leg(side=-1, right="put", strike=1270.0, entry_price=2.0)
    base = compute_regt_margin([leg], 1275.0, cfg)
    spiked = compute_regt_margin([leg], 1275.0, cfg, marks=[40.0])
    # Per-unit floor is unchanged; only the premium component moves by 38*qty.
    assert spiked - base == 38.0 * 500.0
    assert spiked > base


def test_regt_margin_call_minimum_uses_spot():
    """CBOE naked-call minimum is 10% of underlying, not strike."""
    cfg = _base_cfg()
    # Deep OTM short call: strike 2000, spot 1000 → OTM=1000.
    # regt_pct*spot - OTM = 200 - 1000 = -800 → min basis binds.
    # Strike basis would be 0.10*2000=200; spot basis is 0.10*1000=100.
    leg = _leg(side=-1, right="call", strike=2000.0, entry_price=1.0)
    margin = compute_regt_margin([leg], 1000.0, cfg)
    assert margin == 100.0 * 500 + 1.0 * 500  # min_basis*qty + premium


def test_span_margin_positive_and_scales_with_lots():
    cfg = _base_cfg()
    one = compute_span_margin([_leg(side=-1, lots=1)], 1275.0, date(2026, 7, 8), cfg)
    two = compute_span_margin([_leg(side=-1, lots=2)], 1275.0, date(2026, 7, 8), cfg)
    assert one > 0.0
    # Exposure floor alone scales linearly; total is close to 2x.
    assert two > 1.9 * one
    # Margin must clear the exposure floor for the short leg.
    exposure = cfg.span_exposure_pct * 1275.0 * 500
    assert one >= exposure


def test_margin_utilisation_tracked_in_result():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(end=d2, structure="short_put", margin_model="regt")
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.peak_margin > 0.0
    assert result.peak_margin_utilization > 0.0
    assert not result.margin_curve.empty


def test_span_margin_run_populates_curve():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(end=d2, structure="short_straddle", margin_model="span")
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.peak_margin > 0.0
    assert (result.margin_curve > 0).any()


def test_margin_cap_blocks_entry():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    # Reg-T margin (~127500) far exceeds 50% of the 100k default capital.
    cfg = _base_cfg(
        end=d2,
        structure="short_put",
        margin_model="regt",
        margin_cap_pct=0.5,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades == []
    assert any("margin cap" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------
# Expiry mechanics
# --------------------------------------------------------------------------


def _expiry_call_chain(
    day: date, expiry: date, *, settle: float, spot: float
) -> OptionChain:
    contract = _contract(
        right="call",
        strike=1270.0,
        bid=None,
        ask=None,
        last=settle,
        settle=settle,
        expiry=expiry,
        as_of=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
        symbol=f"C1270{day.isoformat()}",
    )
    return _chain(contract, spot=spot, day=day)


def test_settlement_uses_official_settle_price():
    d0, d1, d2 = date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 9)
    expiry = d2
    # Enter d1 (signal on d0), one day before expiry; hold to the d2 expiry.
    # Underlying closes at 1280 → intrinsic 10, but official settle is 7.
    schedule = {
        d1: _expiry_call_chain(d1, expiry, settle=8.0, spot=1276.0),
        d2: _expiry_call_chain(d2, expiry, settle=7.0, spot=1280.0),
    }
    bars = _underlying_bars(d0, d2, start_px=1275.0, end_px=1280.0)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d2,
        structure="long_call",
        exit_dte=0,
        settlement="settle",
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.legs[0].exit_price == 7.0  # settle, not intrinsic 10
    assert trade.details["settlement"] == "cash"
    assert trade.details["settlement_price"] == "settle"
    assert "call:1270" in trade.details["itm_legs"]
    assert trade.details["assigned_legs"] == []  # long leg is exercised, not assigned


def test_physical_settlement_records_short_assignment():
    d0, d1, d2 = date(2026, 7, 7), date(2026, 7, 8), date(2026, 7, 9)
    expiry = d2
    schedule = {
        d1: _expiry_call_chain(d1, expiry, settle=8.0, spot=1276.0),
        d2: _expiry_call_chain(d2, expiry, settle=9.0, spot=1280.0),
    }
    bars = _underlying_bars(d0, d2, start_px=1275.0, end_px=1280.0)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d2,
        structure="short_call",
        exit_dte=0,
        physical_settlement=True,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    trade = result.trades[0]
    assert trade.details["settlement"] == "physical"
    # Short call expiring ITM (1280 > 1270) is assigned.
    assert trade.details["assigned_legs"] == ["call:1270"]
    # Default settlement=intrinsic → exit at intrinsic 10.
    assert trade.legs[0].exit_price == 10.0


def test_expiry_settles_against_last_session_on_or_before_expiry():
    """When the loop day is past expiry, settle on the last ≤ expiry close."""
    # Expiry is Sunday (not a weekday trading day). Position is still open on
    # the following Monday with dte=-1; intrinsic must use Friday's close.
    expiry = date(2026, 7, 12)  # Sunday
    d0 = date(2026, 7, 8)  # Wed signal
    d1 = date(2026, 7, 9)  # Thu entry
    d_fri = date(2026, 7, 10)
    d_mon = date(2026, 7, 13)
    # Flat option marks; settlement is pure intrinsic against underlying.
    schedule = {
        d1: _expiry_call_chain(d1, expiry, settle=20.0, spot=1290.0),
        d_fri: _expiry_call_chain(d_fri, expiry, settle=25.0, spot=1300.0),
        d_mon: _expiry_call_chain(d_mon, expiry, settle=50.0, spot=1400.0),
    }
    # Force known closes: Friday 1300 → intrinsic 30; Monday 1400 → 130.
    bars = _underlying_bars(d0, d_mon, start_px=1275.0, end_px=1275.0)
    bars.loc[pd.Timestamp(d_fri), "close"] = 1300.0
    bars.loc[pd.Timestamp(d_mon), "close"] = 1400.0
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d_mon,
        structure="long_call",
        exit_dte=0,
        settlement="intrinsic",
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.exit_reason == "expiry"
    assert trade.exit_date == d_mon
    # strike 1270, settle-close 1300 → intrinsic 30 (not Monday's 130).
    assert trade.legs[0].exit_price == 30.0


# --------------------------------------------------------------------------
# Entry-day exits + max_hold semantics
# --------------------------------------------------------------------------


def test_entry_day_stop_suppressed_with_slippage():
    """Slippage-only mark/fill gap must not fire stop on the entry session."""
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    # Flat chain: entry fill is inflated by slippage; mid mark stays at last.
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        end=d2,
        structure="long_call",
        stop_pct=1.0,  # 1% — smaller than the ~1.96% slip gap at 2% slip
        slippage_pct=0.02,
        exit_dte=0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.entry_date == d1
    # Must survive the entry session; stop may fire on d2 against the same gap.
    assert trade.exit_date == d2
    assert trade.exit_reason == "stop"


def test_max_hold_one_exits_day_after_entry():
    """max_hold=1 means exit on the first trading day after entry."""
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=11.0, put_last=4.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        end=d2,
        structure="long_call",
        max_hold=1,
        exit_dte=0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.trades
    trade = result.trades[0]
    assert trade.entry_date == d1
    assert trade.exit_date == d2
    assert trade.exit_reason == "time"
    assert trade.details["hold_days"] == 1


# --------------------------------------------------------------------------
# Roll rules
# --------------------------------------------------------------------------


def _two_expiry_chain(
    day: date, near: date, far: date, *, spot: float = 1275.0
) -> OptionChain:
    contracts = []
    for expiry in (near, far):
        for right, strike in (("call", 1270.0), ("put", 1270.0)):
            contracts.append(
                _contract(
                    right=right,
                    strike=strike,
                    last=10.0,
                    bid=9.5,
                    ask=10.5,
                    settle=10.0,
                    expiry=expiry,
                    iv=0.25,
                    as_of=datetime.combine(
                        day, datetime.min.time(), tzinfo=timezone.utc
                    ),
                    symbol=f"{right}{strike}{expiry.isoformat()}{day.isoformat()}",
                )
            )
    return _chain(*contracts, spot=spot, day=day)


def test_roll_at_dte_exits_and_reenters_far_expiry():
    d0, d1, d2, d3, d4 = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
        date(2026, 7, 9),
        date(2026, 7, 10),
    )
    near, far = d3, date(2026, 7, 24)
    schedule = {day: _two_expiry_chain(day, near, far) for day in (d1, d2, d3, d4)}
    bars = _underlying_bars(d0, d4)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d4,
        structure="long_call",
        roll_dte=1,  # near dte hits 1 on d2 → roll into far
        roll_expiry_rule="next",
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    reasons = [t.exit_reason for t in result.trades]
    assert "roll" in reasons
    rolled = next(t for t in result.trades if t.exit_reason == "roll")
    assert rolled.legs[0].expiry == near
    # A later position must live on the far expiry (rolled into it).
    assert any(t.legs[0].expiry == far for t in result.trades)


def test_roll_at_delta_triggers():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    near, far = date(2026, 7, 17), date(2026, 7, 31)
    schedule = {day: _two_expiry_chain(day, near, far) for day in (d1, d2)}
    # Keep the underlying near the 1270 strike so the short put stays ATM
    # (|delta| ~0.44 ≥ the 0.4 roll trigger); the default ramp would push spot
    # to ~1300 and leave the only strike deep OTM.
    bars = _underlying_bars(d0, d2, start_px=1272.0, end_px=1275.0)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d2,
        structure="short_put",
        roll_delta=0.4,  # ATM short put |delta| ~0.44 ≥ 0.4
        roll_expiry_rule="next",
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert any(t.exit_reason == "roll" for t in result.trades)


def test_roll_delta_atm_skips_reentry_that_still_breaches():
    """ATM roll replacement still ≥ roll_delta → warn once, stay closed."""
    d0, d1, d2, d3 = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
        date(2026, 7, 9),
    )
    near, far = date(2026, 7, 17), date(2026, 7, 31)
    schedule = {day: _two_expiry_chain(day, near, far) for day in (d1, d2, d3)}
    bars = _underlying_bars(d0, d3, start_px=1272.0, end_px=1275.0)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d3,
        structure="short_put",
        roll_delta=0.4,
        roll_expiry_rule="next",
        exit_dte=0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    roll_trades = [t for t in result.trades if t.exit_reason == "roll"]
    assert roll_trades
    # No same-session re-entry: nothing opens on a roll exit day.
    roll_exit_days = {t.exit_date for t in roll_trades}
    assert not any(t.entry_date in roll_exit_days for t in result.trades)
    assert any("roll re-entry skipped" in w and "delta" in w for w in result.warnings)
    # Warning is once per symbol even across multiple potential rolls.
    delta_warns = [w for w in result.warnings if "roll re-entry skipped" in w]
    assert len(delta_warns) == 1


def test_roll_same_expiry_skips_reentry():
    """roll_expiry_rule='next' falling back to the only expiry must not churn."""
    d0, d1, d2, d3 = (
        date(2026, 7, 6),
        date(2026, 7, 7),
        date(2026, 7, 8),
        date(2026, 7, 9),
    )
    only = date(2026, 7, 17)
    # Single-expiry chain: "next" falls back to the same contract month.
    schedule = {
        day: _chain(
            *[
                c
                for c in _two_expiry_chain(day, only, date(2026, 7, 31)).contracts
                if c.expiry == only
            ],
            spot=1275.0,
            day=day,
        )
        for day in (d1, d2, d3)
    }
    bars = _underlying_bars(d0, d3)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(
        start=d0,
        end=d3,
        structure="long_call",
        roll_dte=20,  # near always has dte ≤ 20 in this window → roll
        roll_expiry_rule="next",
        exit_dte=0,
    )
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    roll_trades = [t for t in result.trades if t.exit_reason == "roll"]
    assert roll_trades
    roll_exit_days = {t.exit_date for t in roll_trades}
    assert not any(t.entry_date in roll_exit_days for t in result.trades)
    assert any("roll re-entry skipped" in w and "expiry" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_defaults_leave_margin_fields_empty():
    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=15.0, put_last=3.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    cfg = _base_cfg(end=d2, structure="long_call", target_pct=20.0)
    result = _run(cfg, chain_loader=_synthetic_loader(schedule), price_fetcher=fetcher)
    assert result.margin_curve.empty
    assert result.peak_margin == 0.0
    # Default settlement adds no extra detail keys.
    assert "settlement" not in result.trades[0].details


def test_cli_realism_flags_smoke(monkeypatch):
    from click.testing import CliRunner

    from screener.options import cli as options_cli

    d0, d1, d2 = date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)
    schedule = {
        d1: _call_put_pair(d1, call_last=10.0, put_last=5.0),
        d2: _call_put_pair(d2, call_last=10.0, put_last=5.0),
    }
    bars = _underlying_bars(d0, d2)
    fetcher = StubPriceFetcher({"RELIANCE.NS": bars})
    monkeypatch.setattr("screener.options.position_backtest.is_trading_day", _WEEKDAY)
    res = CliRunner().invoke(
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
            "short_put",
            "--fill-model",
            "cross",
            "--slippage-bps",
            "5",
            "--margin-model",
            "regt",
            "--entry",
            "true",
        ],
        obj={
            "chain_loader": _synthetic_loader(schedule),
            "price_fetcher": fetcher,
        },
    )
    assert res.exit_code == 0, res.output
    assert "margin" in res.output.lower()
