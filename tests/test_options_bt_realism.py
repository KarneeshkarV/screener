"""Offline tests for the Phase 4.1 options-backtest realism upgrades.

Covers configurable fill models, short-option margin (SPAN-like / Reg-T) with
portfolio utilisation tracking, explicit expiry settlement + assignment
metadata, and config-driven DTE/delta rolls. All values are hand-computed and
no network is touched.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

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
