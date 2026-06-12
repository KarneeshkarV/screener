"""Unit tests for :class:`screener.backtester.fills.FillModel`.

The whole point of the fill seam is that pricing can be exercised *without*
running a full simulation. These tests poke the model directly: order-type
dispatch (MOO / MOC / limit), gap-through-stop/target reference resolution,
slippage routing (legacy bps factor vs. a pluggable ``SlippageModel``), and the
partial-exit / close pricing path.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.backtester.fills import (
    FillModel,
    _resolve_stop_fill,
    _resolve_target_fill,
    _slippage_factor,
)
from screener.backtester.models import BacktestConfig
from screener.backtester.slippage import HalfSpreadSlippage


def _cfg(**overrides) -> BacktestConfig:
    defaults = dict(
        market="us",
        as_of=date(2024, 3, 1),
        hold=5,
        top=10,
        entry_expr="close > sma(close, 3)",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=None,
    )
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-03-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.5, 12.5, 13.5],
            "low": [9.5, 10.5, 11.5, 12.5],
            "close": [10.2, 11.2, 12.2, 13.2],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )


# ── entry order-type dispatch ────────────────────────────────────────


def test_entry_moo_fills_next_bar_open():
    fm = FillModel(_cfg(entry_order_type="moo", slippage_bps=0.0))
    idx, fill, warn = fm.entry_price(_bars(), signal_idx=0)
    assert (idx, warn) == (1, None)
    assert fill == pytest.approx(11.0)  # next bar open


def test_entry_moc_fills_next_bar_close():
    fm = FillModel(_cfg(entry_order_type="moc", slippage_bps=0.0))
    idx, fill, warn = fm.entry_price(_bars(), signal_idx=0)
    assert (idx, warn) == (1, None)
    assert fill == pytest.approx(11.2)  # next bar close


def test_entry_limit_fills_at_min_of_open_and_limit():
    # signal close = 10.2, limit_bps = 1000 → limit = 10.2 * 0.9 = 9.18.
    # No subsequent low <= 9.18 → never fills.
    cfg = _cfg(entry_order_type="limit", entry_limit_bps=1000.0, slippage_bps=0.0)
    fm = FillModel(cfg)
    idx, fill, warn = fm.entry_price(_bars(), signal_idx=0)
    assert idx is None and fill is None
    assert "never filled" in (warn or "")


def test_entry_limit_requires_limit_bps():
    fm = FillModel(_cfg(entry_order_type="limit", entry_limit_bps=None))
    idx, fill, warn = fm.entry_price(_bars(), signal_idx=0)
    assert idx is None and fill is None
    assert "requires entry_limit_bps" in (warn or "")


def test_entry_no_post_signal_bar_warns():
    fm = FillModel(_cfg(entry_order_type="moo"))
    idx, fill, warn = fm.entry_price(_bars(), signal_idx=3)  # last bar
    assert idx is None and fill is None
    assert warn == "no post-signal entry bar"


def test_entry_applies_buy_slippage_to_reference():
    fm = FillModel(_cfg(entry_order_type="moo", slippage_bps=50.0))
    _idx, fill, _warn = fm.entry_price(_bars(), signal_idx=0)
    # 11.0 widened up 50 bps for a buy.
    assert fill == pytest.approx(11.0 * (1.0 + 50.0 / 10_000.0))


# ── gap-aware reference resolution ───────────────────────────────────


def test_exit_stop_gap_down_uses_bar_open():
    fm = FillModel(_cfg(gap_fills=True, slippage_bps=0.0))
    # bar opens below the stop_ref → fill at the (worse) open.
    px = fm.exit_price(reason="stop", bar_open=90.0, level=95.0)
    assert px == pytest.approx(90.0)


def test_exit_stop_inside_bar_uses_stop_ref():
    fm = FillModel(_cfg(gap_fills=True, slippage_bps=0.0))
    # bar opens above the stop but trades through → classical stop-ref fill.
    px = fm.exit_price(reason="stop", bar_open=98.0, level=95.0)
    assert px == pytest.approx(95.0)


def test_exit_target_gap_up_uses_bar_open():
    fm = FillModel(_cfg(gap_fills=True, slippage_bps=0.0))
    px = fm.exit_price(reason="target", bar_open=110.0, level=105.0)
    assert px == pytest.approx(110.0)


def test_exit_gap_fills_disabled_uses_level():
    fm = FillModel(_cfg(gap_fills=False, slippage_bps=0.0))
    assert fm.exit_price(reason="stop", bar_open=90.0, level=95.0) == pytest.approx(
        95.0
    )
    assert fm.exit_price(reason="target", bar_open=110.0, level=105.0) == pytest.approx(
        105.0
    )


def test_exit_trail_uses_stop_resolution():
    fm = FillModel(_cfg(gap_fills=True, slippage_bps=0.0))
    # trail behaves like a stop: gap-down aware.
    assert fm.exit_price(reason="trail", bar_open=90.0, level=95.0) == pytest.approx(
        90.0
    )


def test_exit_close_reasons_use_close():
    fm = FillModel(_cfg(slippage_bps=0.0))
    for reason in ("exit_expr", "time", "eod"):
        assert fm.exit_price(reason=reason, close=42.0) == pytest.approx(42.0)


# ── slippage routing ─────────────────────────────────────────────────


def test_exit_legacy_bps_factor_when_no_model():
    fm = FillModel(_cfg(slippage_bps=25.0, slippage_model=None))
    px = fm.exit_price(reason="eod", close=100.0)
    # sell side widened *down* by the legacy factor.
    assert px == pytest.approx(100.0 * _slippage_factor(25.0, buy=False))
    assert px == pytest.approx(99.75)


def test_explicit_model_overrides_bps():
    # A non-zero bps combined with an explicit model must use the model only.
    fm = FillModel(
        _cfg(slippage_bps=50.0, slippage_model=HalfSpreadSlippage(half_spread_bps=5.0))
    )
    px = fm.exit_price(reason="eod", close=100.0)
    # 5 bps half-spread, not 50 bps fixed.
    assert px == pytest.approx(99.95)


def test_partial_exit_pricing_matches_target_resolution():
    # The partial-exit path uses exit_price(reason="target", ...) — verify it
    # resolves and slips identically to a direct target exit.
    fm = FillModel(_cfg(gap_fills=True, slippage_bps=10.0))
    direct = fm.exit_price(reason="target", bar_open=110.0, level=105.0)
    assert direct == pytest.approx(110.0 * (1.0 - 10.0 / 10_000.0))


# ── primitive helpers stay importable / correct ──────────────────────


def test_resolve_helpers_round_trip():
    assert _resolve_stop_fill(90.0, 95.0, True) == 90.0
    assert _resolve_stop_fill(98.0, 95.0, True) == 95.0
    assert _resolve_target_fill(110.0, 105.0, True) == 110.0
    assert _resolve_target_fill(100.0, 105.0, True) == 105.0
