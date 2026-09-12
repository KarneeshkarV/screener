"""Per-ticker ATR stop distance: ``stop_mode="atr"``.

The flat ``stop_loss`` gives a quiet large cap and a volatile microcap the same
stop, so the two positions risk very different shares of themselves. These pin
that ``atr`` mode sets each slot's stop from that ticker's own recent range,
that the ``atr_risk`` sizer measures risk against the stop that will actually
fire, and that the flat mode is untouched.

The frames are deliberately degenerate: a constant close with high/low a fixed
``spread`` either side makes every true range ``2 * spread``, so Wilder ATR is
exactly ``2 * spread`` once warmed up and every expected level below is an
exact number rather than a recorded one.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from screener.backtester.models import BacktestConfig
from screener.backtester.portfolio import Portfolio
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.sizing import entry_budget_for
from screener.backtester.stops import entry_stop_fraction, entry_stop_price
from tests.conftest import StubPriceFetcher

_START = "2024-01-01"
_N = 60
_INDEX = pd.bdate_range(_START, periods=_N)
# wilder_atr(min_periods=14) is NaN through position 12 and defined from 13 on.
_WARM = 20


def _flat_bars(close: float, spread: float, n: int = _N) -> pd.DataFrame:
    """Constant close with a fixed high/low spread, so ATR == ``2 * spread``."""
    idx = pd.bdate_range(_START, periods=n)
    close_s = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s,
            "high": close_s + spread,
            "low": close_s - spread,
            "close": close_s,
            "volume": pd.Series(1_000_000.0, index=idx, dtype=float),
        }
    )


def _cfg(**overrides) -> BacktestConfig:
    base = dict(
        market="us",
        as_of=_INDEX[-1].date(),
        benchmark="SPY",
        hold=5,
        top=1,
        strategy_name=None,
        entry_expr="sma(close, 20) > 0",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        tickers=("AAA",),
    )
    base.update(overrides)
    return BacktestConfig(**base)


# ---------------------------------------------------------------------------
# The distance itself
# ---------------------------------------------------------------------------


def test_atr_mode_widens_the_stop_for_the_more_volatile_ticker():
    """Same config, same price, different range: different stop. The point."""
    cfg = _cfg(stop_mode="atr", stop_atr_multiple=2.0)

    quiet = entry_stop_price(cfg, 100.0, _flat_bars(100.0, spread=1.0), _WARM)
    volatile = entry_stop_price(cfg, 100.0, _flat_bars(100.0, spread=5.0), _WARM)

    # ATR is 2 and 10; 2x that over a 100 close is 4% and 20%.
    assert quiet == pytest.approx(96.0)
    assert volatile == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("multiple", "expected"), [(0.5, 0.01), (1.0, 0.02), (1.5, 0.03), (3.0, 0.06)]
)
def test_the_multiple_scales_the_distance_linearly(multiple: float, expected: float):
    cfg = _cfg(stop_mode="atr", stop_atr_multiple=multiple)
    fraction = entry_stop_fraction(cfg, _flat_bars(100.0, spread=1.0), _WARM)
    assert fraction == pytest.approx(expected)


def test_the_window_selects_the_atr_lookback():
    """A 5-bar window is defined where the default 14-bar one is still NaN."""
    bars = _flat_bars(100.0, spread=1.0)
    short = _cfg(stop_mode="atr", stop_atr_multiple=2.0, stop_atr_window=5)
    default = _cfg(stop_mode="atr", stop_atr_multiple=2.0)

    assert entry_stop_fraction(short, bars, 6) == pytest.approx(0.04)
    assert entry_stop_fraction(default, bars, 6) is None


def test_pct_mode_is_untouched_by_the_new_fields():
    cfg = _cfg(stop_loss=0.08)
    bars = _flat_bars(100.0, spread=5.0)
    assert entry_stop_price(cfg, 100.0, bars, _WARM) == pytest.approx(92.0)
    assert entry_stop_price(_cfg(), 100.0, bars, _WARM) is None


def test_an_undefined_atr_falls_back_to_the_flat_stop():
    """Short warmup must not silently leave the slot unprotected."""
    bars = _flat_bars(100.0, spread=1.0)
    with_flat = _cfg(stop_mode="atr", stop_atr_multiple=2.0, stop_loss=0.08)
    without = _cfg(stop_mode="atr", stop_atr_multiple=2.0)

    # Position 5 is inside the 14-bar warmup, so ATR is NaN there.
    assert entry_stop_price(with_flat, 100.0, bars, 5) == pytest.approx(92.0)
    assert entry_stop_price(without, 100.0, bars, 5) is None


def test_a_stop_wider_than_the_entry_price_is_clamped_to_the_position():
    """20x ATR on a 20%-range name is 4x the price; a negative stop never fires."""
    cfg = _cfg(stop_mode="atr", stop_atr_multiple=20.0)
    bars = _flat_bars(100.0, spread=10.0)
    assert entry_stop_fraction(cfg, bars, _WARM) == pytest.approx(1.0)
    assert entry_stop_price(cfg, 100.0, bars, _WARM) == 0.0


def test_atr_mode_requires_a_multiple():
    with pytest.raises(ValidationError, match="stop_atr_multiple"):
        _cfg(stop_mode="atr")


def test_fixed_risk_still_rejects_an_atr_stop():
    """'fixed_risk' divides by one flat fraction; 'atr_risk' is its ATR pair."""
    with pytest.raises(ValidationError, match="fixed_risk"):
        _cfg(sizing_rule="fixed_risk", stop_mode="atr", stop_atr_multiple=2.0)


# ---------------------------------------------------------------------------
# Sizing reads the stop that will actually fire
# ---------------------------------------------------------------------------


def test_atr_risk_sizes_against_the_configured_stop_not_the_sizing_knobs():
    """The regression this fixes: sizing assumed 2x ATR while the stop was 1x."""
    portfolio = Portfolio(100_000.0, 1)
    cfg = _cfg(
        sizing_rule="atr_risk",
        sizing_risk_pct=0.01,
        sizing_atr_multiple=2.0,  # deliberately disagrees with the stop
        stop_mode="atr",
        stop_atr_multiple=1.0,
    )
    bars = _flat_bars(100.0, spread=1.0)

    budget = entry_budget_for(cfg, portfolio, bars, _WARM)
    fraction = entry_stop_fraction(cfg, bars, _WARM)

    # Losing the stop costs budget * fraction, which must be 1% of equity.
    assert fraction == pytest.approx(0.02)
    assert budget * fraction == pytest.approx(1_000.0)


def test_atr_risk_keeps_the_sizing_knobs_when_the_stop_is_flat():
    """No ATR stop configured means no behaviour change for existing runs."""
    portfolio = Portfolio(100_000.0, 1)
    cfg = _cfg(sizing_rule="atr_risk", sizing_risk_pct=0.01, sizing_atr_multiple=2.0)
    bars = _flat_bars(100.0, spread=1.0)

    # equity * 0.01 / (2 * 2 / 100) = 25_000, under the 100_000 slot ceiling.
    assert entry_budget_for(cfg, portfolio, bars, _WARM) == pytest.approx(25_000.0)


# ---------------------------------------------------------------------------
# End to end through the rolling engine
# ---------------------------------------------------------------------------


def _crash_bars(close: float, spread: float, drop_at: int, low: float) -> pd.DataFrame:
    """Flat for the warmup, then bars that trade all the way down to ``low``."""
    bars = _flat_bars(close, spread)
    bars.iloc[drop_at:, bars.columns.get_loc("low")] = low
    bars.iloc[drop_at:, bars.columns.get_loc("close")] = low
    return bars


def _first_stop_per_ticker(trades) -> dict:
    """The opening stop of each name. Later re-entries price off later ATRs."""
    first: dict = {}
    for trade in trades:
        if trade.exit_reason == "stop":
            first.setdefault(trade.ticker, trade)
    return first


def _crash_fetcher() -> StubPriceFetcher:
    return StubPriceFetcher(
        {
            # Same price and same crash; only the range differs.
            "AAA": _crash_bars(100.0, spread=1.0, drop_at=40, low=70.0),
            "BBB": _crash_bars(100.0, spread=5.0, drop_at=40, low=70.0),
            "SPY": _flat_bars(400.0, spread=1.0),
        }
    )


def test_the_rolling_engine_stops_each_ticker_at_its_own_atr_level():
    cfg = _cfg(
        tickers=("AAA", "BBB"),
        top=2,
        hold=200,  # keep the time exit out of the way
        stop_mode="atr",
        stop_atr_multiple=2.0,
    )

    result = run_rolling_backtest(
        cfg,
        _crash_fetcher(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )

    stops = _first_stop_per_ticker(result.trades)
    assert set(stops) == {"AAA", "BBB"}
    # Entries fill at the flat 100 open, so the stop levels are exact.
    assert stops["AAA"].exit_price == pytest.approx(96.0)
    assert stops["BBB"].exit_price == pytest.approx(80.0)


def test_the_rolling_engine_keeps_the_flat_stop_identical():
    """A pct-mode run must produce the level it always did."""
    cfg = _cfg(tickers=("AAA", "BBB"), top=2, hold=200, stop_loss=0.10)

    result = run_rolling_backtest(
        cfg,
        _crash_fetcher(),
        start_date=_INDEX[0].date(),
        end_date=_INDEX[-1].date(),
    )

    stops = _first_stop_per_ticker(result.trades)
    assert set(stops) == {"AAA", "BBB"}
    for trade in stops.values():
        assert trade.exit_price == pytest.approx(90.0)


def test_the_atr_series_is_computed_once_per_frame():
    """Both consumers memo under the same key, so one frame builds one ATR."""
    from screener.backtester.stops import atr_series

    bars = _flat_bars(100.0, spread=1.0)
    cache: dict = {}
    first = atr_series(bars, 14, cache)
    second = atr_series(bars, 14, cache)

    assert first is second
    assert list(cache) == [("atr", 14)]
    assert math.isclose(float(first[_WARM]), 2.0)
