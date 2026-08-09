"""The price fetch must buy enough history for the strategy's own lookback.

A plugin strategy builds its indicator columns in ``prepare_bars`` and then
refers to them by bare name ("mom_12_1 > 0"). The expression parser therefore
measures a lookback of zero for it, while the eligibility gate downstream still
demands ``spec.required_lookback()`` bars before a ticker may be traded.

When the fetch was sized from the parser's number alone, that shortfall did not
raise - it silently made the first months of the backtest window untradable. A
one-year window on a 630-bar strategy produced zero trades and reported a 0.00
Sharpe as if the strategy had simply been flat.
"""

from __future__ import annotations

import pandas as pd

from screener.backtester.core import strategy_lookback_floor
from screener.backtester.warmup import _warmup_days_for_interval


def test_floor_reports_a_plugin_lookback_the_parser_cannot_see() -> None:
    from screener.backtester.pine import parse, required_lookback

    # Barroso's gate needs 252 + 126 + 252 bars, none of which the entry
    # expression reveals.
    assert required_lookback(parse("mom_12_1 > 0 and not momentum_high_vol")) == 0
    assert strategy_lookback_floor("momentum_12_1_volmanaged") == 630


def test_floor_is_zero_for_a_strategy_without_one() -> None:
    assert strategy_lookback_floor(None) == 0
    assert strategy_lookback_floor("no_such_strategy_at_all") == 0


def test_warmup_covers_the_declared_lookback_in_trading_bars() -> None:
    # The fetch is sized in calendar days but the gate counts trading bars, so
    # the padding has to survive weekends and holidays: ~252 bars a year.
    floor = strategy_lookback_floor("momentum_12_1_volmanaged")
    warmup_days = _warmup_days_for_interval(floor, "1d")
    trading_bars = warmup_days * 252 / 365.25
    assert trading_bars > floor


def test_a_short_window_still_leaves_the_whole_window_tradable() -> None:
    """The regression itself: warmup must not eat into the backtest window."""
    start = pd.Timestamp("2025-08-09")
    for strategy in (
        "momentum_12_1",
        "momentum_12_1_volmanaged",
        "momentum_12_1_dynamic",
    ):
        floor = strategy_lookback_floor(strategy)
        fetch_start = start - pd.Timedelta(days=_warmup_days_for_interval(floor, "1d"))
        bars_before_window = (start - fetch_start).days * 252 / 365.25
        assert bars_before_window > floor, strategy
