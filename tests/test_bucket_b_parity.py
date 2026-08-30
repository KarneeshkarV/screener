"""The converted Bucket B expressions reproduce their original Python bodies.

Stage 3b of ``docs/plans/unify-screen-backtest.md``. These eight strategies
needed a derived column each, so they are declared with ``bar_columns`` and
reference those columns by name in the expression. The Pine grammar does not
grow (plan D10); the column does.

As in Bucket A, each strategy keeps its original function unregistered and
these tests compare the expression's trades to it. Where a strategy uses the
numpy EMA, whose warm-up differs from Pine's, the comparison starts after
warm-up and the direction of the difference is pinned separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.strategies.plugins.awesome_oscillator import strat_awesome_oscillator
from screener.strategies.plugins.bb_breakout import strat_bb_breakout
from screener.strategies.plugins.donchian_breakout import strat_donchian_breakout
from screener.strategies.plugins.ma_cross_st_entry import strat_ma_cross_st_entry
from screener.strategies.plugins.ma_cross_st_exit import strat_ma_cross_st_exit
from screener.strategies.plugins.macd_rsi import strat_macd_rsi
from screener.strategies.plugins.parabolic_sar import strat_parabolic_sar
from screener.strategies.plugins.supertrend import strat_supertrend
from screener.strategies.plugins.supertrend_rsi import strat_supertrend_rsi
from screener.strategies.registry import STRATEGIES

# name -> (reference body, bars before the two sides must agree, high/low band)
# The band matters: a channel breakout has to clear the bar's own high, so a
# wide band silently produces zero trades and a vacuous test.
CASES = {
    "supertrend": (strat_supertrend, 20, 1.5),
    "parabolic_sar": (strat_parabolic_sar, 5, 1.5),
    "donchian_breakout": (strat_donchian_breakout, 25, 0.2),
    "bb_breakout": (strat_bb_breakout, 355, 1.5),
    "macd_rsi": (strat_macd_rsi, 40, 1.5),
    "ma_cross_st_entry": (strat_ma_cross_st_entry, 20, 1.5),
    "ma_cross_st_exit": (strat_ma_cross_st_exit, 20, 1.5),
    "supertrend_rsi": (strat_supertrend_rsi, 20, 1.5),
    "awesome_oscillator": (strat_awesome_oscillator, 40, 1.5),
}


def _ohlcv(n: int = 1400, band: float = 1.5) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    x = np.linspace(0, 60, n)
    close = 100 + np.linspace(0, 120, n) + np.sin(x) * 14 + np.sin(x * 3.7) * 5
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + np.sin(x / 2) * 0.5,
            "high": close + band,
            "low": close - band,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 10_000.0),
        }
    )


def _as_tuples(trades) -> list[tuple]:
    return [
        (t.entry_idx, t.exit_idx, round(t.entry_px, 9), round(t.exit_px, 9))
        for t in trades
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_expression_reproduces_the_original_body(name: str) -> None:
    reference, warmup, band = CASES[name]
    df = _ohlcv(band=band)

    expected = _as_tuples(reference(df))
    actual = _as_tuples(STRATEGIES[name](df))

    assert expected, f"{name}: reference body produced no trades on the fixture"

    settled_expected = [t for t in expected if t[0] >= warmup]
    settled_actual = [t for t in actual if t[0] >= warmup]
    assert settled_actual == settled_expected
    assert settled_expected, f"{name}: no trades survive the {warmup}-bar warm-up"


@pytest.mark.parametrize("name", sorted(CASES))
def test_declared_columns_are_reachable_from_the_expression(name: str) -> None:
    """The expression's identifiers resolve, so no name is a silent typo."""
    from screener.strategies.spec import resolve_strategy_spec

    spec = resolve_strategy_spec(name)
    assert spec is not None
    assert spec.bar_columns, f"{name}: declared no bar_columns"
    # Evaluating without raising PineNameError is the real assertion.
    trades = STRATEGIES[name](_ohlcv(band=CASES[name][2]))
    assert isinstance(trades, list)


def test_converted_names_stay_in_both_views() -> None:
    from screener.strategies.expressions import NAMED_STRATEGIES

    for name in CASES:
        assert name in NAMED_STRATEGIES, name
        assert name in STRATEGIES, name
