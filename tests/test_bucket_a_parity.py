"""The converted Bucket A expressions reproduce their original Python bodies.

Stage 3 of ``docs/plans/unify-screen-backtest.md`` converts five callable
strategies into expression strategies. Converting is only safe if the
expression means what the hand written body meant, so each strategy keeps its
original function, unregistered, and these tests compare the two.

One deliberate difference is asserted rather than hidden. The numpy ``ema``
in ``screener/indicators/plugins/ema.py`` seeds ``out[0] = x[0]`` and so emits
a value on every bar, while Pine's ``ema`` is
``ewm(span=n, adjust=False, min_periods=n)`` and is NaN until it has ``n``
bars. The Pine behaviour is the correct one - it refuses to signal off an EMA
seeded from a single bar - so the ema-based strategies are compared once both
sides have enough history, and the warm-up region is asserted separately to
pin that only Pine suppresses those early signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.strategies.plugins.ma_cross import strat_ma_cross
from screener.strategies.plugins.ma_cross_regime import strat_ma_cross_regime
from screener.strategies.plugins.macd_oscillator import strat_macd_oscillator
from screener.strategies.plugins.rsi_ema import strat_rsi_ema
from screener.strategies.plugins.rsi_reversion import strat_rsi_reversion
from screener.strategies.registry import STRATEGIES

# name -> (reference body, bars of history the Pine side needs before its
# answer can be compared to the numpy body's)
CASES = {
    "macd_oscillator": (strat_macd_oscillator, 21),
    "rsi_reversion": (strat_rsi_reversion, 14),
    "ma_cross": (strat_ma_cross, 20),
    "ma_cross_regime": (strat_ma_cross_regime, 600),
    "rsi_ema": (strat_rsi_ema, 600),
}


def _ohlcv(n: int = 1400) -> pd.DataFrame:
    """Trending, oscillating bars long enough to clear a 600-bar warm-up."""
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    x = np.linspace(0, 60, n)
    close = 100 + np.linspace(0, 120, n) + np.sin(x) * 14 + np.sin(x * 3.7) * 5
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + np.sin(x / 2) * 0.5,
            "high": close + 1.5,
            "low": close - 1.5,
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
    """After warm-up, the converted expression yields the same trades."""
    reference, warmup = CASES[name]
    df = _ohlcv()

    expected = _as_tuples(reference(df))
    actual = _as_tuples(STRATEGIES[name](df))

    # A fixture that produces no trades would pass this test while proving
    # nothing about the conversion.
    assert expected, f"{name}: reference body produced no trades on the fixture"

    settled_expected = [t for t in expected if t[0] >= warmup]
    settled_actual = [t for t in actual if t[0] >= warmup]
    assert settled_actual == settled_expected
    assert settled_expected, f"{name}: no trades survive the {warmup}-bar warm-up"


@pytest.mark.parametrize("name", sorted(CASES))
def test_only_the_pine_side_suppresses_warmup_signals(name: str) -> None:
    """Any pre-warm-up divergence is Pine declining to signal, never the reverse.

    The numpy bodies compute an EMA from bar 0, so they can enter on a average
    that has seen one bar. Pine returns NaN there and a NaN never passes a
    comparison. This pins the direction of that difference: the expression may
    have fewer early trades, never extra ones.
    """
    reference, warmup = CASES[name]
    df = _ohlcv()

    early_expected = {t[0] for t in _as_tuples(reference(df)) if t[0] < warmup}
    early_actual = {t[0] for t in _as_tuples(STRATEGIES[name](df)) if t[0] < warmup}

    assert early_actual <= early_expected


def test_converted_names_are_in_both_views() -> None:
    """One definition, both consumers: the whole point of the conversion."""
    from screener.strategies.expressions import NAMED_STRATEGIES

    for name in CASES:
        assert name in NAMED_STRATEGIES, name
        assert name in STRATEGIES, name
