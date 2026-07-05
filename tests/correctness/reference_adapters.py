"""Reconciliation layer between the screener and independent references.

Every convention difference that would otherwise cause a *false* test failure
is encoded here once, so the test bodies stay declarative and the reasoning is
reviewable in a single place. A green test that goes through these adapters
means the screener's *math* agrees with the reference — not that conventions
happen to line up.

Reconciliation rules captured here (confirmed empirically against the installed
``pandas_ta_classic`` 0.6.x and ``talib`` 0.6.x):

* EMA / RSI / ATR / MACD use Wilder/EWM recursion with a different *seed* than
  the references; they only agree on the converged *tail*. Compare past a
  warm-up cutoff with a mask.
* Bollinger Bands: screener uses population std (ddof=0). TA-Lib BBANDS and
  ``pandas_ta_classic.bbands`` also use ddof=0 → exact match (no rescale).
* OBV: TA-Lib seeds OBV[0] = volume[0]; the screener's ``_obv`` starts the
  cumulative sum at 0. They differ by a constant ``volume[0]`` → compare
  first-differences, which removes the constant.
* Supertrend direction: the screener uses the inverted convention
  ``direction < 0 == uptrend``; ``pandas_ta_classic`` uses ``+1 == uptrend``.
  Compare ``screener_dir`` against ``-ref_dir``.
* empyrical Sharpe / annual-volatility use sample std (ddof=1); the screener
  uses population std (ddof=0). Convert with ``sqrt((N-1)/N)``.

* Cross-engine (event-driven backtester vs vectorbt) cost/stop reconciliation:
  the event engine parameterises costs in **basis points** (``commission_bps``,
  ``slippage_bps``); vectorbt takes **fractions** (``fees``, ``slippage``).
  ``bps_to_fraction`` is the one-line conversion (10 bps = 0.001). The fair
  portfolio-return comparison chains each trade's **net** ratio
  ``exit_value / entry_cost`` (both already net of commission and slippage on
  the recorded fill) — ``net_compound_return`` — which is capital-independent
  and therefore equals vectorbt's fully-reinvested ``pf.total_return()`` exactly
  (both apply commission as a per-notional fraction on each side and compound
  multiplicatively across trades). See
  ``test_cross_engine_costs_stops.py`` for the derivation and the pinned
  stop-price / stop-entry-base reconciliation rules.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest


def require_talib():
    """Return the ``talib`` module or skip the test if it is not installed."""
    return pytest.importorskip("talib")


def bps_to_fraction(bps: float) -> float:
    """Convert an event-engine basis-point cost to a vectorbt fraction.

    The event engine expresses ``commission_bps`` / ``slippage_bps`` in basis
    points; vectorbt's ``fees`` / ``slippage`` are plain fractions. 10 bps maps
    to 0.001. This is the single place the unit convention is encoded.
    """
    return bps / 10_000.0


def net_compound_return(trades: Iterable) -> float:
    """Chain each trade's net ``exit_value / entry_cost`` ratio, minus 1.

    Both ``entry_cost`` (``shares * entry_price + entry_commission``) and
    ``exit_value`` (``shares * exit_price - exit_commission``) are already net
    of commission, and ``entry_price`` / ``exit_price`` are already net of
    slippage, so the per-trade ratio is the net multiplicative return on the
    capital committed to that trade. Because ``shares`` cancels, the ratio is
    capital-independent; chaining it reproduces vectorbt's fully-reinvested
    ``pf.total_return()`` exactly (the event engine's own per-slot sizing does
    not compound, which is why we recompute the compound from the ratios rather
    than reading an equity endpoint).
    """
    cap = 1.0
    for tr in trades:
        cap *= tr.exit_value / tr.entry_cost
    return cap - 1.0


def require_quantstats():
    return pytest.importorskip("quantstats")


def finite_tail_mask(*arrays: np.ndarray, start: int) -> np.ndarray:
    """Boolean mask: index >= ``start`` AND every array finite at that index.

    Used to compare recursive indicators only on their converged tail, ignoring
    warm-up regions where seeding conventions legitimately differ.
    """
    length = len(arrays[0])
    mask = np.arange(length) >= start
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask


def ddof0_from_ddof1(value: float, n: int) -> float:
    """Convert a sample-std (ddof=1) statistic to a population-std (ddof=0) one.

    empyrical's Sharpe/volatility divide by ``std(ddof=1)``; the screener divides
    by ``std(ddof=0)``. Scaling by ``sqrt((n-1)/n)`` makes them comparable.
    """
    if n < 2:
        return value
    return value * np.sqrt((n - 1) / n)


def equity_to_returns(equity) -> np.ndarray:
    """Simple period returns from an equity curve, matching ``_daily_returns``."""
    import pandas as pd

    return pd.Series(equity).pct_change().dropna().to_numpy()
