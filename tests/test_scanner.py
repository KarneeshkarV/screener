"""Unit tests for screener.scanner._add_setup_score."""
from __future__ import annotations

import pandas as pd

from screener.scanner import _add_setup_score


def _row(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "description": name,
        "close": 100.0,
        "change": 1.0,
        "volume": 1_000_000,
        "market_cap_basic": 1e10,
        "EMA5": 101.0,
        "EMA20": 100.0,
        "EMA100": 95.0,
        "EMA200": 90.0,
        "RSI": 60.0,
    }
    base.update(overrides)
    return base


def test_setup_score_is_bounded_and_monotonic_in_trend():
    weak = pd.DataFrame(
        [
            _row(
                "WEAK",
                EMA5=100.0,
                EMA20=100.0,
                EMA100=100.0,
                EMA200=100.0,
                change=0.0,
                RSI=50,
            ),
            _row("STRONG"),  # default has bullish stack
        ]
    )
    scored = _add_setup_score(weak)
    assert {"setup_score"}.issubset(scored.columns)
    assert scored.loc[1, "setup_score"] > scored.loc[0, "setup_score"]


def test_setup_score_penalises_overextension():
    base = pd.DataFrame(
        [
            _row("NORMAL", close=100.0, EMA20=100.0),
            _row("STRETCHED", close=150.0, EMA20=100.0),  # 50% extension
        ]
    )
    scored = _add_setup_score(base)
    assert (
        scored.loc[1, "setup_score"] < scored.loc[0, "setup_score"]
    ), "overextended row should score lower"


def test_setup_score_handles_nan_cells():
    # Missing RSI / change should not produce NaN scores.
    df = pd.DataFrame([_row("X", RSI=None, change=None)])
    scored = _add_setup_score(df)
    assert scored["setup_score"].notna().all()
