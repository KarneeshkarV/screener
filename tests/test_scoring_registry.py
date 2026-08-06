"""Registry completeness and per-criterion ranking philosophy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from screener.criteria import CRITERIA
from screener.scoring import (
    OUTPUT_SCORE_COLUMN,
    SCORERS,
    apply_score,
    get_scorer,
    resolve_scorer,
)


def test_every_criterion_has_a_scorer() -> None:
    missing = sorted(set(CRITERIA) - set(SCORERS))
    extra = sorted(set(SCORERS) - set(CRITERIA))
    assert missing == [], f"criteria without scorers: {missing}"
    assert extra == [], f"scorers without criteria: {extra}"


def test_resolve_scorer_single_returns_named_recipe() -> None:
    spec = resolve_scorer(["value"])
    assert spec.name == "value"
    assert "price_earnings_ttm" in spec.columns
    assert "cheap" in spec.description.lower() or "p/e" in spec.description.lower()


def test_resolve_scorer_multi_blends_and_unions_columns() -> None:
    spec = resolve_scorer(["value", "quality"])
    assert spec.name == "value+quality"
    assert "price_earnings_ttm" in spec.columns
    assert "return_on_equity" in spec.columns
    assert "debt_to_equity" in spec.columns


def test_resolve_scorer_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        resolve_scorer([])


def test_value_ranks_cheaper_pe_higher() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "CHEAP",
                "close": 50.0,
                "change": 0.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": 8.0,
            },
            {
                "name": "DEAR",
                "close": 50.0,
                "change": 0.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": 25.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("value")).sort_values(
        OUTPUT_SCORE_COLUMN, ascending=False
    )
    assert scored.iloc[0]["name"] == "CHEAP"
    assert scored.iloc[0][OUTPUT_SCORE_COLUMN] > scored.iloc[1][OUTPUT_SCORE_COLUMN]


def test_quality_ranks_higher_roe_and_lower_debt() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "SOLID",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 25.0,
                "debt_to_equity": 0.2,
                "EMA20": 105.0,
                "EMA200": 90.0,
            },
            {
                "name": "WEAK",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 8.0,
                "debt_to_equity": 2.0,
                "EMA20": 95.0,
                "EMA200": 100.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("quality")).sort_values(
        OUTPUT_SCORE_COLUMN, ascending=False
    )
    assert scored.iloc[0]["name"] == "SOLID"


def test_breakout_ranks_closer_to_52w_high() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "NEAR",
                "close": 98.0,
                "change": 2.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_52_week_high": 100.0,
                "relative_volume_10d_calc": 2.0,
                "EMA20": 95.0,
                "EMA200": 80.0,
                "RSI": 65.0,
            },
            {
                "name": "FAR",
                "close": 70.0,
                "change": 2.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_52_week_high": 100.0,
                "relative_volume_10d_calc": 2.0,
                "EMA20": 68.0,
                "EMA200": 60.0,
                "RSI": 65.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("breakout")).sort_values(
        OUTPUT_SCORE_COLUMN, ascending=False
    )
    assert scored.iloc[0]["name"] == "NEAR"


def test_composite_value_quality_average() -> None:
    # Two rows identical on liquidity; CHEAP has better PE, QUAL has better ROE/DE.
    # Blended scorer should produce a finite score for both.
    df = pd.DataFrame(
        [
            {
                "name": "CHEAP",
                "close": 50.0,
                "change": 0.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": 8.0,
                "return_on_equity": 10.0,
                "debt_to_equity": 1.5,
                "EMA20": 50.0,
                "EMA200": 48.0,
            },
            {
                "name": "QUAL",
                "close": 50.0,
                "change": 0.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": 22.0,
                "return_on_equity": 30.0,
                "debt_to_equity": 0.1,
                "EMA20": 52.0,
                "EMA200": 40.0,
            },
        ]
    )
    scored = apply_score(df, resolve_scorer(["value", "quality"]))
    assert scored[OUTPUT_SCORE_COLUMN].notna().all()
    assert (scored[OUTPUT_SCORE_COLUMN] > 0).all()


def test_negative_pe_does_not_win_value_rank() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "LOSS",
                "close": 40.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": -5.0,
            },
            {
                "name": "OK",
                "close": 40.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "price_earnings_ttm": 12.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("value")).sort_values(
        OUTPUT_SCORE_COLUMN, ascending=False
    )
    assert scored.iloc[0]["name"] == "OK"
