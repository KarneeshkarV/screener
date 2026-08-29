"""Registry completeness and per-criterion ranking philosophy tests."""

from __future__ import annotations

import pandas as pd
import pytest

from screener import screen_workflow
from screener.criteria import CRITERIA
from screener.screen_candidates import UnscreenableStrategyError
from screener.screen_workflow import ScreenRequest, run_screen_workflow
from screener.scoring import (
    DEFAULT_SCORER_NAME,
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


def test_negative_debt_to_equity_does_not_win_low_debt_rank() -> None:
    # Negative D/E means negative shareholder equity, not a pristine balance
    # sheet — it must not outrank a genuinely low-debt name.
    df = pd.DataFrame(
        [
            {
                "name": "NEGEQUITY",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 25.0,
                "debt_to_equity": -3.5,
                "EMA20": 105.0,
                "EMA200": 90.0,
            },
            {
                "name": "LOWDEBT",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 25.0,
                "debt_to_equity": 0.2,
                "EMA20": 105.0,
                "EMA200": 90.0,
            },
        ]
    )
    for name in ("quality", "dividend"):
        scored = apply_score(
            df.assign(dividend_yield_recent=2.0, price_earnings_ttm=15.0),
            get_scorer(name),
        ).set_index("name")
        assert (
            scored.loc["LOWDEBT", OUTPUT_SCORE_COLUMN]
            > scored.loc["NEGEQUITY", OUTPUT_SCORE_COLUMN]
        ), name


def test_zero_debt_still_earns_top_low_debt_rank() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "DEBTFREE",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 20.0,
                "debt_to_equity": 0.0,
                "EMA20": 105.0,
                "EMA200": 90.0,
            },
            {
                "name": "LEVERED",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "return_on_equity": 20.0,
                "debt_to_equity": 0.9,
                "EMA20": 105.0,
                "EMA200": 90.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("quality")).set_index("name")
    assert (
        scored.loc["DEBTFREE", OUTPUT_SCORE_COLUMN]
        > scored.loc["LEVERED", OUTPUT_SCORE_COLUMN]
    )


def test_missing_rvol_is_not_ranked_below_the_worst_observed_rvol() -> None:
    # Partial RVOL coverage: the row without RVOL falls back to change energy
    # rather than being pushed under the lowest actual RVOL in the frame, which
    # is what the old frame-level ``percentile(rvol)`` switch did.
    def _row(name: str, rvol: float, change: float) -> dict[str, object]:
        return {
            "name": name,
            "close": 98.0,
            "change": change,
            "volume": 1_000_000.0,
            "market_cap_basic": 1e9,
            "price_52_week_high": 100.0,
            "relative_volume_10d_calc": rvol,
            "EMA20": 95.0,
            "EMA200": 80.0,
            "RSI": 65.0,
        }

    df = pd.DataFrame(
        [
            _row("HIGHRVOL", 3.0, 0.0),
            _row("LOWRVOL", 1.0, 0.0),
            _row("NORVOL", float("nan"), 10.0),
        ]
    )
    scored = apply_score(df, get_scorer("near_52_high")).set_index("name")
    assert (
        scored.loc["NORVOL", OUTPUT_SCORE_COLUMN]
        > scored.loc["LOWRVOL", OUTPUT_SCORE_COLUMN]
    )


def test_full_rvol_coverage_still_ranks_by_rvol() -> None:
    rows = [
        {
            "name": "SURGE",
            "close": 98.0,
            "change": 1.0,
            "volume": 1_000_000.0,
            "market_cap_basic": 1e9,
            "price_52_week_high": 100.0,
            "relative_volume_10d_calc": 4.0,
            "EMA20": 95.0,
            "EMA200": 80.0,
            "RSI": 65.0,
        },
        {
            "name": "QUIET",
            "close": 98.0,
            "change": 1.0,
            "volume": 1_000_000.0,
            "market_cap_basic": 1e9,
            "price_52_week_high": 100.0,
            "relative_volume_10d_calc": 0.5,
            "EMA20": 95.0,
            "EMA200": 80.0,
            "RSI": 65.0,
        },
    ]
    scored = apply_score(pd.DataFrame(rows), get_scorer("breakout")).set_index("name")
    assert (
        scored.loc["SURGE", OUTPUT_SCORE_COLUMN]
        > scored.loc["QUIET", OUTPUT_SCORE_COLUMN]
    )


def test_momentum_value_ignores_unrelated_extra_columns() -> None:
    # EMA100 is declared by the scorer, so adding columns from another
    # criterion must not change the momentum_value ranking recipe.
    rows = [
        {
            "name": "A",
            "close": 100.0,
            "change": 1.0,
            "volume": 1_000_000.0,
            "market_cap_basic": 1e9,
            "price_earnings_ttm": 12.0,
            "RSI": 60.0,
            "EMA5": 99.0,
            "EMA20": 97.0,
            "EMA100": 92.0,
            "EMA200": 85.0,
        },
        {
            "name": "B",
            "close": 100.0,
            "change": 1.0,
            "volume": 1_000_000.0,
            "market_cap_basic": 1e9,
            "price_earnings_ttm": 30.0,
            "RSI": 45.0,
            "EMA5": 98.0,
            "EMA20": 99.0,
            "EMA100": 101.0,
            "EMA200": 103.0,
        },
    ]
    df = pd.DataFrame(rows)
    base = apply_score(df, get_scorer("momentum_value"))[OUTPUT_SCORE_COLUMN]
    widened = apply_score(
        df.assign(price_52_week_high=110.0, relative_volume_10d_calc=1.2),
        get_scorer("momentum_value"),
    )[OUTPUT_SCORE_COLUMN]
    pd.testing.assert_series_equal(base, widened)


def test_proximity_to_high_stays_float_dtype() -> None:
    from screener.scoring.components import proximity_to_high

    close = pd.Series([50.0, 90.0, 10.0])
    high = pd.Series([100.0, 0.0, float("nan")])
    result = proximity_to_high(close, high)
    assert result.dtype == float
    assert result.tolist() == [0.5, 0.0, 0.0]


def test_resolve_scorer_non_strict_falls_back_to_default() -> None:
    with pytest.raises(KeyError):
        resolve_scorer(["definitely_not_a_criterion"])
    spec = resolve_scorer(["definitely_not_a_criterion"], strict=False)
    assert spec.name == DEFAULT_SCORER_NAME
    # A valid name still resolves normally in non-strict mode.
    assert resolve_scorer(["value"], strict=False).name == "value"


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


def test_momentum_12_1_is_bar_derived_not_a_snapshot_recipe() -> None:
    # The 12-1 recipe moved to the shared price-only layer so the screen and
    # the backtest report one number; see tests/test_score_unification.py for
    # the parity check. Nothing here may read TradingView's Perf.* snapshot.
    spec = get_scorer("momentum_12_1")
    assert spec.bar_score is not None
    assert spec.columns == ()


def test_mark_minervini_ranks_full_trend_stack_and_near_high() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "TEMPLATE",
                "close": 110.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "SMA50": 105.0,
                "SMA150": 100.0,
                "SMA200": 95.0,
                "price_52_week_high": 115.0,
                "price_52_week_low": 70.0,
            },
            {
                "name": "LOOSE",
                "close": 100.0,
                "volume": 1_000_000.0,
                "market_cap_basic": 1e9,
                "SMA50": 98.0,
                "SMA150": 99.0,
                "SMA200": 97.0,
                "price_52_week_high": 130.0,
                "price_52_week_low": 60.0,
            },
        ]
    )
    scored = apply_score(df, get_scorer("mark_minervini")).set_index("name")
    assert (
        scored.loc["TEMPLATE", OUTPUT_SCORE_COLUMN]
        > scored.loc["LOOSE", OUTPUT_SCORE_COLUMN]
    )


def _request(*, criteria: tuple[str, ...], order_by: str) -> ScreenRequest:
    return ScreenRequest(
        market="india",
        criteria_names=criteria,
        limit=5,
        order_by=order_by,
        output_csv=True,
        detail=False,
        refresh=False,
        cache_ttl="15m",
        report_path=None,
    )


def test_screen_sorted_by_a_column_never_resolves_a_scorer(monkeypatch) -> None:
    # `--sort volume` computes no score at all, so a scoring refusal must not
    # fire: the run has no ranking recipe to refuse. Both names here are
    # filters-only criteria, which is what makes combining them legal.
    captured: dict[str, object] = {}

    def fake_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
        captured.update(kwargs)
        return 1, pd.DataFrame({"name": ["AAA"], "description": ["AAA Ltd"]})

    monkeypatch.setattr(screen_workflow, "scan", fake_scan)

    outcome = run_screen_workflow(
        _request(criteria=("ema", "value"), order_by="volume")
    )

    assert outcome.df["name"].tolist() == ["AAA"]
    assert captured["scorer"] is None


@pytest.mark.parametrize("order_by", ["volume", OUTPUT_SCORE_COLUMN])
def test_screen_refuses_a_strategy_alias_combined_with_a_filter_criterion(
    monkeypatch, order_by
) -> None:
    # Since the stage 6 flip, ``momentum_12_1`` names a strategy: a whole entry
    # rule, not a filter set. Two rules do not intersect into one rule, so the
    # combination is refused before anything runs - and unlike the older
    # scorer-blend refusal it does not depend on --sort, because the objection
    # is to the rule, not to the ranking.
    def unexpected_scan(**kwargs: object) -> tuple[int, pd.DataFrame]:
        raise AssertionError("scan must not run once the combination is refused")

    monkeypatch.setattr(screen_workflow, "scan", unexpected_scan)

    with pytest.raises(UnscreenableStrategyError, match="Screen one at a time"):
        run_screen_workflow(
            _request(criteria=("momentum_12_1", "ema"), order_by=order_by)
        )


def test_apply_score_on_an_empty_frame_returns_an_empty_scored_frame() -> None:
    # A bar-derived spec must behave like every snapshot spec here: with no
    # rows there is no price history to resolve, so the missing `market` is
    # not yet an error.
    for spec in (get_scorer("momentum_12_1"), get_scorer("value")):
        scored = apply_score(pd.DataFrame(), spec)
        assert scored.empty
        assert OUTPUT_SCORE_COLUMN in scored.columns
