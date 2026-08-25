"""The screen and the backtest must agree on a shared price-only score.

``momentum_12_1`` used to name two different formulas: a 0-100 cross-sectional
percentile of TradingView's ``Perf.Y``/``Perf.1M`` in the screen, and a raw
``close[t-21]/close[t-252] - 1`` in the backtest. Backtesting the name told you
nothing about what the screen would pick. Both now read one recipe in
``screener.factors``; these tests pin that down.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from screener.factors import get_price_score, score_bars
from screener.scoring import OUTPUT_SCORE_COLUMN, apply_score, get_scorer
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 400
_INDEX = pd.bdate_range("2022-01-03", periods=_N)
_AS_OF = _INDEX[-1].date()
_MARKET = "india"


def _bars(start: float, daily_growth: float, periods: int = _N) -> pd.DataFrame:
    """A deterministic geometric trend, so 12-1 momentum is exactly known."""
    index = _INDEX[-periods:]
    close = pd.Series(
        start * (1.0 + daily_growth) ** np.arange(periods, dtype=float),
        index=index,
    )
    openp = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, close], axis=1).max(axis=1) * 1.001,
            "low": pd.concat([openp, close], axis=1).min(axis=1) * 0.999,
            "close": close,
            "volume": pd.Series(1_000_000.0, index=index),
        }
    )


def _backtest_rank_score(bars_by_tv: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Last-bar ``rank_score`` produced by the backtest ``prepare_bars`` hook."""
    discover_plugins()
    spec = registry.get("momentum_12_1")
    prepare = spec.prepare_bars
    assert prepare is not None
    ctx = _prepare_ctx(bars_by_tv)
    prepared = prepare(ctx)
    return {tv: float(frame["rank_score"].iloc[-1]) for tv, frame in prepared.items()}


def _prepare_ctx(bars_by_tv: dict[str, pd.DataFrame]):
    from screener.strategies.spec import PrepareCtx

    return PrepareCtx(
        market=_MARKET,
        benchmark="^NSEI",
        bars_by_tv=bars_by_tv,
        price_panel={},
        tv_symbols=list(bars_by_tv),
        start=_INDEX[0].date(),
        end=_AS_OF,
        fetcher=StubPriceFetcher({}),
        warnings=[],
    )


def _screen_setup_score(
    bars_by_tv: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run the screen adapter over the same bars via a stub price fetcher."""
    rows = pd.DataFrame(
        [{"ticker": tv, "name": tv.split(":")[-1]} for tv in bars_by_tv]
    )
    fetcher = StubPriceFetcher(
        {f"{tv.split(':')[-1]}.NS": bars for tv, bars in bars_by_tv.items()}
    )
    return apply_score(
        rows,
        get_scorer("momentum_12_1"),
        market=_MARKET,
        as_of=_AS_OF,
        fetcher=fetcher,
    )


def test_screen_and_backtest_momentum_12_1_are_the_same_number() -> None:
    """Headline: one formula, two adapters. The raw value is identical.

    ``setup_score`` is the within-scan 0-100 percentile of that raw value, so
    downstream ``min_score`` thresholds keep working. The number that must
    match the backtest's ``rank_score`` is the ``mom_12_1`` aux column.
    A non-positive 12-1 name is not a candidate on either path: the screen
    drops it before the percentile, the backtest keeps the raw value but
    its ``ENTRY_PURE`` expression refuses the entry.
    """
    bars_by_tv = {
        "NSE:ALPHA": _bars(100.0, 0.0020),
        "NSE:BETA": _bars(50.0, 0.0005),
        "NSE:GAMMA": _bars(75.0, -0.0004),
    }

    backtest = _backtest_rank_score(bars_by_tv)
    screened = _screen_setup_score(bars_by_tv).set_index("ticker")

    assert "NSE:GAMMA" not in screened.index
    assert backtest["NSE:GAMMA"] <= 0
    assert sorted(screened.index) == ["NSE:ALPHA", "NSE:BETA"]
    for tv in screened.index:
        assert screened.loc[tv, "mom_12_1"] == backtest[tv], tv
        assert 0.0 <= screened.loc[tv, OUTPUT_SCORE_COLUMN] <= 100.0

    # Rank order of setup_score matches the backtester's ranking of survivors.
    assert list(screened[OUTPUT_SCORE_COLUMN].sort_values(ascending=False).index) == [
        tv
        for tv, _ in sorted(
            ((name, backtest[name]) for name in screened.index),
            key=lambda kv: kv[1],
            reverse=True,
        )
    ]


def test_shared_recipe_matches_the_hand_written_formula() -> None:
    bars = _bars(100.0, 0.001)
    series = score_bars(get_price_score("momentum_12_1"), bars)
    close = bars["close"]
    expected = close.shift(21) / close.shift(252) - 1.0
    pd.testing.assert_series_equal(series, expected, check_names=False)


def test_short_history_is_excluded_not_bottom_ranked() -> None:
    """NaN means ineligible. The legacy ``fillna(0)`` made it merely last."""
    bars_by_tv = {
        "NSE:LONG": _bars(100.0, 0.0015),
        "NSE:NEWLY": _bars(100.0, 0.0015, periods=60),
    }

    screened = _screen_setup_score(bars_by_tv)

    assert screened["ticker"].tolist() == ["NSE:LONG"]
    assert screened[OUTPUT_SCORE_COLUMN].notna().all()

    # The backtest path agrees: no score, so no slot.
    backtest = _backtest_rank_score(bars_by_tv)
    assert np.isnan(backtest["NSE:NEWLY"])
    assert not np.isnan(backtest["NSE:LONG"])


def test_unified_recipe_never_fills_missing_history_with_zero() -> None:
    bars = _bars(100.0, 0.001, periods=100)
    series = score_bars(get_price_score("momentum_12_1"), bars)
    assert series.isna().all()


def test_bar_scorer_needs_a_market() -> None:
    rows = pd.DataFrame([{"ticker": "NSE:ALPHA", "name": "ALPHA"}])
    with pytest.raises(ValueError, match="bar-derived"):
        apply_score(rows, get_scorer("momentum_12_1"))


def test_bar_scorer_requests_no_tradingview_columns() -> None:
    spec = get_scorer("momentum_12_1")
    assert spec.bar_score is not None
    assert spec.columns == ()


def test_bar_scorer_does_not_blend_with_snapshot_recipes() -> None:
    from screener.scoring import IncompatibleScorerBlendError, resolve_scorer

    with pytest.raises(IncompatibleScorerBlendError, match="cannot blend bar-derived"):
        resolve_scorer(["momentum_12_1", "ema"])


def test_scan_scores_only_the_rows_the_filters_returned(monkeypatch) -> None:
    """Bars are fetched for the scan's rows, never for the whole market.

    ``build_price_fetcher`` is patched where it is actually looked up
    (``screener.backtester.data``, imported inside ``bar_scores_for_tickers``)
    with ``raising=True``, so this test cannot reach the network even if the
    explicit ``fetcher=`` injection below is ever dropped, and a rename of that
    seam breaks the test loudly instead of silently re-enabling it.
    """
    import screener.backtester.data as backtester_data
    import screener.scoring.bar_scores as bar_scores

    bars_by_tv = {
        "NSE:ALPHA": _bars(100.0, 0.002),
        "NSE:BETA": _bars(80.0, 0.001),
    }
    fetcher = StubPriceFetcher(
        {f"{tv.split(':')[-1]}.NS": bars for tv, bars in bars_by_tv.items()}
    )
    requested: list[list[str]] = []
    original = fetcher.fetch

    def _record(tickers, start, end):
        symbols = list(tickers)
        requested.append(symbols)
        return original(symbols, start, end)

    monkeypatch.setattr(fetcher, "fetch", _record)
    monkeypatch.setattr(
        backtester_data,
        "build_price_fetcher",
        lambda **_kwargs: fetcher,
    )

    rows = pd.DataFrame([{"ticker": tv, "name": tv} for tv in bars_by_tv])
    scored = bar_scores.apply_bar_score(
        rows,
        get_price_score("momentum_12_1"),
        market=_MARKET,
        output_column=OUTPUT_SCORE_COLUMN,
        as_of=_AS_OF,
        fetcher=fetcher,
    )

    assert requested == [["ALPHA.NS", "BETA.NS"]]
    assert len(scored) == 2


def test_bar_score_fetch_window_covers_the_required_lookback() -> None:
    spec = get_price_score("momentum_12_1")
    start = bar_scores_start(spec.required_lookback)
    sessions = len(pd.bdate_range(start, _AS_OF))
    assert sessions > spec.required_lookback


def bar_scores_start(lookback: int) -> date:
    from screener.scoring.bar_scores import _fetch_start

    return _fetch_start(_AS_OF, lookback)
