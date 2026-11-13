"""Unit tests for the George-Hwang 52-week high factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.factors import get_price_score, score_bars
from screener.scoring import OUTPUT_SCORE_COLUMN, apply_score, get_scorer
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)
_AS_OF = _INDEX[-1].date()
_MARKET = "india"


def _bars(close: np.ndarray | pd.Series, volume: float = 1_000_000.0) -> pd.DataFrame:
    series = pd.Series(close, index=_INDEX, dtype=float)
    openp = series.shift(1).fillna(series.iloc[0])
    return pd.DataFrame(
        {
            "open": openp,
            "high": pd.concat([openp, series], axis=1).max(axis=1) * 1.001,
            "low": pd.concat([openp, series], axis=1).min(axis=1) * 0.999,
            "close": series,
            "volume": pd.Series(volume, index=_INDEX, dtype=float),
        }
    )


def _growing(start: float, growth: float) -> pd.Series:
    return pd.Series(start * (1.0 + growth) ** np.arange(_N), index=_INDEX)


def test_strategy_registered() -> None:
    discover_plugins()
    spec = registry.get_optional("dist_from_52w_high")
    assert spec is not None
    assert spec.entry == "dist_from_52w_high > -100"
    assert spec.prepare_bars is not None
    assert spec.required_lookback() == 252


def test_recipe_matches_the_hand_written_formula() -> None:
    bars = _bars(_growing(100.0, 0.001))
    series = score_bars(get_price_score("dist_from_52w_high"), bars)
    close = bars["close"]
    expected = (close / close.rolling(252).max() - 1.0) * 100.0
    pd.testing.assert_series_equal(series, expected, check_names=False)


def test_score_is_causal_and_starts_after_a_full_year() -> None:
    close = _growing(100.0, 0.001)
    series = score_bars(get_price_score("dist_from_52w_high"), _bars(close))
    assert series.iloc[:251].isna().all()
    t = 300
    expected = (close.iloc[t] / close.iloc[t - 251 : t + 1].max() - 1.0) * 100.0
    assert series.iloc[t] == pytest.approx(expected)


def test_distance_from_high_is_ordered() -> None:
    """A name 30% below its high ranks under one at its high."""
    at_high = _growing(100.0, 0.0)
    dipped = pd.Series(
        np.r_[np.linspace(100.0, 140.0, 200), np.full(_N - 200, 98.0)],
        index=_INDEX,
    )
    scores = {
        tv: float(
            score_bars(get_price_score("dist_from_52w_high"), _bars(series)).iloc[-1]
        )
        for tv, series in {"HIGH": at_high, "DIP": dipped}.items()
    }
    assert scores["HIGH"] == pytest.approx(0.0)
    assert scores["DIP"] == pytest.approx((98.0 / 140.0 - 1.0) * 100.0)
    assert scores["HIGH"] > scores["DIP"]


def test_short_history_is_ineligible_not_ranked_last() -> None:
    short_index = _INDEX[:120]
    short = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        },
        index=short_index,
    )
    bars_by_tv = {
        "NSE:SEASONED": _bars(_growing(100.0, 0.001)),
        "NSE:NEWLY": short,
    }
    discover_plugins()
    prepare = registry.get("dist_from_52w_high").prepare_bars
    prepared = prepare(_prepare_ctx(bars_by_tv))
    assert np.isnan(prepared["NSE:NEWLY"]["rank_score"].iloc[-1])
    assert not np.isnan(prepared["NSE:SEASONED"]["rank_score"].iloc[-1])


def test_factor_ranks_nearest_high_over_liquidity() -> None:
    """NEAR is nearest its high but trades least; a volume ranker skips it."""
    near = _growing(50.0, 0.0015)
    far = pd.Series(
        np.r_[np.linspace(50.0, 80.0, 200), np.full(_N - 200, 40.0)],
        index=_INDEX,
    )
    data = {
        "NEAR": _bars(near, volume=5_000.0),
        "FAR": _bars(far, volume=5_000_000.0),
        "SPY": _bars(_growing(400.0, 0.0005), volume=1_000_000.0),
    }
    cfg = BacktestConfig(
        market="us",
        as_of=_INDEX[-1].date(),
        hold=10,
        top=1,
        strategy_name="dist_from_52w_high",
        entry_expr="dist_from_52w_high > -100",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=100_000.0,
        benchmark="SPY",
        tickers=("NEAR", "FAR"),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_rolling_backtest(
        cfg,
        StubPriceFetcher(data),
        start_date=_INDEX[260].date(),
        end_date=_INDEX[-1].date(),
    )
    assert {t.ticker for t in result.trades} == {"NEAR"}


def test_screen_and_backtest_report_the_same_score() -> None:
    bars_by_tv = {
        "NSE:ALPHA": _bars(_growing(100.0, 0.001)),
        "NSE:BETA": _bars(_growing(50.0, 0.0)),
    }
    discover_plugins()
    prepare = registry.get("dist_from_52w_high").prepare_bars
    backtest = {
        tv: float(frame["rank_score"].iloc[-1])
        for tv, frame in prepare(_prepare_ctx(bars_by_tv)).items()
    }
    rows = pd.DataFrame(
        [{"ticker": tv, "name": tv.split(":")[-1]} for tv in bars_by_tv]
    )
    fetcher = StubPriceFetcher(
        {f"{tv.split(':')[-1]}.NS": bars for tv, bars in bars_by_tv.items()}
    )
    screened = apply_score(
        rows,
        get_scorer("dist_from_52w_high"),
        market=_MARKET,
        as_of=_AS_OF,
        fetcher=fetcher,
    ).set_index("ticker")
    for tv in bars_by_tv:
        assert screened.loc[tv, "dist_from_52w_high"] == pytest.approx(backtest[tv])
        assert 0.0 <= screened.loc[tv, OUTPUT_SCORE_COLUMN] <= 100.0


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
