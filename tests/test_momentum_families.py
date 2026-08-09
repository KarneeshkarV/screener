"""Unit tests for the dual-momentum, trend-following and risk-managed families.

The published strategies these implement differ from each other mainly in *when*
they are willing to hold equities, so the tests concentrate on the gate: the
signal each paper defines, the state it puts the strategy in, and the causality
of every panel statistic that feeds it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from screener.backtester.pine import parse
from screener.risk_free import (
    INDIA_TBILL_RATE,
    US_TBILL_SYMBOL,
    annualized_rate,
    compounded_hurdle,
)
from screener.strategies.cross_section import (
    attach_column,
    close_panel,
    high_risk_state,
    positive_share,
    quantile_portfolio_returns,
    trailing_return,
)
from screener.strategies.plugins.dual_momentum import (
    ENTRY_DAA,
    ENTRY_GEM,
    ENTRY_MARKET,
    ENTRY_PAA,
    PAA_BREADTH_FLOOR,
    momentum_13612w,
    sma_momentum,
)
from screener.strategies.plugins.momentum_riskmanaged import (
    ENTRY_DYNAMIC,
    ENTRY_VOLMANAGED,
    crash_state,
    momentum_volatility_state,
)
from screener.strategies.plugins.time_series_momentum import (
    ENTRY_BLEND,
    ENTRY_TSMOM,
    trend_blend,
)
from screener.strategies.spec import discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 800
_INDEX = pd.bdate_range("2020-01-01", periods=_N)

NEW_STRATEGIES = (
    "momentum_6_6",
    "momentum_12_1_volmanaged",
    "momentum_12_1_dynamic",
    "dual_momentum_gem",
    "dual_momentum_market",
    "dual_momentum_paa",
    "dual_momentum_daa",
    "tsmom_12",
    "tsmom_blend",
)


def _geometric(start: float, daily_growth: float, n: int = _N) -> pd.Series:
    return pd.Series(start * (1.0 + daily_growth) ** np.arange(n), index=_INDEX[:n])


def _bars(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.Series(1_000_000.0, index=close.index),
        }
    )


# ── registration ───────────────────────────────────────────────────────


def test_every_new_strategy_is_registered() -> None:
    discover_plugins()
    for name in NEW_STRATEGIES:
        spec = registry.get_optional(name)
        assert spec is not None, name
        assert spec.prepare_bars is not None, name
        assert spec.required_lookback is not None, name
        assert spec.required_lookback() > 0, name


@pytest.mark.parametrize("name", [n for n in NEW_STRATEGIES if n != "momentum_6_6"])
def test_gated_strategies_exit_on_their_gate(name: str) -> None:
    # A risk gate that only blocked new entries would hold positions through the
    # decline it exists to avoid, and then sit out the recovery. Every gated
    # strategy must therefore also sell when its state turns against it.
    discover_plugins()
    spec = registry.get_optional(name)
    assert spec is not None
    assert spec.exit, name
    assert parse(spec.exit) is not None


@pytest.mark.parametrize(
    "expression",
    [
        ENTRY_VOLMANAGED,
        ENTRY_DYNAMIC,
        ENTRY_GEM,
        ENTRY_MARKET,
        ENTRY_PAA,
        ENTRY_DAA,
        ENTRY_TSMOM,
        ENTRY_BLEND,
    ],
)
def test_entry_expressions_parse(expression: str) -> None:
    # A typo in a gate would otherwise only surface as a runtime failure deep
    # inside a backtest.
    assert parse(expression) is not None


# ── cross-sectional panel helpers ──────────────────────────────────────


def test_close_panel_keeps_missing_cells_missing() -> None:
    early = _geometric(100.0, 0.001, n=10)
    late = _geometric(50.0, 0.001, n=10).iloc[5:]
    panel = close_panel({"EARLY": _bars(early), "LATE": _bars(late)})
    assert panel.shape == (10, 2)
    # A ticker that had not started trading must not be filled forward into a
    # price it never had.
    assert panel["LATE"].iloc[:5].isna().all()
    assert panel["LATE"].iloc[5:].notna().all()


def test_quantile_portfolio_returns_uses_prior_day_ranks() -> None:
    index = pd.bdate_range("2024-01-01", periods=4)
    closes = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0, 6.0], "B": [10.0, 9.0, 8.0, 16.0]}, index=index
    )
    # Scores rank A first for the first three dates and B first on the last one.
    scores = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0, 1.0]}, index=index
    )
    returns = quantile_portfolio_returns(closes, scores, quantile=0.5)
    # Day 0 has no prior ranks, so nothing is held.
    assert np.isnan(returns.iloc[0])
    # Days 1 and 2 hold A, whose returns are +10% and +9.09%.
    assert returns.iloc[1] == pytest.approx(0.1)
    assert returns.iloc[2] == pytest.approx(12 / 11 - 1)
    # Day 3 still holds A - B only became the top name *on* day 3, so its day-3
    # return cannot be earned. This is the causality guarantee.
    assert returns.iloc[3] == pytest.approx(6 / 12 - 1)


def test_high_risk_state_flags_only_the_top_tail() -> None:
    # Long calm stretch, then a climb into territory the trailing year has not
    # seen. Only the climb is a risk state.
    values = np.concatenate([np.full(400, 0.2), np.linspace(0.2, 0.6, 252)])
    volatility = pd.Series(values, index=pd.bdate_range("2023-01-02", periods=652))
    state = high_risk_state(volatility, percentile=0.8, window=252)
    assert not state.iloc[:251].any()  # warmup is never a risk state
    assert not state.iloc[300]  # calm, and ranked mid-pack against its own past
    assert state.iloc[-1]


def test_positive_share_counts_only_defined_scores() -> None:
    index = pd.bdate_range("2024-01-01", periods=2)
    scores = pd.DataFrame(
        {"A": [1.0, -1.0], "B": [-1.0, -1.0], "C": [np.nan, 1.0]}, index=index
    )
    share = positive_share(scores)
    assert share.iloc[0] == pytest.approx(0.5)  # 1 of 2 defined
    assert share.iloc[1] == pytest.approx(1 / 3)


def test_attach_column_broadcasts_and_defaults() -> None:
    close = _geometric(100.0, 0.001, n=6)
    values = pd.Series([True, True], index=[close.index[2], close.index[4]])
    out = attach_column({"A": _bars(close)}, values, "gate", False)
    gate = out["A"]["gate"]
    assert not gate.iloc[0]  # before the first observation, the default holds
    assert gate.iloc[2] and gate.iloc[3]  # forward-filled between observations
    assert gate.iloc[5]


def test_trailing_return_skips_the_reversal_window() -> None:
    closes = pd.DataFrame({"A": _geometric(100.0, 0.001, n=300)})
    twelve_one = trailing_return(closes, window=252, skip=21)
    t = 280
    expected = closes["A"].iloc[t - 21] / closes["A"].iloc[t - 252] - 1.0
    assert twelve_one["A"].iloc[t] == pytest.approx(expected)


# ── risk-managed momentum ──────────────────────────────────────────────


def test_crash_state_needs_both_bear_and_high_volatility() -> None:
    rng = np.random.default_rng(7)
    # A calm bull market: no crash state anywhere.
    calm = pd.Series(
        100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.004, _N)), index=_INDEX
    )
    assert not crash_state(calm).any()

    # A two-year decline whose late leg is far more volatile than its own past.
    quiet = rng.normal(-0.0008, 0.004, _N - 120)
    violent = rng.normal(-0.002, 0.035, 120)
    bear = pd.Series(
        100.0 * np.cumprod(1.0 + np.concatenate([quiet, violent])), index=_INDEX
    )
    state = crash_state(bear)
    assert state.iloc[-40:].any()
    # The state is causal: it cannot fire before the two-year window exists.
    assert not state.iloc[:504].any()


def test_momentum_volatility_state_is_boolean_and_causal() -> None:
    rng = np.random.default_rng(11)
    closes = pd.DataFrame(
        {
            name: 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, _N))
            for name in ("A", "B", "C", "D", "E")
        },
        index=_INDEX,
    )
    scores = closes.shift(21) / closes.shift(252) - 1.0
    state = momentum_volatility_state(closes, scores)
    assert state.dtype == bool
    # 252 momentum warmup + 126 volatility window + 252 percentile window.
    assert not state.iloc[:600].any()


def test_momentum_volatility_state_is_empty_without_a_panel() -> None:
    assert momentum_volatility_state(pd.DataFrame(), pd.DataFrame()).empty


# ── dual momentum ──────────────────────────────────────────────────────


def test_momentum_13612w_weights_the_four_horizons() -> None:
    close = _geometric(100.0, 0.001)
    blend = momentum_13612w(close)
    t = 400
    expected = sum(
        weight * (close.iloc[t] / close.iloc[t - months * 21] - 1.0)
        for months, weight in ((1, 12.0), (3, 4.0), (6, 2.0), (12, 1.0))
    )
    assert blend.iloc[t] == pytest.approx(expected)
    # Undefined until the longest leg has history.
    assert blend.iloc[:252].isna().all()


def test_momentum_13612w_is_negative_in_a_downtrend() -> None:
    assert momentum_13612w(_geometric(100.0, -0.001)).iloc[-1] < 0


def test_sma_momentum_matches_price_over_its_average() -> None:
    close = _geometric(100.0, 0.001)
    mom = sma_momentum(close)
    t = 400
    expected = close.iloc[t] / close.iloc[t - 251 : t + 1].mean() - 1.0
    assert mom.iloc[t] == pytest.approx(expected)
    assert mom.iloc[:251].isna().all()


def test_paa_floor_is_the_full_protection_boundary() -> None:
    # Keller & Keuning's a=2 reaches full cash when half the universe is
    # negative; the gate must be open strictly above that share.
    assert PAA_BREADTH_FLOOR == 0.5
    assert f"> {PAA_BREADTH_FLOOR}" in ENTRY_PAA


# ── time-series momentum ───────────────────────────────────────────────


def test_trend_blend_is_the_average_of_three_signs() -> None:
    close = _geometric(100.0, 0.001)
    blend = trend_blend(close)
    assert blend.iloc[-1] == pytest.approx(1.0)  # all three horizons up
    assert blend.iloc[:252].isna().all()  # the 12-month leg gates the blend
    assert set(np.unique(blend.dropna().round(6))) <= {-1.0, -1 / 3, 1 / 3, 1.0}


def test_trend_blend_disagreement_lands_between_the_poles() -> None:
    # Flat for a year, a steep two-month slide, then a partial one-month
    # recovery: the short horizon votes up, the two longer ones vote down.
    values = np.concatenate(
        [
            np.full(_N - 63, 100.0),
            np.linspace(100.0, 40.0, 42),
            np.linspace(40.0, 50.0, 21),
        ]
    )
    blend = trend_blend(pd.Series(values, index=_INDEX))
    assert blend.iloc[-1] == pytest.approx(-1 / 3)


# ── risk-free hurdle ───────────────────────────────────────────────────


def test_india_hurdle_uses_the_documented_constant() -> None:
    index = pd.DatetimeIndex(_INDEX[:5])
    rate = annualized_rate(
        "india", index, StubPriceFetcher({}), index[0].date(), index[-1].date()
    )
    assert (rate == INDIA_TBILL_RATE).all()


def test_us_hurdle_reads_the_bill_series_and_falls_back(monkeypatch) -> None:
    # FMP is the preferred US source, so pin it off: this test is about the
    # ^IRX path and its constant fallback, and must not reach the network.
    from screener import fmp

    monkeypatch.setattr(fmp, "resolve_api_key", lambda: None)
    index = pd.DatetimeIndex(_INDEX[:5])
    bills = _bars(pd.Series(5.0, index=index))  # ^IRX quotes percent
    rate = annualized_rate(
        "us",
        index,
        StubPriceFetcher({US_TBILL_SYMBOL: bills}),
        index[0].date(),
        index[-1].date(),
    )
    assert rate.iloc[0] == pytest.approx(0.05)

    missing = annualized_rate(
        "us", index, StubPriceFetcher({}), index[0].date(), index[-1].date()
    )
    assert missing.notna().all()


def test_compounded_hurdle_scales_with_the_window() -> None:
    rate = pd.Series([0.06, 0.06], index=_INDEX[:2])
    assert compounded_hurdle(rate, months=12).tolist() == pytest.approx([0.06, 0.06])
    assert compounded_hurdle(rate, months=6).tolist() == pytest.approx([0.03, 0.03])
