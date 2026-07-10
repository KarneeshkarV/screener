"""Unit tests for the generic multi-factor combo strategy factory."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from screener.backtester.models import BacktestConfig
from screener.strategies.combo import (
    combine_rank_scores,
    cross_sectional_zscore,
    is_combo_strategy,
    parse_combo_spec,
    resolve_combo_spec,
    validate_combo_components,
)
from screener.strategies.expressions import resolve_strategy
from screener.strategies.spec import PrepareCtx, discover_plugins


def test_parse_combo_spec_valid() -> None:
    parts = parse_combo_spec("combo:momentum_12_1=0.6,low_volatility=0.4")
    assert parts == [("momentum_12_1", 0.6), ("low_volatility", 0.4)]


def test_parse_combo_spec_rejects_bad_weights() -> None:
    # 1e999 overflows to inf → rejected as non-finite.
    with pytest.raises(ValueError, match="finite"):
        parse_combo_spec("combo:momentum_12_1=1e999")
    with pytest.raises(ValueError, match="at least one"):
        parse_combo_spec("combo:")
    with pytest.raises(ValueError, match="invalid combo component"):
        parse_combo_spec("combo:momentum_12_1")


def test_validate_requires_known_factor_components() -> None:
    discover_plugins()
    with pytest.raises(ValueError, match="unknown combo component"):
        validate_combo_components([("does_not_exist", 1.0)])
    # Expression-only strategies without prepare_bars cannot be combined.
    with pytest.raises(ValueError, match="prepare_bars"):
        validate_combo_components([("breakout", 1.0)])


def test_resolve_strategy_accepts_combo_prefix() -> None:
    named = resolve_strategy("combo:momentum_12_1=0.5,low_volatility=0.5")
    assert "mom_12_1" in named.entry
    assert "vol_252" in named.entry
    assert named.exit is None


def test_is_combo_strategy() -> None:
    assert is_combo_strategy("combo:momentum_12_1=1")
    assert not is_combo_strategy("momentum_12_1")
    assert not is_combo_strategy(None)


def test_cross_sectional_zscore_and_weighted_blend() -> None:
    idx = pd.bdate_range("2024-01-02", periods=1)
    # Factor 1 scores: A=3, B=1 → z = +1, -1 (pop std 1, mean 2)
    f1 = pd.DataFrame({"A": [3.0], "B": [1.0]}, index=idx)
    # Factor 2 scores: A=1, B=3 → z = -1, +1
    f2 = pd.DataFrame({"A": [1.0], "B": [3.0]}, index=idx)
    z1 = cross_sectional_zscore(f1)
    assert z1.loc[idx[0], "A"] == pytest.approx(1.0)
    assert z1.loc[idx[0], "B"] == pytest.approx(-1.0)

    # 0.75 * f1 + 0.25 * f2 → A: 0.75*1 + 0.25*(-1) = 0.5
    #                         B: 0.75*(-1) + 0.25*1 = -0.5
    blended = combine_rank_scores([(f1, 0.75), (f2, 0.25)])
    assert blended.loc[idx[0], "A"] == pytest.approx(0.5)
    assert blended.loc[idx[0], "B"] == pytest.approx(-0.5)
    # Ranking by blended score: A > B
    assert blended.loc[idx[0]].idxmax() == "A"


def test_combo_prepare_writes_weighted_rank_score() -> None:
    discover_plugins()
    idx = pd.bdate_range("2022-01-03", periods=320)

    # Distinct growth/noise so momentum and low-vol disagree.
    def frame(growth: float, noise: float) -> pd.DataFrame:
        drift = 50.0 * (1.0 + growth) ** np.arange(len(idx))
        wiggle = noise * 50.0 * np.sin(np.arange(len(idx)))
        close = pd.Series(drift + wiggle, index=idx)
        openp = close.shift(1).fillna(close.iloc[0])
        return pd.DataFrame(
            {
                "open": openp,
                "high": pd.concat([openp, close], axis=1).max(axis=1) + 0.5,
                "low": pd.concat([openp, close], axis=1).min(axis=1) - 0.5,
                "close": close,
                "volume": pd.Series(100_000.0, index=idx),
            }
        )

    bars_by_tv = {
        "HIMOM": frame(0.0020, 0.03),
        "LOVOL": frame(0.0006, 0.002),
        "MID": frame(0.0012, 0.01),
    }
    name = "combo:momentum_12_1=1.0,low_volatility=0.0"
    spec = resolve_combo_spec(name)
    assert spec.prepare_bars is not None
    assert spec.required_lookback is not None
    assert spec.required_lookback() >= 252

    ctx = PrepareCtx(
        cfg=BacktestConfig(
            market="us",
            as_of=idx[-1].date(),
            hold=5,
            top=1,
            strategy_name=name,
            entry_expr=spec.entry or "close > 0",
            exit_expr=None,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
            initial_capital=100_000.0,
            benchmark="SPY",
            tickers=tuple(bars_by_tv),
            min_price=None,
            min_avg_dollar_volume=None,
        ),
        bars_by_tv=bars_by_tv,
        price_panel={},
        tv_symbols=list(bars_by_tv),
        start=idx[0].date(),
        end=idx[-1].date(),
        fetcher=lambda *a, **k: {},
        warnings=[],
    )
    prepared = spec.prepare_bars(ctx)
    # Pure-momentum weights → highest rank_score should be HIMOM on late bars.
    late = idx[-1]
    scores = {
        tv: float(prepared[tv].loc[late, "rank_score"])
        for tv in bars_by_tv
        if math.isfinite(float(prepared[tv].loc[late, "rank_score"]))
    }
    assert scores
    assert max(scores, key=scores.get) == "HIMOM"  # type: ignore[arg-type]
