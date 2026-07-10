"""Unit tests for IC / quantile factor tearsheet pure math."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from screener.backtester.factor_tearsheet import (
    analyze_horizon,
    build_score_and_close_matrices,
    daily_spearman_ic,
    forward_returns,
    quantile_mean_returns,
    summarize_ic,
    top_quantile_turnover,
)


def _dates(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def test_forward_returns_no_lookahead_on_score_alignment() -> None:
    idx = _dates(5)
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=idx)
    fwd = forward_returns(close, horizon=1)
    # Return from t0 to t1 uses close[t1]/close[t0]-1; last row NaN.
    assert fwd.iloc[0, 0] == pytest.approx(0.10)
    assert math.isnan(fwd.iloc[-1, 0])


def test_positive_ic_for_predictive_factor() -> None:
    """Construct scores that perfectly rank next-day returns → IC ~ 1."""
    idx = _dates(40)
    # Three names; next-day return order is always A > B > C.
    # close paths: A grows fastest, C slowest.
    rng = np.arange(len(idx), dtype=float)
    close = pd.DataFrame(
        {
            "A": 100 * (1.02**rng),
            "B": 100 * (1.01**rng),
            "C": 100 * (1.005**rng),
        },
        index=idx,
    )
    # Score at t equals the *realized* 1-day forward return (oracle) — IC should
    # be near +1. This is only for unit-test synthetic data.
    fwd = forward_returns(close, 1)
    scores = fwd.copy()  # perfect foresight scores
    # Drop the last NaN row for a fair series.
    ic = daily_spearman_ic(scores.iloc[:-1], fwd.iloc[:-1])
    summary = summarize_ic(ic, horizon=1)
    assert summary.n_days > 10
    assert summary.ic_mean == pytest.approx(1.0, abs=1e-9)
    assert summary.pct_positive == pytest.approx(1.0)


def test_quantile_top_minus_bottom_positive() -> None:
    idx = _dates(50)
    # Scores rank names A > B > C > D > E every day; returns follow the same order.
    tickers = list("ABCDE")
    scores = pd.DataFrame(
        {t: float(i) for i, t in enumerate(tickers)},
        index=idx,
    )
    # Constant cross-section of scores every day.
    for t in tickers:
        scores[t] = float(ord(t) - ord("A"))
    fwd = pd.DataFrame(
        {t: float(ord(t) - ord("A")) * 0.01 for t in tickers},
        index=idx,
    )
    means, spread = quantile_mean_returns(scores, fwd, n_quantiles=5)
    assert spread > 0
    assert means[5] > means[1]


def test_top_quantile_turnover_full_churn() -> None:
    idx = _dates(4)
    # Alternate which name is the top score each day → turnover ≈ 1.
    scores = pd.DataFrame(
        {
            "A": [10.0, 1.0, 10.0, 1.0],
            "B": [1.0, 10.0, 1.0, 10.0],
        },
        index=idx,
    )
    # With only 2 names and 2 quantiles, top quantile is a singleton alternating.
    turnover = top_quantile_turnover(scores, n_quantiles=2)
    assert turnover == pytest.approx(1.0)


def test_analyze_horizon_bundle() -> None:
    idx = _dates(20)
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 120, len(idx)),
            "B": np.linspace(100, 110, len(idx)),
            "C": np.linspace(100, 105, len(idx)),
            "D": np.linspace(100, 102, len(idx)),
        },
        index=idx,
    )
    scores = pd.DataFrame(
        {
            "A": 4.0,
            "B": 3.0,
            "C": 2.0,
            "D": 1.0,
        },
        index=idx,
    )
    summary, qres, ic = analyze_horizon(scores, close, horizon=1, n_quantiles=4)
    assert summary.horizon == 1
    assert qres.n_quantiles == 4
    assert len(ic) == len(idx)


def test_build_score_and_close_matrices() -> None:
    idx = _dates(3)
    bars = {
        "AAA": pd.DataFrame(
            {
                "close": [1.0, 2.0, 3.0],
                "rank_score": [0.5, 0.6, 0.7],
            },
            index=idx,
        ),
        "BBB": pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx),
    }
    scores, closes = build_score_and_close_matrices(bars)
    assert list(scores.columns) == ["AAA"]
    assert set(closes.columns) == {"AAA", "BBB"}
