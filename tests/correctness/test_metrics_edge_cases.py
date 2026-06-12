"""Edge-case and guard-rail tests for screener metrics.

These tests exercise the sentinel paths documented in the metrics.py guards:
empty inputs, single-element inputs, all-zero returns, all-positive returns,
constant equity, and the PSR/DSR short-series guards.

The guards are READ from the source first (see inline comments).  A failure
means a guard was removed or changed, not that a convention drifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.backtester.metrics import (
    _alpha_beta,
    _cagr,
    _calmar,
    _daily_returns,
    _dsr,
    _max_drawdown,
    _psr,
    _sharpe,
    _sortino,
    _vol_annual,
)

# ---------------------------------------------------------------------------
# Empty inputs — every public function must return a safe sentinel
# ---------------------------------------------------------------------------


def test_sharpe_empty_returns_zero():
    """Guard: `if daily.empty or daily.std(ddof=0) == 0: return 0.0`"""
    assert _sharpe(pd.Series(dtype=float)) == 0.0


def test_sortino_empty_returns_zero():
    """Guard: `if daily.empty: return 0.0`"""
    assert _sortino(pd.Series(dtype=float)) == 0.0


def test_vol_annual_empty_returns_zero():
    """Guard: `if daily.empty: return 0.0`"""
    assert _vol_annual(pd.Series(dtype=float)) == 0.0


def test_cagr_empty_returns_zero():
    """Guard: `if equity.empty or len(equity) < 2: return 0.0`"""
    assert _cagr(pd.Series(dtype=float)) == 0.0


def test_max_drawdown_empty_returns_zero():
    """Guard: `if equity.empty: return 0.0`"""
    assert _max_drawdown(pd.Series(dtype=float)) == 0.0


def test_calmar_empty_returns_zero():
    """Guard: delegates to _cagr / _max_drawdown which both guard for empty."""
    assert _calmar(pd.Series(dtype=float)) == 0.0


def test_daily_returns_empty_is_empty():
    """Guard: `if equity.empty or len(equity) < 2: return pd.Series(dtype=float)`"""
    result = _daily_returns(pd.Series(dtype=float))
    assert isinstance(result, pd.Series)
    assert result.empty


def test_alpha_beta_empty_returns_zeros():
    """Guard: `if daily.empty or bench_daily.empty: return 0.0, 0.0`"""
    empty = pd.Series(dtype=float)
    alpha, beta = _alpha_beta(empty, empty)
    assert alpha == 0.0
    assert beta == 0.0


def test_psr_empty_returns_zero():
    """Guard: `if daily.empty or len(daily) < 30: return 0.0`"""
    assert _psr(pd.Series(dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# Single-element inputs
# ---------------------------------------------------------------------------


def test_cagr_single_element_equity_returns_zero():
    """Single-bar equity → len(equity) < 2 guard → 0.0."""
    assert _cagr(pd.Series([100.0])) == 0.0


def test_calmar_single_element_equity_returns_zero():
    """Single-bar equity → _cagr returns 0 → calmar returns 0.0."""
    assert _calmar(pd.Series([100.0])) == 0.0


def test_daily_returns_single_element_is_empty():
    """Single-bar equity → len < 2 guard → empty Series."""
    result = _daily_returns(pd.Series([100.0]))
    assert result.empty


def test_alpha_beta_single_aligned_row_returns_zeros():
    """After dropna alignment, len(aligned) < 2 → return (0.0, 0.0).

    Each Series has one element on the same index so alignment yields one row,
    which is below the polyfit minimum of 2.
    """
    alpha, beta = _alpha_beta(
        pd.Series([0.01], index=[0]),
        pd.Series([0.02], index=[0]),
    )
    assert alpha == 0.0
    assert beta == 0.0


# ---------------------------------------------------------------------------
# All-zero returns
# ---------------------------------------------------------------------------


def test_sharpe_all_zero_returns_zero():
    """std(ddof=0) == 0 for zero returns → guard fires → 0.0."""
    zeros = pd.Series([0.0] * 10)
    assert _sharpe(zeros) == 0.0


def test_sortino_all_zero_no_downside_returns_zero():
    """excess=zeros → no negative elements → downside empty → guard fires → 0.0."""
    zeros = pd.Series([0.0] * 10)
    assert _sortino(zeros) == 0.0


def test_vol_annual_all_zero_is_zero():
    """std(ddof=0) of zeros = 0 → vol_annual = 0."""
    zeros = pd.Series([0.0] * 10)
    assert _vol_annual(zeros) == 0.0


def test_psr_all_zero_short_returns_zero():
    """All-zero series shorter than 30 hits the length guard first."""
    assert _psr(pd.Series([0.0] * 25)) == 0.0


def test_psr_all_zero_long():
    """All-zero series of length ≥ 30: std == 0 → _sharpe returns 0 → sr_per = 0.

    The PSR formula with sr_per=0 and sr_bench_per=0: z=0 → phi(0)=0.5.
    (No guard fires for std=0 inside _psr; it proceeds to phi(0).)
    """
    result = _psr(pd.Series([0.0] * 30))
    # sr_per = 0 / sqrt(252) = 0; denom = sqrt(1 - 0 + 0) = 1; z = 0
    assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# All-positive returns (no downside)
# ---------------------------------------------------------------------------


def test_sortino_all_positive_returns_zero():
    """No negative excess returns → downside empty → guard fires → 0.0.

    Source guard: `if downside.empty or downside.std(ddof=0) == 0: return 0.0`
    """
    all_pos = pd.Series([0.005, 0.01, 0.015, 0.02, 0.025])
    assert _sortino(all_pos) == 0.0


def test_sortino_single_negative_computes_correctly():
    """One negative value gives a downside std of 0 (single element, ddof=0) → 0.0."""
    one_neg = pd.Series([0.01, 0.02, -0.005, 0.015])
    # downside = [-0.005]; std(ddof=0) of a single element = 0 → guard fires
    assert _sortino(one_neg) == 0.0


def test_sortino_two_equal_negatives_zero_std_returns_zero():
    """Two identical negative returns → std(ddof=0) = 0 → 0.0."""
    returns = pd.Series([0.01, -0.005, -0.005, 0.02])
    # downside std(ddof=0) = 0 → guard fires
    assert _sortino(returns) == 0.0


# ---------------------------------------------------------------------------
# Constant equity
# ---------------------------------------------------------------------------


def test_max_drawdown_constant_equity_is_zero():
    """Constant equity → peak always equals equity → drawdown is always 0."""
    equity = pd.Series([100.0] * 20)
    assert _max_drawdown(equity) == 0.0


def test_cagr_constant_equity_is_zero():
    """start == end → (end/start)^(1/years) - 1 = 1^(1/years) - 1 = 0."""
    equity = pd.Series([100.0] * 10)
    assert _cagr(equity) == 0.0


def test_calmar_constant_equity_is_zero():
    """Constant equity → mdd = 0 → calmar guard (mdd >= 0) fires → 0.0."""
    equity = pd.Series([100.0] * 10)
    assert _calmar(equity) == 0.0


def test_daily_returns_constant_equity_all_zeros():
    """pct_change on constant equity = 0 for every bar."""
    equity = pd.Series([100.0] * 5)
    result = _daily_returns(equity)
    assert len(result) == 4
    assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# PSR short-series boundary
# ---------------------------------------------------------------------------


def test_psr_returns_zero_when_len_less_than_30():
    """Strict guard: len(daily) < 30 → return 0.0."""
    rng = np.random.default_rng(0)
    for n in (0, 1, 10, 25, 29):
        series = pd.Series(rng.normal(0.001, 0.01, n) if n > 0 else [], dtype=float)
        assert _psr(series) == 0.0, f"PSR(len={n}) should be 0.0"


def test_psr_nonzero_when_len_equals_30():
    """len == 30 passes the guard (strictly, guard is len < 30)."""
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0.001, 0.01, 30))
    result = _psr(series)
    assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# DSR n_trials <= 1 path
# ---------------------------------------------------------------------------


def test_dsr_n_trials_1_equals_psr_zero_benchmark():
    """n_trials <= 1 → `return _psr(daily, 0.0)` directly."""
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0.001, 0.01, 252))
    assert _dsr(series, n_trials=1) == _psr(series, 0.0)


def test_dsr_n_trials_0_equals_psr_zero_benchmark():
    """n_trials=0 also satisfies n_trials <= 1 → same path."""
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0.001, 0.01, 252))
    assert _dsr(series, n_trials=0) == _psr(series, 0.0)


def test_dsr_n_trials_greater_than_1_uses_higher_benchmark():
    """n_trials > 1 raises the benchmark SR → DSR < PSR(0.0)."""
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(0.001, 0.01, 252))
    psr = _psr(series, 0.0)
    dsr = _dsr(series, n_trials=10)
    # Higher benchmark → probability of exceeding it is lower
    assert dsr < psr


# ---------------------------------------------------------------------------
# Negative start equity guard (CAGR)
# ---------------------------------------------------------------------------


def test_cagr_nonpositive_start_returns_zero():
    """Guard: `if start <= 0: return 0.0`"""
    equity_zero_start = pd.Series([0.0, 100.0, 110.0])
    assert _cagr(equity_zero_start) == 0.0


# ---------------------------------------------------------------------------
# Alpha/beta degenerate bench
# ---------------------------------------------------------------------------


def test_alpha_beta_constant_benchmark_returns_zeros():
    """x.std() == 0 (constant bench) → return (0.0, 0.0) to avoid polyfit singularity."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03])
    const_bench = pd.Series([0.005] * 5)
    alpha, beta = _alpha_beta(returns, const_bench)
    assert alpha == 0.0
    assert beta == 0.0


def test_alpha_beta_non_overlapping_indices_returns_zeros():
    """Non-overlapping indices → aligned has 0 rows → len(aligned) < 2 → (0.0, 0.0)."""
    returns = pd.Series([0.01, 0.02], index=[0, 1])
    bench = pd.Series([0.01, 0.02], index=[2, 3])
    alpha, beta = _alpha_beta(returns, bench)
    assert alpha == 0.0
    assert beta == 0.0
