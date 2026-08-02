"""Screener metrics vs empyrical-reloaded 0.5.12 — reconciled comparisons.

Every convention difference that would cause a false failure is encoded as a
scalar transformation (documented in reference_adapters.py).  A red test here
means the screener's *math* is wrong, not that conventions happen to differ.

Reconciliation facts (all confirmed empirically before writing — see inline
comments for the derivation):

1. SHARPE  — screener uses population std (ddof=0); empyrical uses sample std
   (ddof=1).  std is in the DENOMINATOR, so screener > empyrical:
       screener_equity_curve_sharpe = empyrical_equity_curve_sharpe * sqrt(N / (N-1))

2. VOL_ANNUAL — same ddof difference.  std is in the NUMERATOR, so screener
   < empyrical:
       screener_vol = empyrical_vol * sqrt((N-1) / N)

3. CAGR — off-by-one FIXED.  An N-point equity curve spans N-1 return periods,
   so `_cagr` now annualizes over `(len(equity)-1)/252`, matching empyrical's
   `len(returns)/252`.  The previous `len(equity)/252` overstated the horizon by
   one bar and understated CAGR; screener and empyrical now agree to <1e-9.

4. SORTINO — FIXED.  Screener now uses the canonical target-downside-deviation
   RMS(min(excess, 0)) over ALL N observations, the same denominator empyrical
   uses.  screener and empyrical now agree to <1e-9.  (Previously the screener
   divided by std(downside_only, ddof=0), which inflated Sortino.)

5. ALPHA ANNUALIZATION — FIXED.  Screener now annualizes geometrically
   ((1+daily_intercept)^252 - 1), matching empyrical/quantstats.  The daily
   intercept still matches scipy.stats.linregress.  (Previously intercept*252,
   arithmetic, which overstated alpha by ignoring compounding.)

6. BETA and MAX-DRAWDOWN agree with empyrical to floating-point precision.
"""

from __future__ import annotations

import math

import empyrical
import numpy as np
import pandas as pd
import pytest
import scipy.stats

from screener.backtester.metrics import (
    _alpha_beta,
    _cagr,
    _max_drawdown,
    equity_curve_sharpe,
    _sortino,
    _vol_annual,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_returns() -> pd.Series:
    """252 synthetic daily returns (seed=42, ~5 bps/day, ~1 % daily vol)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=0.0005, scale=0.01, size=252))


@pytest.fixture(scope="module")
def sample_equity(sample_returns: pd.Series) -> pd.Series:
    """Equity curve corresponding to sample_returns.

    Construction: equity[0] = 100, equity[i] = equity[i-1] * (1 + r[i-1]).
    This yields len(equity) = len(returns) + 1 = 253, which is the *standard*
    construction used by the backtester.
    """
    values = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + sample_returns.values)])
    return pd.Series(values)


@pytest.fixture(scope="module")
def sample_bench(sample_returns: pd.Series) -> pd.Series:
    """252 synthetic benchmark returns (seed=7, correlated with sample_returns)."""
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(loc=0.0003, scale=0.008, size=252))


# ---------------------------------------------------------------------------
# 1. Sharpe — ddof reconciliation
# ---------------------------------------------------------------------------


def test_equity_curve_sharpe_exceeds_empyrical_by_sqrt_n_over_n_minus_1(
    sample_returns: pd.Series,
):
    """Screener Sharpe is larger than empyrical by exactly sqrt(N/(N-1)).

    Derivation:
        screener  uses std(ddof=0) = sqrt(sum(xi^2)/N)
        empyrical uses std(ddof=1) = sqrt(sum(xi^2)/(N-1))
        ratio of denominators (ddof=0 / ddof=1) = sqrt((N-1)/N)
        therefore screener_equity_curve_sharpe / empyrical_equity_curve_sharpe = ddof1 / ddof0 = sqrt(N/(N-1))
    """
    N = len(sample_returns)
    screener = equity_curve_sharpe(sample_returns)
    emp = empyrical.sharpe_ratio(sample_returns, period="daily")
    expected_ratio = math.sqrt(N / (N - 1))

    assert screener > emp, "screener Sharpe (ddof=0) must exceed empyrical (ddof=1)"
    assert abs(screener / emp - expected_ratio) < 1e-10, (
        f"Ratio {screener / emp} != sqrt(N/(N-1)) = {expected_ratio}"
    )


def test_equity_curve_sharpe_reconciliation_is_exact_across_sizes():
    """The sqrt(N/(N-1)) rescaling holds for different sample sizes."""
    rng = np.random.default_rng(0)
    for N in (50, 126, 252, 504):
        rets = pd.Series(rng.normal(0.0005, 0.01, N))
        screener = equity_curve_sharpe(rets)
        emp = empyrical.sharpe_ratio(rets, period="daily")
        if emp == 0:
            continue
        assert abs(screener / emp - math.sqrt(N / (N - 1))) < 1e-10, (
            f"N={N}: ratio {screener / emp} != {math.sqrt(N / (N - 1))}"
        )


# ---------------------------------------------------------------------------
# 2. Annual volatility — ddof reconciliation
# ---------------------------------------------------------------------------


def test_vol_annual_less_than_empyrical_by_sqrt_n_minus_1_over_n(
    sample_returns: pd.Series,
):
    """Screener annual vol is smaller than empyrical by sqrt((N-1)/N).

    Derivation: std is in the NUMERATOR.
        screener  uses std(ddof=0) * sqrt(252)
        empyrical uses std(ddof=1) * sqrt(252)
        ratio = ddof0 / ddof1 = sqrt((N-1)/N)  → screener < empyrical
    """
    N = len(sample_returns)
    screener = _vol_annual(sample_returns)
    emp = empyrical.annual_volatility(sample_returns)
    expected_ratio = math.sqrt((N - 1) / N)

    assert screener < emp, (
        "screener vol (ddof=0) must be smaller than empyrical (ddof=1)"
    )
    assert abs(screener / emp - expected_ratio) < 1e-10, (
        f"Ratio {screener / emp} != sqrt((N-1)/N) = {expected_ratio}"
    )


# ---------------------------------------------------------------------------
# 3. CAGR — off-by-one FIXED: screener now annualizes over N-1 return periods
# ---------------------------------------------------------------------------


def test_cagr_screener_uses_elapsed_periods_formula(sample_equity: pd.Series):
    """Screener _cagr(equity) annualizes over (len(equity)-1)/252 — pin exactly.

    An N-point equity curve spans N-1 daily returns, so the horizon is
    (N-1)/252 years. (Previously this used len(equity)/252, an off-by-one bug
    that understated CAGR; now fixed in metrics._cagr.)
    """
    start = float(sample_equity.iloc[0])
    end = float(sample_equity.iloc[-1])
    years = (len(sample_equity) - 1) / 252  # elapsed return periods
    expected = (end / start) ** (1.0 / years) - 1.0

    screener = _cagr(sample_equity)
    assert abs(screener - expected) < 1e-12, (
        f"screener CAGR {screener} != elapsed-periods formula {expected}"
    )


def test_cagr_matches_empyrical_after_off_by_one_fix(
    sample_equity: pd.Series, sample_returns: pd.Series
):
    """Independent oracle: screener _cagr now AGREES with empyrical.cagr.

    With equity of len(returns)+1 bars (normal construction) both annualize
    over N = len(returns) return periods:
        screener years  = (len(equity)-1) / 252 = N / 252
        empyrical years =  len(returns)   / 252 = N / 252
    The previously-documented one-bar divergence is gone after the fix.
    """
    assert len(sample_equity) == len(sample_returns) + 1, (
        "Fixture invariant: equity must have exactly one more bar than returns"
    )
    screener = _cagr(sample_equity)
    emp = empyrical.cagr(sample_returns)

    # The off-by-one is fixed: screener and empyrical now match the independent
    # oracle (end/start = prod(1+r) for a properly constructed equity curve).
    assert abs(screener - emp) < 1e-9, (
        f"screener CAGR {screener} != empyrical {emp} (diff {abs(screener - emp):.2e})"
    )

    # And screener matches the elapsed-periods hand formula.
    start = float(sample_equity.iloc[0])
    end = float(sample_equity.iloc[-1])
    years = (len(sample_equity) - 1) / 252
    screener_formula = (end / start) ** (1.0 / years) - 1.0
    assert abs(screener - screener_formula) < 1e-12


# ---------------------------------------------------------------------------
# 4. Sortino matches empyrical (canonical target-downside-deviation)
# ---------------------------------------------------------------------------


def test_sortino_matches_empyrical(sample_returns: pd.Series):
    """_sortino now AGREES with empyrical.sortino_ratio to <1e-9.

    Both use the canonical target-downside-deviation denominator:
        sqrt(mean(min(r, 0)^2)) — RMS over ALL N observations (target = 0).

    (Previously the screener divided by std(r[r < 0], ddof=0) — the std of the
    negative subset only — which inflated Sortino.  That non-standard choice is
    fixed; the screener and empyrical now match the same oracle.)
    """
    screener = _sortino(sample_returns)
    emp = empyrical.sortino_ratio(sample_returns, period="daily")

    assert abs(screener - emp) < 1e-9, (
        f"screener Sortino {screener} != empyrical {emp} "
        f"(diff {abs(screener - emp):.2e})"
    )

    # Independent hand oracle: RMS of min(r, 0) over all N (target = 0).
    rms_denominator = float(np.sqrt(np.mean(np.minimum(sample_returns.values, 0) ** 2)))
    hand_sortino = float(sample_returns.mean() / rms_denominator * math.sqrt(252))
    assert abs(screener - hand_sortino) < 1e-10, (
        f"screener Sortino {screener} != RMS-downside formula {hand_sortino}"
    )


# ---------------------------------------------------------------------------
# 5. Alpha / Beta vs scipy + empyrical
# ---------------------------------------------------------------------------


def test_beta_matches_empyrical_and_scipy(
    sample_returns: pd.Series, sample_bench: pd.Series
):
    """Beta (OLS slope) agrees with empyrical.beta and scipy.stats.linregress.

    OLS slope is unique regardless of annualization convention, so all three
    should agree to floating-point precision.
    """
    _, screener_beta = _alpha_beta(sample_returns, sample_bench)
    emp_beta = empyrical.beta(sample_returns, sample_bench)
    scipy_result = scipy.stats.linregress(sample_bench.values, sample_returns.values)

    assert abs(screener_beta - emp_beta) < 1e-10, (
        f"Beta vs empyrical: {screener_beta} != {emp_beta}"
    )
    assert abs(screener_beta - scipy_result.slope) < 1e-10, (
        f"Beta vs scipy slope: {screener_beta} != {scipy_result.slope}"
    )


def test_alpha_screener_is_geometric_annualization(
    sample_returns: pd.Series, sample_bench: pd.Series
):
    """Screener alpha = (1 + OLS daily intercept)^252 - 1 (geometric annualization).

    This matches the standard empyrical/quantstats convention.  (Previously the
    screener used arithmetic intercept*252, which overstated alpha by ignoring
    compounding.)  The daily intercept itself must still match scipy.stats.linregress.
    """
    screener_alpha, _ = _alpha_beta(sample_returns, sample_bench)
    emp_alpha = empyrical.alpha(sample_returns, sample_bench, period="daily")
    scipy_result = scipy.stats.linregress(sample_bench.values, sample_returns.values)

    daily_intercept = scipy_result.intercept

    # Screener alpha = geometric annualization of the daily intercept.
    geometric_alpha = (1.0 + daily_intercept) ** 252 - 1.0
    assert abs(screener_alpha - geometric_alpha) < 1e-10, (
        f"screener alpha {screener_alpha} != geometric formula {geometric_alpha}"
    )

    # And it now agrees with empyrical's geometric alpha to oracle precision.
    assert abs(screener_alpha - emp_alpha) < 1e-6, (
        f"screener alpha {screener_alpha} != empyrical {emp_alpha}"
    )


# ---------------------------------------------------------------------------
# 6. Max drawdown agrees with empyrical exactly
# ---------------------------------------------------------------------------


def test_max_drawdown_matches_empyrical(
    sample_equity: pd.Series, sample_returns: pd.Series
):
    """_max_drawdown(equity) matches empyrical.max_drawdown(returns) to 1e-12.

    empyrical computes drawdown on the cumulative return series, which is
    equivalent to the equity curve ratio.  No convention differences.
    """
    screener = _max_drawdown(sample_equity)
    emp = empyrical.max_drawdown(sample_returns)

    assert abs(screener - emp) < 1e-10, (
        f"max drawdown mismatch: screener {screener} vs empyrical {emp}"
    )
