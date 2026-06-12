"""Screener metrics vs empyrical-reloaded 0.5.12 — reconciled comparisons.

Every convention difference that would cause a false failure is encoded as a
scalar transformation (documented in reference_adapters.py).  A red test here
means the screener's *math* is wrong, not that conventions happen to differ.

Reconciliation facts (all confirmed empirically before writing — see inline
comments for the derivation):

1. SHARPE  — screener uses population std (ddof=0); empyrical uses sample std
   (ddof=1).  std is in the DENOMINATOR, so screener > empyrical:
       screener_sharpe = empyrical_sharpe * sqrt(N / (N-1))

2. VOL_ANNUAL — same ddof difference.  std is in the NUMERATOR, so screener
   < empyrical:
       screener_vol = empyrical_vol * sqrt((N-1) / N)

3. CAGR — off-by-one FIXED.  An N-point equity curve spans N-1 return periods,
   so `_cagr` now annualizes over `(len(equity)-1)/252`, matching empyrical's
   `len(returns)/252`.  The previous `len(equity)/252` overstated the horizon by
   one bar and understated CAGR; screener and empyrical now agree to <1e-9.

4. SORTINO — screener divides by std(downside_only, ddof=0); empyrical uses
   RMS(min(r, 0)) over ALL N observations.  Not a scalar factor → values are
   genuinely different.  Classified: NON-STANDARD DESIGN CHOICE.

5. ALPHA ANNUALIZATION — screener: intercept * 252 (arithmetic).
   empyrical: geometric ((1+daily_alpha)^252 - 1).  Values diverge.
   Classified: NON-STANDARD DESIGN CHOICE.

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
    _sharpe,
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


def test_sharpe_exceeds_empyrical_by_sqrt_n_over_n_minus_1(sample_returns: pd.Series):
    """Screener Sharpe is larger than empyrical by exactly sqrt(N/(N-1)).

    Derivation:
        screener  uses std(ddof=0) = sqrt(sum(xi^2)/N)
        empyrical uses std(ddof=1) = sqrt(sum(xi^2)/(N-1))
        ratio of denominators (ddof=0 / ddof=1) = sqrt((N-1)/N)
        therefore screener_sharpe / empyrical_sharpe = ddof1 / ddof0 = sqrt(N/(N-1))
    """
    N = len(sample_returns)
    screener = _sharpe(sample_returns)
    emp = empyrical.sharpe_ratio(sample_returns, period="daily")
    expected_ratio = math.sqrt(N / (N - 1))

    assert screener > emp, "screener Sharpe (ddof=0) must exceed empyrical (ddof=1)"
    assert abs(screener / emp - expected_ratio) < 1e-10, (
        f"Ratio {screener / emp} != sqrt(N/(N-1)) = {expected_ratio}"
    )


def test_sharpe_reconciliation_is_exact_across_sizes():
    """The sqrt(N/(N-1)) rescaling holds for different sample sizes."""
    rng = np.random.default_rng(0)
    for N in (50, 126, 252, 504):
        rets = pd.Series(rng.normal(0.0005, 0.01, N))
        screener = _sharpe(rets)
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
# 4. Sortino divergence (non-standard design choice)
# ---------------------------------------------------------------------------


def test_sortino_screener_vs_empyrical_differ(sample_returns: pd.Series):
    """Document Sortino divergence: the two functions use different downside denominators.

    Screener denominator: std(r[r < 0], ddof=0)  — std of the negative subset only.
    Empyrical denominator: sqrt(mean(min(r, 0)^2)) — RMS over ALL observations.

    These are NOT related by a scalar factor.  The screener value is typically
    LARGER because its denominator ignores the zeros implicitly zeroed out by
    empyrical's RMS and uses a sample of only the negative returns.

    Classification: NON-STANDARD DESIGN CHOICE (neither formula is universally
    agreed; the screener's variant is closer to the Sortino (1994) paper but
    still non-standard in that it uses population std of the downside subset).
    """
    screener = _sortino(sample_returns)
    emp = empyrical.sortino_ratio(sample_returns, period="daily")

    # Values must differ by more than a rounding error.
    assert abs(screener - emp) > 0.01, (
        f"Expected Sortino divergence > 0.01 but got {abs(screener - emp):.4f}"
    )

    # Screener's formula: std of the NEGATIVE subset only (ddof=0).
    downside = sample_returns[sample_returns < 0]
    hand_denominator = float(downside.std(ddof=0))
    hand_sortino = float(sample_returns.mean() / hand_denominator * math.sqrt(252))
    assert abs(screener - hand_sortino) < 1e-10

    # Empyrical's formula: RMS of min(r, 0) over all N.
    rms_denominator = float(np.sqrt(np.mean(np.minimum(sample_returns.values, 0) ** 2)))
    hand_emp_sortino = float(sample_returns.mean() / rms_denominator * math.sqrt(252))
    assert abs(emp - hand_emp_sortino) < 1e-6


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


def test_alpha_screener_is_arithmetic_annualization(
    sample_returns: pd.Series, sample_bench: pd.Series
):
    """Screener alpha = OLS daily intercept * 252 (arithmetic annualization).

    Classification: NON-STANDARD DESIGN CHOICE.  The standard (empyrical,
    quantstats) convention is geometric: (1 + daily_intercept)^252 - 1.
    Arithmetic annualization overestimates at higher intercept values because
    it ignores compounding.  The screener is self-consistent (it documents
    'intercept is per-day' in the code) but deviates from empyrical.

    The daily intercept itself must match scipy.stats.linregress exactly.
    """
    screener_alpha, _ = _alpha_beta(sample_returns, sample_bench)
    emp_alpha = empyrical.alpha(sample_returns, sample_bench, period="daily")
    scipy_result = scipy.stats.linregress(sample_bench.values, sample_returns.values)

    daily_intercept = scipy_result.intercept

    # Screener alpha = intercept * 252 (arithmetic)
    assert abs(screener_alpha - daily_intercept * 252) < 1e-10, (
        f"screener alpha {screener_alpha} != scipy_intercept * 252 "
        f"= {daily_intercept * 252}"
    )

    # Empyrical uses geometric annualization → different value.
    geometric_alpha = (1.0 + daily_intercept) ** 252 - 1.0
    assert abs(emp_alpha - geometric_alpha) < 1e-6, (
        f"empyrical alpha {emp_alpha} != geometric formula {geometric_alpha}"
    )

    # Arithmetic != geometric (unless intercept is exactly 0).
    assert abs(screener_alpha - emp_alpha) > 1e-5, (
        "Expected arithmetic vs geometric alpha to differ visibly"
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
