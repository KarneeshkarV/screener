"""scipy-based reference witnesses for _phi, _phi_inv, _psr, and _dsr.

Since no public library implements PSR/DSR, we build INDEPENDENT witnesses
from the López de Prado (2012/2014) formulae using scipy.stats primitives.
A red test means the screener's PSR/DSR math is wrong.

Key cross-checks encoded here:

1. _phi  ←→ scipy.stats.norm.cdf    (agrees to 1e-9)
2. _phi_inv ←→ scipy.stats.norm.ppf (agrees to 1e-9)
3. PSR witness vs _psr: replicate the formula from metrics.py lines ~129-146
   using scipy.stats.norm.cdf + scipy.stats.skew(bias=False) + scipy.stats.kurtosis
   (these are the EXACT equivalents of pandas .skew() / .kurt() — confirmed).
4. DSR witness vs _dsr: replicate lines ~149-168 using scipy.stats.norm.ppf
   for _phi_inv calls in the sr0_annual computation.
5. Documented guards:
   - _psr returns 0.0 for len < 30
   - _dsr with n_trials <= 1 reduces to _psr(daily, 0.0)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.stats
import pytest

from screener.backtester.metrics import (
    _dsr,
    _phi,
    _phi_inv,
    _psr,
)

# Euler–Mascheroni constant (must match metrics.py)
_EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def long_returns() -> pd.Series:
    """252 synthetic returns with non-trivial skew/kurtosis (seed=42)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(loc=0.0005, scale=0.01, size=252))


@pytest.fixture(scope="module")
def medium_returns() -> pd.Series:
    """60-bar returns — above the PSR guard threshold (seed=17)."""
    rng = np.random.default_rng(17)
    return pd.Series(rng.normal(loc=0.0003, scale=0.012, size=60))


# ---------------------------------------------------------------------------
# Witness helpers
# ---------------------------------------------------------------------------


def _scipy_psr(daily: pd.Series, sr_benchmark_annual: float = 0.0) -> float:
    """Independent PSR witness built from scipy primitives.

    Replicates metrics.py _psr() formula using:
      - scipy.stats.norm.cdf  in place of _phi
      - scipy.stats.skew(bias=False) in place of daily.skew()
      - scipy.stats.kurtosis(fisher=True, bias=False) in place of daily.kurt()
    These are the exact equivalents (confirmed empirically for N in [30,252]).
    """
    T = len(daily)
    if T < 30:
        return 0.0
    sr_bench_per = sr_benchmark_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    std0 = float(daily.std(ddof=0))
    if std0 == 0:
        sr_per = 0.0
        skew = 0.0
        kurt_excess = 0.0
    else:
        # Per-period Sharpe computed INDEPENDENTLY of metrics._sharpe, so this
        # witness cannot inherit a Sharpe regression: _sharpe just annualizes
        # by sqrt(252), hence the per-period ratio is mean / std(ddof=0).
        sr_per = float(daily.mean()) / std0
        skew = float(scipy.stats.skew(daily.values, bias=False))
        kurt_excess = float(scipy.stats.kurtosis(daily.values, fisher=True, bias=False))
    denom_sq = 1.0 - skew * sr_per + (kurt_excess / 4.0) * sr_per**2
    denom = math.sqrt(max(denom_sq, 1e-12))
    z = (sr_per - sr_bench_per) * math.sqrt(max(T - 1, 1)) / denom
    return float(scipy.stats.norm.cdf(z))


def _scipy_dsr(
    daily: pd.Series,
    n_trials: int = 1,
    sr_trial_std_annual: float = 0.5,
) -> float:
    """Independent DSR witness built from scipy primitives.

    Replicates metrics.py _dsr() using scipy.stats.norm.ppf in place of
    _phi_inv for the sr0_annual computation.
    """
    if n_trials <= 1:
        return _scipy_psr(daily, 0.0)
    sr0_annual = sr_trial_std_annual * (
        (1.0 - _EULER_MASCHERONI) * scipy.stats.norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * scipy.stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    return _scipy_psr(daily, sr0_annual)


# ---------------------------------------------------------------------------
# 1. _phi vs scipy.stats.norm.cdf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z", [-3.0, -1.96, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96, 3.0])
def test_phi_matches_scipy_norm_cdf(z: float):
    """_phi(z) == scipy.stats.norm.cdf(z) to 1e-9 for all standard z values."""
    impl = _phi(z)
    ref = scipy.stats.norm.cdf(z)
    assert abs(impl - ref) < 1e-9, (
        f"_phi({z}) = {impl} differs from scipy {ref} by {abs(impl - ref):.2e}"
    )


def test_phi_boundary_zero():
    """_phi(0) = 0.5 exactly (half the normal distribution lies below 0)."""
    assert abs(_phi(0.0) - 0.5) < 1e-15


def test_phi_symmetry():
    """_phi(-z) + _phi(z) = 1 (symmetry of the normal CDF)."""
    for z in [0.5, 1.0, 2.0]:
        assert abs(_phi(-z) + _phi(z) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. _phi_inv vs scipy.stats.norm.ppf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
def test_phi_inv_matches_scipy_norm_ppf(p: float):
    """_phi_inv(p) == scipy.stats.norm.ppf(p) to 1e-9 for standard probabilities."""
    impl = _phi_inv(p)
    ref = scipy.stats.norm.ppf(p)
    assert abs(impl - ref) < 1e-9, (
        f"_phi_inv({p}) = {impl} differs from scipy {ref} by {abs(impl - ref):.2e}"
    )


def test_phi_inv_at_half_is_zero():
    """_phi_inv(0.5) = 0.0 (median of the standard normal)."""
    assert abs(_phi_inv(0.5)) < 1e-9


def test_phi_inv_roundtrip():
    """phi(phi_inv(p)) == p for p in (0, 1)."""
    for p in [0.05, 0.3, 0.5, 0.7, 0.95]:
        assert abs(_phi(_phi_inv(p)) - p) < 1e-9


def test_phi_inv_extreme_boundaries():
    """_phi_inv(0) = -inf; _phi_inv(1) = +inf."""
    assert _phi_inv(0.0) == -float("inf")
    assert _phi_inv(1.0) == float("inf")


# ---------------------------------------------------------------------------
# 3. scipy skew/kurtosis match pandas — precondition for PSR witness validity
# ---------------------------------------------------------------------------


def test_scipy_skew_matches_pandas_skew(long_returns: pd.Series):
    """scipy.stats.skew(x, bias=False) == x.skew() (both are bias-corrected).

    This is the key precondition: _psr uses x.skew() and the witness uses
    scipy.stats.skew(bias=False).  If they diverge, the witness would be
    measuring a different thing.
    """
    pd_skew = float(long_returns.skew())
    sc_skew = float(scipy.stats.skew(long_returns.values, bias=False))
    assert abs(pd_skew - sc_skew) < 1e-10, (
        f"pandas skew {pd_skew} != scipy skew {sc_skew}"
    )


def test_scipy_kurtosis_matches_pandas_kurt(long_returns: pd.Series):
    """scipy.stats.kurtosis(x, fisher=True, bias=False) == x.kurt().

    Both produce the bias-corrected EXCESS kurtosis.  Verified here as a
    precondition for the PSR witness.
    """
    pd_kurt = float(long_returns.kurt())
    sc_kurt = float(scipy.stats.kurtosis(long_returns.values, fisher=True, bias=False))
    assert abs(pd_kurt - sc_kurt) < 1e-10, (
        f"pandas kurt {pd_kurt} != scipy kurtosis {sc_kurt}"
    )


# ---------------------------------------------------------------------------
# 4. PSR witness vs _psr implementation
# ---------------------------------------------------------------------------


def test_psr_matches_scipy_witness_long_series(long_returns: pd.Series):
    """_psr(returns, 0.0) matches the scipy witness to 1e-9."""
    impl = _psr(long_returns, sr_benchmark_annual=0.0)
    witness = _scipy_psr(long_returns, sr_benchmark_annual=0.0)
    assert abs(impl - witness) < 1e-9, f"PSR impl {impl} != scipy witness {witness}"


def test_psr_matches_scipy_witness_medium_series(medium_returns: pd.Series):
    """_psr works correctly for series of 60 bars (above the 30-bar guard)."""
    impl = _psr(medium_returns, sr_benchmark_annual=0.0)
    witness = _scipy_psr(medium_returns, sr_benchmark_annual=0.0)
    assert abs(impl - witness) < 1e-9


def test_psr_with_nonzero_benchmark_matches_witness(long_returns: pd.Series):
    """_psr with a positive Sharpe benchmark is correctly adjusted."""
    for sr_bench in [0.0, 0.5, 1.0, -0.5]:
        impl = _psr(long_returns, sr_benchmark_annual=sr_bench)
        witness = _scipy_psr(long_returns, sr_benchmark_annual=sr_bench)
        assert abs(impl - witness) < 1e-9, (
            f"PSR(sr_bench={sr_bench}): impl {impl} != witness {witness}"
        )


def test_psr_is_in_unit_interval(long_returns: pd.Series):
    """PSR must be in [0, 1] — it is a probability."""
    for sr_bench in [-1.0, 0.0, 0.5, 1.0, 2.0]:
        result = _psr(long_returns, sr_benchmark_annual=sr_bench)
        assert 0.0 <= result <= 1.0, (
            f"PSR(sr_bench={sr_bench}) = {result} is outside [0, 1]"
        )


def test_psr_increases_as_benchmark_decreases(long_returns: pd.Series):
    """Lower SR benchmark → easier to beat → PSR monotonically increases."""
    benchmarks = [2.0, 1.0, 0.5, 0.0, -0.5]
    psrs = [_psr(long_returns, b) for b in benchmarks]
    for i in range(len(psrs) - 1):
        assert psrs[i] < psrs[i + 1], (
            f"PSR not monotone: sr_bench {benchmarks[i]}→{benchmarks[i + 1]}: "
            f"{psrs[i]}→{psrs[i + 1]}"
        )


def test_psr_returns_zero_for_short_series():
    """Guard: len < 30 → return 0.0, independent of content."""
    rng = np.random.default_rng(5)
    for n in [0, 1, 10, 29]:
        series = pd.Series(rng.normal(0.001, 0.01, n) if n > 0 else [], dtype=float)
        assert _psr(series) == 0.0, f"PSR(len={n}) should be 0.0"


# ---------------------------------------------------------------------------
# 5. DSR witness vs _dsr implementation
# ---------------------------------------------------------------------------


def test_dsr_matches_scipy_witness_n_trials_1(long_returns: pd.Series):
    """DSR with n_trials=1 → _psr(daily, 0.0) → matches PSR witness."""
    impl = _dsr(long_returns, n_trials=1)
    witness = _scipy_dsr(long_returns, n_trials=1)
    psr_zero = _psr(long_returns, 0.0)
    assert abs(impl - psr_zero) < 1e-15  # exact equality (same code path)
    assert abs(impl - witness) < 1e-9


@pytest.mark.parametrize("n_trials", [2, 5, 10, 20, 50])
def test_dsr_matches_scipy_witness_multiple_trials(
    long_returns: pd.Series, n_trials: int
):
    """DSR with multiple trials matches the scipy witness to 1e-9."""
    impl = _dsr(long_returns, n_trials=n_trials)
    witness = _scipy_dsr(long_returns, n_trials=n_trials)
    assert abs(impl - witness) < 1e-9, (
        f"DSR(n={n_trials}): impl {impl} != witness {witness}"
    )


def test_dsr_decreases_as_n_trials_increases(long_returns: pd.Series):
    """More trials → higher expected-max benchmark → DSR is monotone non-increasing."""
    prev = _dsr(long_returns, n_trials=1)
    for n in [2, 5, 10, 20, 50]:
        curr = _dsr(long_returns, n_trials=n)
        assert curr <= prev + 1e-12, (
            f"DSR not monotone non-increasing: n={n}, prev={prev}, curr={curr}"
        )
        prev = curr


def test_dsr_n_trials_0_equals_psr(long_returns: pd.Series):
    """n_trials=0 satisfies n_trials <= 1 → same code path as n_trials=1."""
    dsr_0 = _dsr(long_returns, n_trials=0)
    psr_0 = _psr(long_returns, 0.0)
    assert dsr_0 == psr_0  # exact equality — same function call


def test_dsr_sr0_annual_formula_uses_euler_mascheroni(long_returns: pd.Series):
    """The sr0_annual benchmark is computed via the Euler–Mascheroni constant.

    Verify that the sr0_annual computed inside _dsr (n=10) equals the value
    produced by the scipy witness.  We do this indirectly: the witness gives
    the same DSR ↔ it used the same sr0_annual.
    """
    n_trials = 10
    sr_std = 0.5
    # Compute sr0_annual with scipy
    sr0 = sr_std * (
        (1.0 - _EULER_MASCHERONI) * scipy.stats.norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * scipy.stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    # PSR with this benchmark should equal _dsr(daily, n_trials=10)
    psr_bench = _psr(long_returns, sr_benchmark_annual=sr0)
    dsr_impl = _dsr(long_returns, n_trials=n_trials)
    assert abs(dsr_impl - psr_bench) < 1e-12, (
        f"DSR(n=10) {dsr_impl} != PSR(sr0={sr0:.6f}) {psr_bench}"
    )
