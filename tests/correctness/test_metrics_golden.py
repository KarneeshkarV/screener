"""Hand-derived golden values for screener metrics.

Expected values are RE-DERIVED here with plain arithmetic in comments —
NOT copied from the implementation output.  A mismatch is a real defect.

Convention reminders (all population std, ddof=0):
  * _sharpe / _sortino / _vol_annual work on DAILY-RETURNS Series.
  * _cagr / _max_drawdown / _calmar work on an EQUITY-CURVE Series.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from screener.backtester.metrics import (
    _calmar,
    _cagr,
    _daily_returns,
    _max_drawdown,
    _sharpe,
    _sortino,
    _vol_annual,
)

# ---------------------------------------------------------------------------
# _sharpe golden
# ---------------------------------------------------------------------------


def test_sharpe_five_bar_golden():
    """_sharpe([0.01, 0.02, -0.01, 0.00, 0.03])

    Derivation (rf=0):
        mean   = (0.01+0.02-0.01+0.00+0.03) / 5 = 0.05/5 = 0.01
        deviations from mean: [0, 0.01, -0.02, -0.01, 0.02]
        var_pop = (0^2 + 0.01^2 + 0.02^2 + 0.01^2 + 0.02^2) / 5
               = (0 + 1e-4 + 4e-4 + 1e-4 + 4e-4) / 5 = 1e-3 / 5 = 2e-4
        std_pop = sqrt(2e-4) = sqrt(2) / 100 ≈ 0.01414213...
        sharpe  = 0.01 / 0.01414213... * sqrt(252) ≈ 11.2249722
    """
    returns = pd.Series([0.01, 0.02, -0.01, 0.0, 0.03])

    # Arithmetic re-derivation (no magic numbers — derive from constants).
    mean = 0.05 / 5  # = 0.01
    var_pop = 1e-3 / 5  # = 2e-4
    std_pop = math.sqrt(var_pop)  # = sqrt(2)/100
    expected = mean / std_pop * math.sqrt(252)

    assert abs(_sharpe(returns) - expected) < 1e-7


def test_sharpe_matches_numpy_formula_directly():
    """_sharpe matches the population-std formula applied via numpy."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 252))
    expected = float(returns.mean() / returns.std(ddof=0) * math.sqrt(252))
    assert abs(_sharpe(returns) - expected) < 1e-12


# ---------------------------------------------------------------------------
# _sortino golden
# ---------------------------------------------------------------------------


def test_sortino_five_bar_golden():
    """_sortino([0.02, -0.01, 0.03, -0.02, 0.01])

    Derivation (rf=0):
        mean           = (0.02-0.01+0.03-0.02+0.01) / 5 = 0.03/5 = 0.006
        downside only  = [-0.01, -0.02]
        mean_downside  = (-0.01 + -0.02) / 2 = -0.015
        deviations     = [-0.01-(-0.015), -0.02-(-0.015)] = [0.005, -0.005]
        var_pop        = (0.005^2 + 0.005^2) / 2 = 5e-5
        std_pop        = sqrt(5e-5) = 0.005
        sortino        = 0.006 / 0.005 * sqrt(252) = 1.2 * sqrt(252) ≈ 19.0494
    """
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])

    mean = 3 / 500  # = 0.006
    std_down = 5e-3  # = 0.005 (derived above)
    expected = mean / std_down * math.sqrt(252)

    assert abs(_sortino(returns) - expected) < 1e-7


# ---------------------------------------------------------------------------
# _vol_annual golden
# ---------------------------------------------------------------------------


def test_vol_annual_golden():
    """_vol_annual = std(ddof=0) * sqrt(252).

    Use a three-element series for a clean hand derivation:
        returns = [0.01, -0.01, 0.0]
        mean = 0; deviations = [0.01, -0.01, 0]; var_pop = 2e-4/3
        std_pop = sqrt(2e-4/3); vol_annual = std_pop * sqrt(252)
    """
    returns = pd.Series([0.01, -0.01, 0.0])
    var_pop = (0.01**2 + 0.01**2 + 0.0**2) / 3  # mean=0
    std_pop = math.sqrt(var_pop)
    expected = std_pop * math.sqrt(252)

    assert abs(_vol_annual(returns) - expected) < 1e-12


# ---------------------------------------------------------------------------
# _cagr golden
# ---------------------------------------------------------------------------


def test_cagr_linear_equity_one_year_golden():
    """_cagr(equity over 253 points) = 1.0.

    Derivation (annualize over elapsed return periods, N-1):
        253 points span 252 daily returns → years = (253-1)/252 = 1
        (200 / 100) ^ (1 / 1) - 1 = 2 - 1 = 1.0
    """
    equity = pd.Series(np.linspace(100.0, 200.0, 253))
    assert abs(_cagr(equity) - 1.0) < 1e-12


def test_cagr_two_year_equity_golden():
    """_cagr(linspace(100, 200, 505)) matches the elapsed-periods formula.

    505 points span 504 daily returns → years = 504/252 = 2
        (200/100)^(1/2) - 1 = sqrt(2) - 1
    """
    equity = pd.Series(np.linspace(100.0, 200.0, 505))
    years = (505 - 1) / 252
    expected = (200.0 / 100.0) ** (1.0 / years) - 1.0
    assert abs(_cagr(equity) - expected) < 1e-12


def test_cagr_decreasing_equity_is_negative():
    """Falling equity curve → CAGR < 0."""
    equity = pd.Series(np.linspace(200.0, 100.0, 252))
    assert _cagr(equity) < 0.0


# ---------------------------------------------------------------------------
# _max_drawdown golden
# ---------------------------------------------------------------------------


def test_max_drawdown_peak_trough_golden():
    """_max_drawdown([100, 120, 90, 110, 80, 95])

    Derivation:
        running peak = [100, 120, 120, 120, 120, 120]
        dd ratio     = (equity - peak) / peak
                     = [0/100, 0/120, -30/120, -10/120, -40/120, -25/120]
                     = [0, 0, -0.25, -0.0833, -0.3333, -0.2083]
        min = -40/120 = -1/3 ≈ -0.333...
    """
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 95.0])
    expected = -40.0 / 120.0  # = -1/3

    assert abs(_max_drawdown(equity) - expected) < 1e-12


def test_max_drawdown_monotone_rising_is_zero():
    """Strictly increasing equity → no drawdown → max_drawdown = 0."""
    equity = pd.Series(np.linspace(100.0, 200.0, 50))
    assert _max_drawdown(equity) == 0.0


def test_max_drawdown_matches_empyrical_on_random_equity():
    """_max_drawdown(equity) agrees with empyrical.max_drawdown(returns) to 1e-10.

    (empyrical computes from the cumulative returns, which is equivalent to
    the equity-curve ratio — no convention differences.)
    """
    import empyrical

    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0005, 0.01, 252))
    equity = pd.Series(
        np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns.values)])
    )

    screener_mdd = _max_drawdown(equity)
    emp_mdd = empyrical.max_drawdown(returns)

    assert abs(screener_mdd - emp_mdd) < 1e-10


# ---------------------------------------------------------------------------
# _calmar golden
# ---------------------------------------------------------------------------


def test_calmar_matches_empyrical_oracle():
    """_calmar(equity) agrees with empyrical.calmar_ratio(returns).

    Independent oracle (not a self-composition): empyrical computes Calmar as
    annual_return / |max_drawdown| straight from the returns series. After the
    CAGR off-by-one fix, screener _cagr and _max_drawdown both match empyrical,
    so _calmar matches empyrical's Calmar too.
    """
    import empyrical

    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.001, 0.01, 252))
    equity = pd.Series(
        np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns.values)])
    )

    screener = _calmar(equity)
    emp = empyrical.calmar_ratio(returns)
    assert abs(screener - emp) < 1e-9, f"_calmar {screener} != empyrical {emp}"


def test_calmar_returns_zero_when_no_drawdown():
    """No drawdown (mdd >= 0) → _calmar returns 0.0 (guarded divide-by-zero)."""
    equity = pd.Series(np.linspace(100.0, 200.0, 252))
    # max_drawdown = 0 → calmar guard triggers → 0.0
    assert _calmar(equity) == 0.0


# ---------------------------------------------------------------------------
# _daily_returns
# ---------------------------------------------------------------------------


def test_daily_returns_pct_change_golden():
    """_daily_returns([100, 110, 99]) = [0.1, -0.1/1.1]

    Derivation:
        r[0] = (110 - 100) / 100 = 0.10
        r[1] = (99  - 110) / 110 = -11/110 = -0.10000 (exact? no: -11/110)
    """
    equity = pd.Series([100.0, 110.0, 99.0])
    expected = [10.0 / 100.0, (99.0 - 110.0) / 110.0]
    result = _daily_returns(equity).tolist()

    assert len(result) == 2
    assert abs(result[0] - expected[0]) < 1e-12
    assert abs(result[1] - expected[1]) < 1e-12
