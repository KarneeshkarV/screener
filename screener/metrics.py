import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    return float(curve.iloc[-1] / curve.iloc[0] - 1.0)


def cagr(curve: pd.Series) -> float:
    if curve.empty or len(curve) < 2:
        return float("nan")
    days = (curve.index[-1] - curve.index[0]).days
    if days <= 0:
        return float("nan")
    years = days / 365.25
    ratio = curve.iloc[-1] / curve.iloc[0]
    if ratio <= 0:
        return float("nan")
    return float(ratio ** (1 / years) - 1.0)


def annualised_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0) * math.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    if r.empty or r.std(ddof=0) == 0:
        return float("nan")
    excess = r - rf / TRADING_DAYS
    return float(excess.mean() / r.std(ddof=0) * math.sqrt(TRADING_DAYS))


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    return float(dd.min())


def hit_rate(per_ticker_returns: pd.Series) -> float:
    r = per_ticker_returns.dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def alpha_beta(
    returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[Optional[float], Optional[float]]:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < 2:
        return None, None
    a = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values
    var_b = np.var(b, ddof=0)
    if var_b == 0:
        return None, None
    beta = float(np.cov(a, b, ddof=0)[0, 1] / var_b)
    alpha_daily = float(a.mean() - beta * b.mean())
    alpha_ann = alpha_daily * TRADING_DAYS
    return alpha_ann, beta


def summarise(
    basket_curve: pd.Series,
    basket_returns: pd.Series,
    benchmark_curve: pd.Series,
    benchmark_returns: pd.Series,
    per_ticker_returns: pd.Series,
) -> dict:
    a, b = alpha_beta(basket_returns, benchmark_returns)
    return {
        "basket": {
            "total_return": total_return(basket_curve),
            "cagr": cagr(basket_curve),
            "vol": annualised_vol(basket_returns),
            "sharpe": sharpe(basket_returns),
            "max_drawdown": max_drawdown(basket_curve),
            "hit_rate": hit_rate(per_ticker_returns),
            "alpha": a,
            "beta": b,
        },
        "benchmark": {
            "total_return": total_return(benchmark_curve),
            "cagr": cagr(benchmark_curve),
            "vol": annualised_vol(benchmark_returns),
            "sharpe": sharpe(benchmark_returns),
            "max_drawdown": max_drawdown(benchmark_curve),
            "hit_rate": float("nan"),
            "alpha": None,
            "beta": None,
        },
    }
