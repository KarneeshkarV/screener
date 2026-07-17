"""Dependency-free Black-Scholes pricing, greeks, and IV inversion."""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt
from typing import Literal

Right = Literal["call", "put"]

_INITIAL_VOLATILITY_UPPER_BOUND = 5.0
_MAX_VOLATILITY = 100.0


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _d1_d2(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float] | None:
    if spot <= 0 or strike <= 0 or time_years <= 0 or volatility <= 0:
        return None
    root_time = sqrt(time_years)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * time_years
    ) / (volatility * root_time)
    return d1, d1 - volatility * root_time


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    right: Right,
    *,
    dividend_yield: float = 0.0,
) -> float | None:
    """Return a European option price, or ``None`` for invalid inputs."""
    ds = _d1_d2(spot, strike, time_years, rate, volatility, dividend_yield)
    if ds is None:
        return None
    d1, d2 = ds
    discounted_spot = spot * exp(-dividend_yield * time_years)
    discounted_strike = strike * exp(-rate * time_years)
    if right == "call":
        return discounted_spot * _cdf(d1) - discounted_strike * _cdf(d2)
    return discounted_strike * _cdf(-d2) - discounted_spot * _cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    right: Right,
    *,
    dividend_yield: float = 0.0,
) -> dict[str, float] | None:
    """Return delta/gamma/theta/vega/rho for a European option.

    Theta is annualized; vega and rho are changes per one absolute unit of
    volatility/rate. Callers can scale them for display if desired.
    """
    ds = _d1_d2(spot, strike, time_years, rate, volatility, dividend_yield)
    if ds is None:
        return None
    d1, d2 = ds
    root_time = sqrt(time_years)
    spot_discount = exp(-dividend_yield * time_years)
    strike_discount = exp(-rate * time_years)
    gamma = spot_discount * _pdf(d1) / (spot * volatility * root_time)
    vega = spot * spot_discount * _pdf(d1) * root_time
    common_theta = -(spot * spot_discount * _pdf(d1) * volatility) / (2 * root_time)
    if right == "call":
        delta = spot_discount * _cdf(d1)
        theta = (
            common_theta
            - rate * strike * strike_discount * _cdf(d2)
            + dividend_yield * spot * spot_discount * _cdf(d1)
        )
        rho = strike * time_years * strike_discount * _cdf(d2)
    else:
        delta = spot_discount * (_cdf(d1) - 1.0)
        theta = (
            common_theta
            + rate * strike * strike_discount * _cdf(-d2)
            - dividend_yield * spot * spot_discount * _cdf(-d1)
        )
        rho = -strike * time_years * strike_discount * _cdf(-d2)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    right: Right,
    *,
    dividend_yield: float = 0.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float | None:
    """Invert Black-Scholes IV with a bracketed bisection search.

    ``tolerance`` is measured in absolute volatility units. ``None`` is
    returned when the price does not identify a finite volatility, a bracket
    cannot be established, or the iteration budget is exhausted.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or time_years <= 0 or tolerance <= 0:
        return None
    discounted_spot = spot * exp(-dividend_yield * time_years)
    discounted_strike = strike * exp(-rate * time_years)
    intrinsic = (
        max(discounted_spot - discounted_strike, 0.0)
        if right == "call"
        else max(discounted_strike - discounted_spot, 0.0)
    )
    upper_bound = discounted_spot if right == "call" else discounted_strike
    if price < intrinsic or price >= upper_bound:
        return None

    low = 1e-6
    low_price = black_scholes_price(
        spot,
        strike,
        time_years,
        rate,
        low,
        right,
        dividend_yield=dividend_yield,
    )
    if low_price is None or price <= low_price:
        return None

    high = _INITIAL_VOLATILITY_UPPER_BOUND
    high_price = black_scholes_price(
        spot,
        strike,
        time_years,
        rate,
        high,
        right,
        dividend_yield=dividend_yield,
    )
    while high_price is not None and high_price < price and high < _MAX_VOLATILITY:
        high = min(high * 2.0, _MAX_VOLATILITY)
        high_price = black_scholes_price(
            spot,
            strike,
            time_years,
            rate,
            high,
            right,
            dividend_yield=dividend_yield,
        )
    if high_price is None or high_price < price:
        return None

    for _ in range(max(int(max_iterations), 1)):
        mid = (low + high) / 2.0
        estimate = black_scholes_price(
            spot,
            strike,
            time_years,
            rate,
            mid,
            right,
            dividend_yield=dividend_yield,
        )
        if estimate is None:
            return None
        if estimate > price:
            high = mid
        else:
            low = mid
        if high - low <= tolerance:
            return (low + high) / 2.0
    return None


__all__ = ["black_scholes_greeks", "black_scholes_price", "implied_volatility"]
