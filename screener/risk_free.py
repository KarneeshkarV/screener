"""Risk-free rate series used by absolute-momentum (dual momentum) strategies.

Antonacci's absolute-momentum gate compares an asset's trailing 12-month return
with the Treasury-bill return over the same window, not with zero. That hurdle
is what this module supplies.

Three sources, in order:

- **US, preferred** — FMP's ``treasury`` series, ``month3``. This is the
  constant-maturity 3-month bill *yield*, which is what Antonacci's hurdle
  actually calls for.
- **US, fallback** — the 13-week Treasury-bill quote ``^IRX`` via yfinance,
  fetched through the backtester's own
  :class:`~screener.backtester.data.PriceFetcher` so it shares the on-disk
  cache. Note this is a *discount rate*, not a yield: it runs a few basis
  points below the FMP series and diverges more as rates rise. It is the
  fallback rather than the primary for that reason.
- **India** — no free daily T-bill series exists upstream (neither yfinance nor
  FMP publishes one), so a documented constant stands in. ``INDIA_TBILL_RATE``
  is the approximate decade average of the 91-day T-bill yield. Override it
  with ``SCREENER_RISK_FREE_INDIA`` when a different hurdle is wanted.

The constant is a real approximation. It only changes decisions for assets
whose trailing 12-month return lands within a couple of percent of the hurdle;
away from that boundary the gate is unaffected by the exact level.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from screener.backtester.data import PriceFetcher

# Annualized decimal rates.
US_TBILL_SYMBOL = "^IRX"
INDIA_TBILL_RATE = 0.060
US_TBILL_FALLBACK_RATE = 0.020
# FMP's treasury row carries every maturity; month3 is the 13-week bill ^IRX
# also tracks, so the two sources stay directly comparable.
FMP_TBILL_FIELD = "month3"
FMP_TIMEOUT_SECONDS = 30.0

TRADING_DAYS = 252


def _constant_rate(market: str) -> float:
    env = os.environ.get(f"SCREENER_RISK_FREE_{market.upper()}")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return INDIA_TBILL_RATE if market == "india" else US_TBILL_FALLBACK_RATE


def _fmp_tbill_rate(
    index: pd.DatetimeIndex, start: date, end: date
) -> pd.Series | None:
    """3-month constant-maturity bill yield from FMP, or None if unavailable.

    Returning None rather than a constant keeps this a *source* decision: the
    caller still has ``^IRX`` to try before it gives up and uses a flat rate.
    """
    from screener import fmp

    api_key = fmp.resolve_api_key()
    if not api_key:
        return None
    try:
        client = fmp.FmpClient(
            api_key, base_url=fmp.FMP_V4_BASE_URL, timeout=FMP_TIMEOUT_SECONDS
        )
        payload = client.get(
            "treasury", {"from": start.isoformat(), "to": end.isoformat()}
        )
    except Exception:  # noqa: BLE001 - a missing hurdle must not fail a backtest
        return None
    rows = (
        [r for r in payload if isinstance(r, dict)] if isinstance(payload, list) else []
    )
    observations = {
        pd.Timestamp(r["date"]): float(r[FMP_TBILL_FIELD]) / 100.0
        for r in rows
        if r.get("date") and isinstance(r.get(FMP_TBILL_FIELD), (int, float))
    }
    if not observations:
        return None
    series = pd.Series(observations).sort_index()
    return series.reindex(index, method="ffill")


def annualized_rate(
    market: str,
    index: pd.DatetimeIndex,
    fetcher: PriceFetcher,
    start: date,
    end: date,
) -> pd.Series:
    """Return the annualized risk-free rate for each date in ``index``.

    Falls back to :func:`_constant_rate` whenever the upstream series is
    unavailable, so a strategy never silently loses its hurdle.
    """
    constant = pd.Series(_constant_rate(market), index=index, dtype=float)
    if market != "us":
        return constant
    fmp_rate = _fmp_tbill_rate(index, start, end)
    if fmp_rate is not None:
        return fmp_rate.fillna(constant)
    try:
        frames = fetcher.fetch([US_TBILL_SYMBOL], start, end)
    except Exception:  # noqa: BLE001 - a missing hurdle must not fail a backtest
        return constant
    bars = frames.get(US_TBILL_SYMBOL)
    if bars is None or bars.empty or "close" not in bars:
        return constant
    rate = pd.to_numeric(bars["close"], errors="coerce").astype(float) / 100.0
    rate = rate.reindex(index, method="ffill")
    # Leading gap before ^IRX's first observation, and any all-NaN pull.
    return rate.fillna(constant)


def compounded_hurdle(rate: pd.Series, months: int = 12) -> pd.Series:
    """Convert an annualized rate series into a trailing ``months`` return hurdle.

    The gate compares a *realized* trailing return against what bills would have
    paid over the same window, so the annualized rate observed today stands in
    for the average rate across the window. This is the same simplification
    Antonacci's monthly implementation makes when it uses the current bill
    yield.
    """
    return rate * (months / 12.0)


__all__ = [
    "INDIA_TBILL_RATE",
    "US_TBILL_SYMBOL",
    "annualized_rate",
    "compounded_hurdle",
]
