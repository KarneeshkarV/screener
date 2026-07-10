"""Sentiment providers for the earnings backtest.

Analyst upgrades/downgrades (yfinance, US) and options-implied sentiment —
put/call ratio + median IV — from yfinance (US) or the NSE option chain (India).

Split out of :mod:`screener.earnings_backtest.data`, which re-exports every
public name here for backwards compatibility. Cross-function and seam
dependencies are looked up through the ``data`` module (``data.<name>``) so the
test suite's ``monkeypatch.setattr(data, ...)`` patches keep taking effect.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import yfinance as yf

from screener.earnings_backtest import data
from screener.unusual_volume.option_chain import (
    compute_oc_iv_volume,
    compute_oc_metrics,
)

logger = logging.getLogger(__name__)


# ── Analyst upgrades/downgrades ────────────────────────────────────────


def fetch_analyst_sentiment(ticker: str, market: str = "us") -> Optional[dict]:
    """Compute analyst sentiment.

    For US: uses yfinance upgrades_downgrades.
    For India: returns None; this avoids Yahoo lookups and keeps India on
    NSE/OpenScreener sources.
    """
    if market == "india":
        return None

    def _fetch() -> Optional[dict]:
        data._configure_yfinance()
        try:
            t = yf.Ticker(ticker)
            ud = t.upgrades_downgrades
            if ud is None or ud.empty:
                return None

            if "Action" in ud.columns:
                counts = ud["Action"].value_counts().to_dict()
                # up = upgrade, reit = reiterate (half weight), down = downgrade
                upgrades = counts.get("up", 0) + 0.5 * counts.get("reit", 0)
                downgrades = counts.get("down", 0)
            elif "ToGrade" in ud.columns:
                bullish = {"Strong Buy", "Buy", "Outperform", "Overweight"}
                bearish = {"Sell", "Strong Sell", "Underperform", "Underweight"}
                grades = ud["ToGrade"].value_counts().to_dict()
                upgrades = sum(grades.get(g, 0) for g in bullish)
                downgrades = sum(grades.get(g, 0) for g in bearish)
                counts = {str(k): int(v) for k, v in grades.items()}
            else:
                return None

            result = {
                "upgrades": upgrades,
                "downgrades": downgrades,
                "net": upgrades - downgrades,
                "grade_counts": counts if "Action" in ud.columns else {},
            }
            return cast("dict[Any, Any] | None", data._jsonable(result))
        except Exception as exc:
            logger.debug(
                "analyst_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
            )
            return None

    return data.cached_json_call(
        "analyst",
        (market, ticker),
        ttl_seconds=data.SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


# ── Options / IV sentiment ──────────────────────────────────────────────


def fetch_iv_sentiment_yf(ticker: str) -> Optional[dict]:
    """Compute put/call ratio and IV percentile from yfinance (US only)."""

    def _fetch() -> Optional[dict]:
        data._configure_yfinance()
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return None

            today = pd.Timestamp(date.today())
            target_expiry = None
            for d in dates:
                exp = pd.Timestamp(d)
                if (exp - today).days >= 5:
                    target_expiry = d
                    break
            if target_expiry is None:
                target_expiry = dates[0]

            chain = t.option_chain(target_expiry)
            calls = chain.calls
            puts = chain.puts
            if calls.empty and puts.empty:
                return None

            total_calls = (
                int(calls["volume"].sum()) if "volume" in calls.columns else len(calls)
            )
            total_puts = (
                int(puts["volume"].sum()) if "volume" in puts.columns else len(puts)
            )
            total_oi_calls = (
                int(calls["openInterest"].sum())
                if "openInterest" in calls.columns
                else 0
            )
            total_oi_puts = (
                int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
            )

            denom = total_calls or 1
            pc_ratio = (
                total_puts / denom
                if total_calls > 0
                else (total_oi_puts / (total_oi_calls or 1))
            )

            iv_vals = []
            if "impliedVolatility" in calls.columns:
                iv_vals.extend(calls["impliedVolatility"].dropna().tolist())
            if "impliedVolatility" in puts.columns:
                iv_vals.extend(puts["impliedVolatility"].dropna().tolist())
            # Median IV across all strikes (expressed as %, e.g. 40.11 for 40.11%)
            # yfinance returns IV as decimals (0.4011 = 40.11%), so multiply by 100
            iv_vals_pct = [v * 100 for v in iv_vals]
            median_iv = (
                float(np.percentile(iv_vals_pct, 50)) if iv_vals_pct else float("nan")
            )

            result = {
                "pc_ratio": round(pc_ratio, 4),
                "median_iv": round(median_iv, 2),
                "total_calls": total_calls,
                "total_puts": total_puts,
            }
            return result
        except Exception as exc:
            logger.debug(
                "iv_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
            )
            return None

    return data.cached_json_call(
        "iv_yf",
        ticker,
        ttl_seconds=data.SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


def fetch_iv_sentiment_nse(symbol: str) -> Optional[dict]:
    """Compute put/call ratio and IV from the NSE option chain.

    *symbol* is the NSE symbol (e.g. 'RELIANCE'), NOT the yfinance ticker.
    Routes through the shared ``unusual_volume`` NSE seam
    (:func:`~screener.unusual_volume.option_chain.fetch_option_chain` — primed
    session, soft-block reprime, circuit breaker) and reuses
    :func:`~screener.unusual_volume.option_chain.compute_oc_metrics` for the OI
    put/call ratio and :func:`~screener.unusual_volume.option_chain.compute_oc_iv_volume`
    for median strike IV and traded volume. The returned dict shape matches
    :func:`fetch_iv_sentiment_yf` so strategy consumers are source-agnostic.
    """

    def _fetch() -> Optional[dict]:
        try:
            raw = data.fetch_option_chain(symbol)
            if not raw or "records" not in raw:
                return None
            if not (raw.get("records") or {}).get("data"):
                return None

            oc_metrics = compute_oc_metrics(raw)
            iv_volume = compute_oc_iv_volume(raw)

            # P/C ratio on OI (more stable than volume). ``compute_oc_metrics``
            # collapses a zero-OI leg to None; preserve the legacy default of
            # 1.0 (neutral) when call OI is absent so downstream numeric
            # comparisons in the iv_sentiment strategy never see None.
            ce_oi = oc_metrics.get("ce_oi")
            pe_oi = oc_metrics.get("pe_oi") or 0.0
            pc_ratio = round(pe_oi / ce_oi, 4) if ce_oi else 1.0

            median_iv = iv_volume["median_iv"]
            result = {
                "pc_ratio": pc_ratio,
                "median_iv": round(median_iv, 2) if median_iv is not None else None,
                "total_calls": int(iv_volume["total_call_volume"]),
                "total_puts": int(iv_volume["total_put_volume"]),
            }
            return result
        except Exception as exc:
            logger.debug(
                "nse_iv_sentiment_error", extra={"symbol": symbol, "error": str(exc)}
            )
            return None

    return data.cached_json_call(
        "iv_nse",
        symbol,
        ttl_seconds=data.SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


def fetch_iv_sentiment(ticker: str, market: str = "us") -> Optional[dict]:
    """Dispatch IV sentiment to the appropriate source."""
    if market == "india":
        # Strip .NS suffix for the NSE symbol.
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        return data.fetch_iv_sentiment_nse(symbol)
    return data.fetch_iv_sentiment_yf(ticker)
