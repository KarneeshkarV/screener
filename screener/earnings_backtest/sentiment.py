"""Sentiment providers for the earnings backtest.

Analyst upgrades/downgrades (yfinance, US) and options-implied sentiment —
put/call ratio + median IV — from yfinance (US) or the NSE option chain (India).

The compatibility facade re-exports the public names from this canonical
module; this module never imports the facade back.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from typing import Any, cast

import pandas as pd
import yfinance as yf

from screener.backtester.data import _configure_yfinance, call_yfinance_with_timeout
from screener.cache import cached_json_call
from screener.earnings_backtest.common import SENTIMENT_CACHE_DAYS, jsonable
from screener.options.metrics import compute_chain_metrics
from screener.options.nse_live import parse_nse_chain
from screener.options.yf_chain import chain_from_yfinance_ticker
from screener.unusual_volume.option_chain import fetch_option_chain

logger = logging.getLogger(__name__)


# ── Analyst upgrades/downgrades ────────────────────────────────────────


def fetch_analyst_sentiment(ticker: str, market: str = "us") -> dict | None:
    """Compute analyst sentiment.

    For US: uses yfinance upgrades_downgrades.
    For India: returns None; this avoids Yahoo lookups and keeps India on
    NSE/OpenScreener sources.
    """
    if market == "india":
        return None

    def _fetch() -> dict | None:
        _configure_yfinance()

        def _request() -> dict | None:
            t = yf.Ticker(ticker)
            ud = t.upgrades_downgrades
            if ud is None or ud.empty:
                return None

            if "Action" in ud.columns:
                counts = ud["Action"].value_counts().to_dict()
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
            return cast("dict[Any, Any] | None", jsonable(result))

        try:
            return call_yfinance_with_timeout(_request)
        except Exception as exc:
            logger.debug(
                "analyst_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
            )
            return None

    return cached_json_call(
        "analyst",
        (market, ticker),
        ttl_seconds=SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


# ── Options / IV sentiment ──────────────────────────────────────────────


def fetch_iv_sentiment_yf(ticker: str) -> dict | None:
    """Compute put/call ratio and IV percentile from yfinance (US only)."""

    def _fetch() -> dict | None:
        _configure_yfinance()

        def _request() -> dict | None:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return None
            today = pd.Timestamp(date.today())
            target_expiry = next(
                (d for d in dates if (pd.Timestamp(d) - today).days >= 5), dates[0]
            )
            normalized = chain_from_yfinance_ticker(
                t,
                ticker,
                [target_expiry],
                now=datetime.combine(date.today(), time.min, tzinfo=UTC),
                missing_volume_as_count=True,
            )
            if normalized is None:
                return None
            metrics = compute_chain_metrics(normalized)
            total_calls = int(metrics.call_volume)
            total_puts = int(metrics.put_volume)
            total_oi_calls = int(metrics.call_oi)
            total_oi_puts = int(metrics.put_oi)
            pc_ratio = (
                total_puts / total_calls
                if total_calls > 0
                else total_oi_puts / (total_oi_calls or 1)
            )
            median_iv = (
                metrics.median_iv * 100.0
                if metrics.median_iv is not None
                else float("nan")
            )
            return {
                "pc_ratio": round(pc_ratio, 4),
                "median_iv": round(median_iv, 2),
                "total_calls": total_calls,
                "total_puts": total_puts,
            }

        try:
            return call_yfinance_with_timeout(_request)
        except Exception as exc:
            logger.debug(
                "iv_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
            )
            return None

    return cached_json_call(
        "iv_yf",
        ticker,
        ttl_seconds=SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


def fetch_iv_sentiment_nse(symbol: str) -> dict | None:
    """Compute put/call ratio and IV from the NSE option chain.

    *symbol* is the NSE symbol (e.g. 'RELIANCE'), NOT the yfinance ticker.
    Routes through the shared ``unusual_volume`` NSE seam
    (:func:`~screener.unusual_volume.option_chain.fetch_option_chain` — primed
    session, soft-block reprime, circuit breaker) and derives the OI put/call
    ratio, median strike IV, and traded volume via
    :func:`~screener.options.metrics.compute_chain_metrics`. The returned dict
    shape matches :func:`fetch_iv_sentiment_yf` so strategy consumers are
    source-agnostic.
    """

    def _fetch() -> dict | None:
        try:
            raw = fetch_option_chain(symbol)
            if not raw or "records" not in raw:
                return None
            if not (raw.get("records") or {}).get("data"):
                return None

            chain = parse_nse_chain(raw, symbol=symbol)
            if chain is None:
                return None
            metrics = compute_chain_metrics(chain)

            # P/C ratio on OI (more stable than volume). A zero-OI call leg
            # collapses to None; preserve the legacy default of 1.0 (neutral)
            # when call OI is absent so downstream numeric comparisons in the
            # iv_sentiment strategy never see None.
            ce_oi = metrics.call_oi or None
            pe_oi = metrics.put_oi
            pc_ratio = round(pe_oi / ce_oi, 4) if ce_oi else 1.0

            median_iv = (
                metrics.median_iv * 100.0 if metrics.median_iv is not None else None
            )
            result = {
                "pc_ratio": pc_ratio,
                "median_iv": round(median_iv, 2) if median_iv is not None else None,
                "total_calls": int(metrics.call_volume),
                "total_puts": int(metrics.put_volume),
            }
            return result
        except Exception as exc:
            logger.debug(
                "nse_iv_sentiment_error", extra={"symbol": symbol, "error": str(exc)}
            )
            return None

    return cached_json_call(
        "iv_nse",
        symbol,
        ttl_seconds=SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )


def fetch_iv_sentiment(ticker: str, market: str = "us") -> dict | None:
    """Dispatch IV sentiment to the appropriate source."""
    if market == "india":
        # Strip .NS suffix for the NSE symbol.
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        return fetch_iv_sentiment_nse(symbol)
    return fetch_iv_sentiment_yf(ticker)
