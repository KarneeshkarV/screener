"""Data acquisition for earnings backtest.

Fetches earnings dates, price bars, volume, analyst recommendations,
and options data. Uses yfinance for US and jugaad_data for India (NSE).

Designed for batch processing under tight RAM constraints (~2 GB).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from screener.backtester.data import (
    YFinancePriceFetcher,
    _configure_yfinance,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "earnings_backtest"


# ── Universe loaders ────────────────────────────────────────────────────


def load_sp500() -> list[str]:
    """Return current S&P 500 ticker list."""
    from screener.universes import load_current_universe

    univ = load_current_universe("sp500")
    return list(univ.symbols)


def load_nifty500() -> list[str]:
    """Return Nifty 500 ticker list with .NS suffix."""
    import io
    import requests
    from screener.resilience import call_with_resilience

    cache_path = CACHE_DIR / "nifty500_symbols.txt"
    if cache_path.exists():
        age = (date.today() - date.fromtimestamp(cache_path.stat().st_mtime)).days
        if age < 7:
            symbols = cache_path.read_text().strip().splitlines()
            if symbols:
                return symbols

    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

    def _fetch():
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text

    text = call_with_resilience("nse", "nifty500 constituents", _fetch, fallback=None)
    if text is None:
        raise RuntimeError("Nifty 500 constituents unavailable")

    df = pd.read_csv(io.StringIO(text))
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    symbols = df[col].dropna().astype(str).str.strip().str.upper().tolist()
    symbols = [f"{s}.NS" for s in symbols]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(symbols))
    return symbols


def load_universe(market: str) -> list[str]:
    if market == "us":
        return load_sp500()
    if market == "india":
        return load_nifty500()
    raise ValueError(f"Unknown market: {market!r}")


# ── Earnings dates (yfinance) ────────────────────────────────────────────


def fetch_earnings_dates_yf(
    ticker: str,
    years: int = 3,
) -> Optional[pd.DataFrame]:
    """Return yfinance earnings_dates for *ticker*."""
    _configure_yfinance()
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return None
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
        cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
        ed = ed[ed.index >= cutoff]
        return ed if not ed.empty else None
    except Exception as exc:
        logger.debug(
            "earnings_dates_failed", extra={"ticker": ticker, "error": str(exc)}
        )
        return None


def fetch_earnings_dates_nse() -> Optional[pd.DataFrame]:
    """Fetch earnings result dates from NSE corporate announcements via jugaad_data."""
    try:
        from jugaad_data.nse import NSELive

        nse = NSELive()
        announcements = nse.corporate_announcements()
        if not announcements:
            return None

        rows = []
        for ann in announcements:
            desc = str(ann.get("desc", "")).lower()
            text = str(ann.get("attchmntText", "")).lower()
            # Filter for financial results announcements
            if any(
                kw in desc or kw in text
                for kw in [
                    "financial result",
                    "earnings",
                    "quarterly result",
                    "audited financial",
                    "unaudited financial",
                ]
            ):
                symbol = ann.get("symbol", "")
                dt_str = ann.get("sort_date", "")
                if not symbol or not dt_str:
                    continue
                try:
                    ann_date = pd.Timestamp(dt_str).normalize()
                except Exception:
                    continue
                rows.append(
                    {
                        "ticker": f"{symbol}.NS",
                        "earnings_date": ann_date,
                        "desc": ann.get("desc", ""),
                    }
                )

        if not rows:
            return None
        return pd.DataFrame(rows)

    except Exception as exc:
        logger.warning("nse_earnings_fetch_failed", extra={"error": str(exc)})
        return None


# ── Batch earnings collector ────────────────────────────────────────────


def collect_earnings_events(
    tickers: list[str],
    years: int = 3,
    batch_size: int = 50,
    market: str = "us",
) -> pd.DataFrame:
    """Collect earnings dates for all *tickers*.

    For India: tries jugaad_data (NSE announcements) first, falls back to yfinance.
    For US: uses yfinance only.
    """
    rows: list[dict] = []

    if market == "india":
        # Try NSE corporate announcements first (broader coverage)
        nse_events = fetch_earnings_dates_nse()
        if nse_events is not None and not nse_events.empty:
            # Only keep tickers in our universe
            ticker_set = set(tickers)
            filtered = nse_events[nse_events["ticker"].isin(ticker_set)]
            # Convert to unified format
            cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
            filtered = filtered[filtered["earnings_date"] >= cutoff]
            for _, row in filtered.iterrows():
                rows.append(
                    {
                        "ticker": row["ticker"],
                        "earnings_date": row["earnings_date"],
                        "eps_estimate": float("nan"),
                        "reported_eps": float("nan"),
                        "surprise_pct": float("nan"),
                    }
                )
            # Fill in EPS from yfinance for tickers that have it
            nse_found = (
                set(rows_dict["ticker"] for rows_dict in rows) if rows else set()
            )
            missing = [t for t in tickers if t not in nse_found]
            if missing:
                logger.info("yfinance_earnings_fill", extra={"count": len(missing)})
                for i in range(0, len(missing), batch_size):
                    batch = missing[i : i + batch_size]
                    for ticker in batch:
                        ed = fetch_earnings_dates_yf(ticker, years=years)
                        if ed is None:
                            continue
                        for idx, row in ed.iterrows():
                            rows.append(
                                {
                                    "ticker": ticker,
                                    "earnings_date": idx.date()
                                    if hasattr(idx, "date")
                                    else idx,
                                    "eps_estimate": row.get(
                                        "EPS Estimate", float("nan")
                                    ),
                                    "reported_eps": row.get(
                                        "Reported EPS", float("nan")
                                    ),
                                    "surprise_pct": row.get(
                                        "Surprise(%)", float("nan")
                                    ),
                                }
                            )
        else:
            # Fallback: yfinance for all
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                for ticker in batch:
                    ed = fetch_earnings_dates_yf(ticker, years=years)
                    if ed is None:
                        continue
                    for idx, row in ed.iterrows():
                        rows.append(
                            {
                                "ticker": ticker,
                                "earnings_date": idx.date()
                                if hasattr(idx, "date")
                                else idx,
                                "eps_estimate": row.get("EPS Estimate", float("nan")),
                                "reported_eps": row.get("Reported EPS", float("nan")),
                                "surprise_pct": row.get("Surprise(%)", float("nan")),
                            }
                        )
    else:
        # US: yfinance
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(
                "earnings_batch",
                extra={"batch": f"{i}-{i + len(batch)}", "size": len(batch)},
            )
            for ticker in batch:
                try:
                    ed = fetch_earnings_dates_yf(ticker, years=years)
                    if ed is None:
                        continue
                    for idx, row in ed.iterrows():
                        rows.append(
                            {
                                "ticker": ticker,
                                "earnings_date": idx.date()
                                if hasattr(idx, "date")
                                else idx,
                                "eps_estimate": row.get("EPS Estimate", float("nan")),
                                "reported_eps": row.get("Reported EPS", float("nan")),
                                "surprise_pct": row.get("Surprise(%)", float("nan")),
                            }
                        )
                except Exception as exc:
                    logger.debug(
                        "earnings_collect_error",
                        extra={"ticker": ticker, "error": str(exc)},
                    )
                    continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "earnings_date",
                "eps_estimate",
                "reported_eps",
                "surprise_pct",
            ]
        )
    return pd.DataFrame(rows)


# ── Analyst upgrades/downgrades ────────────────────────────────────────


def fetch_analyst_sentiment(ticker: str, market: str = "us") -> Optional[dict]:
    """Compute analyst sentiment.

    For US: uses yfinance upgrades_downgrades.
    For India: uses yfinance (usually empty) — returns None gracefully.
    """
    try:
        _configure_yfinance()
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

        return {
            "upgrades": upgrades,
            "downgrades": downgrades,
            "net": upgrades - downgrades,
            "grade_counts": counts if "Action" in ud.columns else {},
        }
    except Exception as exc:
        logger.debug(
            "analyst_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
        )
        return None


# ── Options / IV sentiment ──────────────────────────────────────────────


def fetch_iv_sentiment_yf(ticker: str) -> Optional[dict]:
    """Compute put/call ratio and IV percentile from yfinance (US only)."""
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
            int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
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

        return {
            "pc_ratio": round(pc_ratio, 4),
            "median_iv": round(median_iv, 2),
            "total_calls": total_calls,
            "total_puts": total_puts,
        }
    except Exception as exc:
        logger.debug("iv_sentiment_error", extra={"ticker": ticker, "error": str(exc)})
        return None


def fetch_iv_sentiment_nse(symbol: str) -> Optional[dict]:
    """Compute put/call ratio and IV from NSE option chain via jugaad_data.

    *symbol* is the NSE symbol (e.g. 'RELIANCE'), NOT the yfinance ticker.
    Uses openInterest for P/C ratio (more stable than volume).
    NSE option chain does include impliedVolatility per strike.
    """
    try:
        from jugaad_data.nse import NSELive

        nse = NSELive()
        oc = nse.equities_option_chain(symbol)
        if not oc or "records" not in oc:
            return None

        data = oc["records"].get("data", [])
        if not data:
            return None

        total_ce_oi = 0
        total_pe_oi = 0
        total_ce_vol = 0
        total_pe_vol = 0
        iv_vals = []

        for item in data:
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            if ce:
                total_ce_oi += ce.get("openInterest", 0) or 0
                total_ce_vol += ce.get("totalTradedVolume", 0) or 0
                iv = ce.get("impliedVolatility")
                if iv and iv > 0:
                    iv_vals.append(float(iv))
            if pe:
                total_pe_oi += pe.get("openInterest", 0) or 0
                total_pe_vol += pe.get("totalTradedVolume", 0) or 0
                iv = pe.get("impliedVolatility")
                if iv and iv > 0:
                    iv_vals.append(float(iv))

        # P/C ratio on OI (more stable than volume)
        denom = total_ce_oi or 1
        pc_ratio = total_pe_oi / denom if total_ce_oi > 0 else 1.0

        # IV percentile from NSE strike-level IV
        iv_vals_pct = [
            v for v in iv_vals if v > 0 and v < 500
        ]  # Filter outliers (< 500%)
        median_iv = (
            float(np.percentile(iv_vals_pct, 50)) if iv_vals_pct else float("nan")
        )

        return {
            "pc_ratio": round(pc_ratio, 4),
            "median_iv": round(median_iv, 2) if not np.isnan(median_iv) else None,
            "total_calls": total_ce_vol,
            "total_puts": total_pe_vol,
        }
    except Exception as exc:
        logger.debug(
            "nse_iv_sentiment_error", extra={"symbol": symbol, "error": str(exc)}
        )
        return None


def fetch_iv_sentiment(ticker: str, market: str = "us") -> Optional[dict]:
    """Dispatch IV sentiment to the appropriate source."""
    if market == "india":
        # Strip .NS suffix for jugaad_data
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        return fetch_iv_sentiment_nse(symbol)
    return fetch_iv_sentiment_yf(ticker)


# ── Price / volume data ─────────────────────────────────────────────────


def fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
    fetcher: Optional[YFinancePriceFetcher] = None,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV bars for *tickers* from *start* to *end*."""
    if fetcher is None:
        fetcher = YFinancePriceFetcher(auto_adjust=True)

    all_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = fetcher.fetch(batch, start, end)
        all_data.update(data)
        for k in list(data.keys()):
            if data[k].empty:
                del data[k]
    return all_data
