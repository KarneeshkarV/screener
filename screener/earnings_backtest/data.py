"""Data acquisition for earnings backtest.

Fetches earnings dates, price bars, volume, analyst recommendations,
and options data from yfinance. Designed for batch processing under
tight RAM constraints (~2 GB).
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

# ── Universe loaders ────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".screener" / "earnings_backtest"


def load_sp500() -> list[str]:
    """Return current S&P 500 ticker list."""
    from screener.universes import load_current_universe

    univ = load_current_universe("sp500")
    # Convert to yfinance-style; no suffix needed for US tickers
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
    # Add .NS suffix for yfinance
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


# ── Earnings dates ──────────────────────────────────────────────────────


def fetch_earnings_dates(
    ticker: str,
    years: int = 3,
) -> Optional[pd.DataFrame]:
    """Return yfinance earnings_dates for *ticker* covering the last *years*.

    Returns DataFrame with columns: [Earnings Date, EPS Estimate, Reported EPS, Surprise(%)]
    or None if unavailable.
    """
    _configure_yfinance()
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return None
        # yfinance returns earnings_dates with tz-aware index
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
        # Filter to last N years
        cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
        ed = ed[ed.index >= cutoff]
        return ed if not ed.empty else None
    except Exception as exc:
        logger.debug(
            "earnings_dates_failed", extra={"ticker": ticker, "error": str(exc)}
        )
        return None


# ── Batch earnings collector ────────────────────────────────────────────


def collect_earnings_events(
    tickers: list[str],
    years: int = 3,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Collect earnings dates for all *tickers*.

    Returns a DataFrame with columns:
        ticker, earnings_date, eps_estimate, reported_eps, surprise_pct

    Processes in *batch_size* chunks to limit API pressure.
    """
    rows: list[dict] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(
            "earnings_batch",
            extra={"batch": f"{i}-{i + len(batch)}", "size": len(batch)},
        )
        for ticker in batch:
            try:
                ed = fetch_earnings_dates(ticker, years=years)
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


def fetch_analyst_sentiment(ticker: str) -> Optional[dict]:
    """Compute analyst sentiment from yfinance upgrades/downgrades.

    Action key from yfinance:
      - "up"    → upgrade (bullish)
      - "down"  → downgrade (bearish)
      - "reit"  → reiterate/maintain (neutral-bullish, weight=0.5)
      - "main"  → main/hold (neutral)
      - "init"  → initiate (neutral)

    Returns dict with keys: upgrades, downgrades, net, grade_counts, or None.
    """
    try:
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return None

        if "Action" in ud.columns:
            counts = ud["Action"].value_counts().to_dict()
            # Strict: up is upgrade, down is downgrade, reit is half-upgrade
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


def fetch_iv_sentiment(ticker: str) -> Optional[dict]:
    """Compute put/call ratio and IV percentile for *ticker*.

    P/C ratio < 0.7 is considered bullish.
    Returns dict with keys: pc_ratio, iv_percentile, total_calls, total_puts,
    or None if no options data (e.g. India).
    """
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return None

        # Use nearest expiry that's >= 5 days out (avoid noise from expiry-day effects)
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

        # P/C ratio on volume; fall back to OI if volume is zero
        denom = total_calls or 1
        pc_ratio = (
            total_puts / denom
            if total_calls > 0
            else (total_oi_puts / (total_oi_calls or 1))
        )

        # IV percentile: use mean IV of calls as proxy
        iv_vals = []
        if "impliedVolatility" in calls.columns:
            iv_vals.extend(calls["impliedVolatility"].dropna().tolist())
        if "impliedVolatility" in puts.columns:
            iv_vals.extend(puts["impliedVolatility"].dropna().tolist())
        iv_percentile = float(np.percentile(iv_vals, 50)) if iv_vals else float("nan")

        return {
            "pc_ratio": round(pc_ratio, 4),
            "iv_percentile": round(iv_percentile, 4),
            "total_calls": total_calls,
            "total_puts": total_puts,
        }
    except Exception as exc:
        logger.debug("iv_sentiment_error", extra={"ticker": ticker, "error": str(exc)})
        return None


# ── Price / volume data ─────────────────────────────────────────────────


def fetch_price_data(
    tickers: list[str],
    start: date,
    end: date,
    fetcher: Optional[YFinancePriceFetcher] = None,
    batch_size: int = 50,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV bars for *tickers* from *start* to *end*.

    Processes in *batch_size* chunks and frees memory between batches.
    """
    if fetcher is None:
        fetcher = YFinancePriceFetcher(auto_adjust=True)

    all_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        data = fetcher.fetch(batch, start, end)
        all_data.update(data)
        # Free memory from stale frames
        for k in list(data.keys()):
            if data[k].empty:
                del data[k]
    return all_data
