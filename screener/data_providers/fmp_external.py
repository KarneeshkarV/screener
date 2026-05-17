"""Fetch external signal data from Financial Modeling Prep (FMP) free tier.

Sources:
- Insider trading (Form 4)
- Short interest
- Earnings surprises
- Institutional ownership changes

Free tier: 250 requests/day. Sign up: https://financialmodelingprep.com/developer/docs/
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

FMP_BASE = "https://financialmodelingprep.com/api"
FMP_KEY = os.environ.get("FMP_API_KEY", "")


def _get(endpoint: str, params: dict | None = None) -> dict | list:
    """Make authenticated FMP GET request."""
    url = f"{FMP_BASE}/{endpoint}"
    p = params or {}
    p["apikey"] = FMP_KEY
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_insider_trading(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Fetch SEC Form 4 insider transactions.

    Returns DataFrame with columns:
        reportingName, transactionDate, securityName,
        transactionCode (P=Buy, S=Sell),
        securitiesTransacted (shares), price (avg price)
    """
    data = _get(f"v4/insider-trading", {"symbol": ticker, "limit": limit})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "transactionDate" in df.columns:
        df["transactionDate"] = pd.to_datetime(df["transactionDate"])
    # P = Purchase, S = Sale
    df["is_buy"] = df["transactionCode"].astype(str).str.upper() == "P"
    return df


def compute_insider_score(ticker: str, lookback_days: int = 30) -> dict[str, float]:
    """Compute insider signal features.

    Returns:
        insider_buy_ratio: net buy transactions / total transactions
        insider_buy_shares_ratio: buy shares / (buy + sell shares)
        insider_buy_dollar_ratio: buy $ / (buy + sell $)
    """
    df = get_insider_trading(ticker, limit=200)
    if df.empty:
        return {
            "insider_buy_ratio": 0.5,
            "insider_buy_shares_ratio": 0.5,
            "insider_buy_dollar_ratio": 0.5,
            "insider_n_transactions": 0,
        }

    cutoff = datetime.now() - timedelta(days=lookback_days)
    recent = df[df["transactionDate"] >= cutoff]
    if recent.empty:
        recent = df  # fallback to all

    buys = recent[recent["is_buy"]]
    sells = recent[~recent["is_buy"]]

    n_buys = len(buys)
    n_sells = len(sells)
    total = n_buys + n_sells

    buy_shares = buys["securitiesTransacted"].astype(float).sum() if n_buys else 0.0
    sell_shares = sells["securitiesTransacted"].astype(float).sum() if n_sells else 0.0

    buy_dollars = (buys["securitiesTransacted"].astype(float) * buys["price"].astype(float)).sum() if n_buys else 0.0
    sell_dollars = (sells["securitiesTransacted"].astype(float) * sells["price"].astype(float)).sum() if n_sells else 0.0

    return {
        "insider_buy_ratio": n_buys / total if total > 0 else 0.5,
        "insider_buy_shares_ratio": buy_shares / (buy_shares + sell_shares) if (buy_shares + sell_shares) > 0 else 0.5,
        "insider_buy_dollar_ratio": buy_dollars / (buy_dollars + sell_dollars) if (buy_dollars + sell_dollars) > 0 else 0.5,
        "insider_n_transactions": total,
    }


def get_short_interest(ticker: str) -> pd.DataFrame:
    """Fetch short interest history."""
    data = _get(f"v3/historical-shares-float/{ticker}")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    for col in ["shortInt", "floatShares", "outstandingShares"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_short_signal(ticker: str) -> dict[str, float]:
    """Compute short-interest-based features.

    Returns:
        short_pct_float: short interest / float
        short_trend: current / 30d avg (decreasing short = bullish)
    """
    df = get_short_interest(ticker)
    if df.empty or "shortInt" not in df.columns:
        return {"short_pct_float": 0.05, "short_trend": 1.0}

    df = df.sort_values("date", ascending=False)
    latest = df.iloc[0]
    float_shares = latest.get("floatShares", 0)
    short_int = latest.get("shortInt", 0)

    pct = short_int / float_shares if float_shares and float_shares > 0 else 0.05

    # 30-day trend
    recent = df.head(30)
    avg_short = recent["shortInt"].mean() if len(recent) > 0 else short_int
    trend = short_int / avg_short if avg_short and avg_short > 0 else 1.0

    return {"short_pct_float": pct, "short_trend": trend}


def get_earnings_surprises(ticker: str, limit: int = 20) -> pd.DataFrame:
    """Fetch earnings surprise history."""
    data = _get(f"v3/earnings-surprises/{ticker}")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    for col in ["actualEarningResult", "estimatedEarning"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_earnings_signal(ticker: str) -> dict[str, float]:
    """Compute earnings-based features.

    Returns:
        earnings_surprise_last: (actual - estimate) / |estimate|
        earnings_beat_streak: consecutive beats
        days_since_earnings: days from last report
    """
    df = get_earnings_surprises(ticker)
    if df.empty:
        return {
            "earnings_surprise_last": 0.0,
            "earnings_beat_streak": 0.0,
            "days_since_earnings": 90.0,
        }

    df = df.sort_values("date", ascending=False)
    latest = df.iloc[0]
    actual = latest.get("actualEarningResult", 0)
    estimate = latest.get("estimatedEarning", 0)
    surprise = (actual - estimate) / abs(estimate) if estimate and estimate != 0 else 0.0

    # Beat streak
    df["beat"] = df["actualEarningResult"] > df["estimatedEarning"]
    streak = 0
    for b in df["beat"]:
        if b:
            streak += 1
        else:
            break

    days_since = (datetime.now() - df.iloc[0]["date"]).days if "date" in df.columns else 90

    return {
        "earnings_surprise_last": surprise,
        "earnings_beat_streak": float(streak),
        "days_since_earnings": float(days_since),
    }


def get_external_features(ticker: str) -> dict[str, float]:
    """Fetch all external features for a ticker in one call.

    Usage:
        features = get_external_features("AAPL")
        # Returns ~10 features ready for ML model
    """
    insider = compute_insider_score(ticker)
    short_sig = compute_short_signal(ticker)
    earnings = compute_earnings_signal(ticker)

    return {
        **insider,
        **short_sig,
        **earnings,
    }


# ---------------------------------------------------------------------------
# Example / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if not FMP_API_KEY:
        print("ERROR: Set FMP_API_KEY environment variable.")
        print("Get free key: https://financialmodelingprep.com/developer/docs/")
        sys.exit(1)

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching external features for {ticker}...")
    features = get_external_features(ticker)
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
