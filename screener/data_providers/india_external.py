"""External signal data for Indian markets (NSE/BSE).

India-specific sources:
- NSE Option Chain (unusual call/put activity)
- FII/DII flow data (foreign/domestic institutional flow)
- Delivery percentage (already in screener — but can be trended)
- SEBI Insider trading disclosures
- Promoter pledge %
- Bulk/Block deals
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

NSE_BASE = "https://www.nseindia.com"
FMP_BASE = "https://financialmodelingprep.com/api"
FMP_KEY = os.environ.get("FMP_API_KEY", "")

# NSE requires session cookies — use a simple scraper
_NSE_SESSION: requests.Session | None = None


def _nse_session() -> requests.Session:
    global _NSE_SESSION
    if _NSE_SESSION is None:
        s = requests.Session()
        # NSE requires visiting homepage first to get cookies
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        })
        s.get("https://www.nseindia.com", timeout=10)
        _NSE_SESSION = s
    return _NSE_SESSION


def get_nse_option_chain(symbol: str) -> dict[str, Any]:
    """Fetch NSE option chain for a symbol.

    Returns call/put OI, volume, change in OI.
    """
    sym = symbol.replace(".NS", "").upper()
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={sym}"
    try:
        r = _nse_session().get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def compute_options_signal_nse(option_data: dict) -> dict[str, float]:
    """Compute options-based features from NSE option chain.

    Features:
        call_put_oi_ratio: call OI / put OI (>1 = bullish bias)
        call_put_volume_ratio: call volume / put volume
        pcr: put/call ratio (low = bullish)
        max_pain: strike with minimum total OI loss
    """
    if "error" in option_data or "records" not in option_data:
        return {
            "call_put_oi_ratio": 1.0,
            "call_put_volume_ratio": 1.0,
            "pcr": 1.0,
            "max_pain_strike": 0.0,
        }

    records = option_data["records"]["data"]
    if not records:
        return {
            "call_put_oi_ratio": 1.0,
            "call_put_volume_ratio": 1.0,
            "pcr": 1.0,
            "max_pain_strike": 0.0,
        }

    total_call_oi = sum(
        r["CE"]["openInterest"] for r in records if "CE" in r and r["CE"]
    )
    total_put_oi = sum(
        r["PE"]["openInterest"] for r in records if "PE" in r and r["PE"]
    )
    total_call_vol = sum(
        r["CE"]["totalTradedVolume"] for r in records if "CE" in r and r["CE"]
    )
    total_put_vol = sum(
        r["PE"]["totalTradedVolume"] for r in records if "PE" in r and r["PE"]
    )

    call_put_oi = total_call_oi / total_put_oi if total_put_oi > 0 else 1.0
    call_put_vol = total_call_vol / total_put_vol if total_put_vol > 0 else 1.0
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    # Max pain: strike with minimum (CE OI + PE OI)
    strike_pain = {}
    for r in records:
        strike = r.get("strikePrice", 0)
        ce_oi = r.get("CE", {}).get("openInterest", 0) if r.get("CE") else 0
        pe_oi = r.get("PE", {}).get("openInterest", 0) if r.get("PE") else 0
        strike_pain[strike] = ce_oi + pe_oi

    max_pain = min(strike_pain, key=strike_pain.get) if strike_pain else 0.0

    return {
        "call_put_oi_ratio": call_put_oi,
        "call_put_volume_ratio": call_put_vol,
        "pcr": pcr,
        "max_pain_strike": float(max_pain),
    }


def get_fii_dii_data() -> pd.DataFrame:
    """Fetch FII/DII net flow data from NSE.

    Returns daily net buy/sell in crores.
    """
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    try:
        r = _nse_session().get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_fii_signal() -> dict[str, float]:
    """Compute FII/DII flow features.

    Returns:
        fii_5d_net: FII net buy last 5 days (crores)
        fii_trend: FII net buy / 20d avg
        dii_5d_net: DII net buy last 5 days
    """
    df = get_fii_dii_data()
    if df.empty:
        return {"fii_5d_net": 0.0, "fii_trend": 1.0, "dii_5d_net": 0.0}

    df = df.sort_values("date", ascending=False)
    fii_5d = df[df["category"] == "FII"].head(5)["netValue"].sum() if "FII" in df["category"].values else 0.0
    fii_20d = df[df["category"] == "FII"].head(20)["netValue"].mean() if "FII" in df["category"].values else 1.0
    dii_5d = df[df["category"] == "DII"].head(5)["netValue"].sum() if "DII" in df["category"].values else 0.0

    return {
        "fii_5d_net": float(fii_5d),
        "fii_trend": float(fii_5d / fii_20d) if fii_20d else 1.0,
        "dii_5d_net": float(dii_5d),
    }


def get_delivery_data_nse(symbol: str) -> pd.DataFrame:
    """Fetch delivery percentage history from NSE.

    Already used in screener — but here for trend computation.
    """
    sym = symbol.replace(".NS", "").upper()
    url = f"https://www.nseindia.com/api/historical/securityArchives?symbol={sym}&dataType=priceVolumeDeliverable&series=EQ"
    try:
        r = _nse_session().get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_delivery_signal(symbol: str) -> dict[str, float]:
    """Compute delivery percentage trend features.

    Returns:
        delivery_pct_last: latest delivery %
        delivery_trend: current / 20d avg
        delivery_spike: (current - 20d avg) / 20d std
    """
    df = get_delivery_data_nse(symbol)
    if df.empty or "delivery" not in df.columns:
        return {"delivery_pct_last": 50.0, "delivery_trend": 1.0, "delivery_spike": 0.0}

    df = df.sort_values("date", ascending=False)
    latest = pd.to_numeric(df["delivery"], errors="coerce").iloc[0] if "delivery" in df.columns else 50.0
    recent = pd.to_numeric(df["delivery"], errors="coerce").head(20)

    avg = recent.mean() if len(recent) > 0 else 50.0
    std = recent.std() if len(recent) > 1 else 10.0

    return {
        "delivery_pct_last": float(latest),
        "delivery_trend": float(latest / avg) if avg > 0 else 1.0,
        "delivery_spike": float((latest - avg) / std) if std > 0 else 0.0,
    }


def get_promoter_pledge_fmp(ticker: str) -> dict[str, float]:
    """Fetch promoter pledge data via FMP (India companies supported).

    Returns:
        pledge_pct: % of shares pledged by promoters
    """
    # FMP uses .NS suffix for NSE
    symbol = ticker if ".NS" in ticker else f"{ticker}.NS"
    url = f"{FMP_BASE}/v4/insider-ownership?symbol={symbol}&apikey={FMP_KEY}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            # Promoter pledge is often in 'shares' or 'ownershipPercent'
            pledge = data[0].get("ownershipPercent", 0.0)
            return {"pledge_pct": float(pledge)}
    except Exception:
        pass
    return {"pledge_pct": 0.0}


def get_india_external_features(symbol: str) -> dict[str, float]:
    """Fetch all India-specific external features.

    Usage:
        features = get_india_external_features("RELIANCE.NS")
    """
    # Options
    opt_data = get_nse_option_chain(symbol)
    opt_features = compute_options_signal_nse(opt_data)

    # Delivery
    del_features = compute_delivery_signal(symbol)

    # FII/DII (market-wide, not ticker-specific)
    fii_features = compute_fii_signal()

    # Promoter pledge
    pledge = get_promoter_pledge_fmp(symbol)

    return {
        **opt_features,
        **del_features,
        **fii_features,
        **pledge,
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(f"Fetching India external features for {ticker}...")
    features = get_india_external_features(ticker)
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
