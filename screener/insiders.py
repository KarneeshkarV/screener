"""Detect promoter/insider holding increases.

TradingView's screener API does not expose promoter/insider holding fields, so
we scan a liquid universe with TradingView, then enrich each ticker from two
complementary sources:

* yfinance ``Ticker.insider_purchases`` — 6-month aggregate of insider buy/sell
  transactions. Available for both US tickers and ``.NS`` Indian listings.
  Positive net shares ⇒ insiders bought more than they sold.

* openscreener (screener.in) — quarterly shareholding pattern (``promoters``,
  ``fiis``, ``diis``). Indian-only. We compute the latest-quarter delta in
  promoter % vs. the previous quarter; positive ⇒ promoter holding increased.

For India we use openscreener as the primary signal (this is the canonical
source for "promoter holding") and yfinance as a secondary cross-check.
For US the yfinance feed is the only signal.
"""
from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import yfinance as yf


_INDIA_SUFFIXES = (".NS", ".BO")
_SCREENER_URL = "https://www.screener.in/company/{symbol}/"
_SCREENER_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; screener-cli/1.0)"}


def _tv_to_yf(ticker: str, market: str) -> str:
    symbol = ticker.split(":", 1)[1] if ":" in ticker else ticker
    if market == "india" and not symbol.endswith(_INDIA_SUFFIXES):
        return f"{symbol}.NS"
    return symbol


# ── yfinance insider purchases ─────────────────────────────────────────────


def _row_value(df: pd.DataFrame, label: str, column: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    match = df[df["Insider Purchases Last 6m"] == label]
    if match.empty:
        return None
    val = match.iloc[0].get(column)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fetch_yf_one(name: str, yf_symbol: str) -> Optional[dict]:
    try:
        purchases = yf.Ticker(yf_symbol).insider_purchases
    except Exception:
        return None
    if purchases is None or purchases.empty:
        return None

    return {
        "name": name,
        "yf_symbol": yf_symbol,
        "yf_net_shares_6m": _row_value(purchases, "Net Shares Purchased (Sold)", "Shares"),
        "yf_net_pct_6m": _row_value(purchases, "% Net Shares Purchased (Sold)", "Shares"),
        "yf_total_held": _row_value(purchases, "Total Insider Shares Held", "Shares"),
        "yf_buy_trans_6m": _row_value(purchases, "Purchases", "Trans"),
        "yf_sell_trans_6m": _row_value(purchases, "Sales", "Trans"),
    }


def fetch_yfinance_insiders(
    universe: pd.DataFrame,
    market: str,
    max_workers: int = 12,
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()

    jobs = [
        (str(row["name"]), _tv_to_yf(str(row["ticker"]), market))
        for _, row in universe.iterrows()
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_yf_one, n, s) for n, s in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


# ── openscreener (screener.in) quarterly promoter % ────────────────────────


class _HttpScraper:
    """Minimal scraper compatible with openscreener's Stock(scraper=...) interface.

    Skips Playwright (the bundled scraper hangs on screener.in's networkidle
    state because of long-tail analytics requests). screener.in returns the
    full shareholding section in the initial HTML response, so a plain HTTP
    GET is sufficient.
    """

    base_url = _SCREENER_URL
    consolidated = False

    def fetch_page(self, symbol: str) -> str:
        req = urllib.request.Request(
            self.base_url.format(symbol=symbol.upper()), headers=_SCREENER_HEADERS
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", "ignore")

    def fetch_pages(self, symbols):
        return {s.upper(): self.fetch_page(s) for s in symbols}


def _fetch_openscreener_one(name: str) -> Optional[dict]:
    try:
        from openscreener import Stock
    except ImportError:
        return None
    try:
        rows = Stock(name, scraper=_HttpScraper()).shareholding_quarterly()
    except Exception:
        return None
    if not rows or len(rows) < 2:
        return None

    latest, prev = rows[-1], rows[-2]
    p_latest = latest.get("promoters")
    p_prev = prev.get("promoters")
    if p_latest is None or p_prev is None:
        return None
    try:
        change = float(p_latest) - float(p_prev)
    except (TypeError, ValueError):
        return None

    return {
        "name": name,
        "promoter_pct_latest": float(p_latest),
        "promoter_pct_prev": float(p_prev),
        "promoter_change": change,
        "latest_quarter": latest.get("date"),
        "fii_pct_latest": latest.get("fiis"),
        "dii_pct_latest": latest.get("diis"),
    }


def fetch_openscreener_promoters(
    universe: pd.DataFrame,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Fetch quarterly promoter % from screener.in for each Indian ticker."""
    if universe.empty:
        return pd.DataFrame()

    names = universe["name"].astype(str).tolist()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_openscreener_one, n) for n in names]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


# ── filtering ──────────────────────────────────────────────────────────────


def filter_promoter_increased(
    insiders: pd.DataFrame,
    market: str,
    min_promoter_change_pct: float = 0.0,
    min_yf_net_pct: Optional[float] = None,
    require_both: bool = False,
) -> pd.DataFrame:
    """Keep tickers where holdings increased.

    For India: require ``promoter_change >= min_promoter_change_pct`` from
    openscreener. If ``require_both`` is set, also require positive yfinance
    net buys.

    For US: require ``yf_net_shares_6m > 0`` (and optional ``min_yf_net_pct``).
    """
    if insiders.empty:
        return insiders

    if market == "india":
        change = pd.to_numeric(insiders.get("promoter_change"), errors="coerce")
        mask = change > min_promoter_change_pct
        if require_both:
            yf_net = pd.to_numeric(insiders.get("yf_net_shares_6m"), errors="coerce")
            mask = mask & (yf_net > 0)
    else:
        net = pd.to_numeric(insiders.get("yf_net_shares_6m"), errors="coerce")
        mask = net > 0
        if min_yf_net_pct is not None:
            pct = pd.to_numeric(insiders.get("yf_net_pct_6m"), errors="coerce")
            mask = mask & (pct >= min_yf_net_pct)

    return insiders[mask.fillna(False)].copy()
