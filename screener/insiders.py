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

import json
import logging
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import yfinance as yf

from screener.providers import CachedProvider, ProviderSpec
from screener.resilience import call_with_resilience


logger = logging.getLogger(__name__)
_INDIA_SUFFIXES = (".NS", ".BO")
_SCREENER_URL = "https://www.screener.in/company/{symbol}/"
_SCREENER_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; screener-cli/1.0)"}

_FMP_INSIDER_URL = "https://financialmodelingprep.com/api/v4/insider-trading"
# FMP only exposes Form 3/4/5 (SEC) data, which covers US listings. Indian
# tickers stay on the screener.in promoter feed.
_FMP_WINDOW_DAYS = 182
_FMP_MAX_PAGES = 10

# Provider seams (cache + resilience). Default TTLs match the legacy
# hand-wired call sites; ``cache_ttl`` is overridden per-call to honour CLI
# flags. The yfinance feed shares the "yfinance" breaker, FMP the "fmp"
# breaker, and the screener.in scrape the "screener-in" breaker.
_YF_INSIDER_PROVIDER = CachedProvider(
    ProviderSpec(provider="yfinance", namespace="yfinance_insiders", ttl_seconds=86400)
)
_FMP_INSIDER_PROVIDER = CachedProvider(
    ProviderSpec(provider="fmp", namespace="fmp_insiders", ttl_seconds=86400)
)
_OPENSCREENER_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="screener-in",
        namespace="openscreener_promoters",
        ttl_seconds=7 * 86400,
    )
)


def _tv_to_yf(ticker: str, market: str) -> str:
    symbol = ticker.split(":", 1)[1] if ":" in ticker else ticker
    if market == "india" and not symbol.endswith(_INDIA_SUFFIXES):
        return f"{symbol}.NS"
    return symbol


# ── yfinance insider purchases ─────────────────────────────────────────────


def _row_value(df: pd.DataFrame, label: str, column: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "Insider Purchases Last 6m" not in df.columns:
        logger.debug(
            "yfinance insider purchases payload missing label column; columns=%s",
            list(df.columns),
        )
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


def _fetch_yf_one(
    name: str,
    yf_symbol: str,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> Optional[dict]:
    def _fetch() -> Optional[dict]:
        purchases = yf.Ticker(yf_symbol).insider_purchases
        if purchases is None or purchases.empty:
            return None

        return {
            "name": name,
            "yf_symbol": yf_symbol,
            "yf_net_shares_6m": _row_value(
                purchases, "Net Shares Purchased (Sold)", "Shares"
            ),
            "yf_net_pct_6m": _row_value(
                purchases, "% Net Shares Purchased (Sold)", "Shares"
            ),
            "yf_total_held": _row_value(
                purchases, "Total Insider Shares Held", "Shares"
            ),
            "yf_buy_trans_6m": _row_value(purchases, "Purchases", "Trans"),
            "yf_sell_trans_6m": _row_value(purchases, "Sales", "Trans"),
        }

    return _YF_INSIDER_PROVIDER.fetch(
        ("insider_purchases", name, yf_symbol),
        _fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"insider purchases {yf_symbol}",
    )


def fetch_yfinance_insiders(
    universe: pd.DataFrame,
    market: str,
    max_workers: int = 12,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()

    jobs = [
        (str(row["name"]), _tv_to_yf(str(row["ticker"]), market))
        for _, row in universe.iterrows()
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_fetch_yf_one, n, s, cache_ttl=cache_ttl, refresh=refresh)
            for n, s in jobs
        ]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


# ── FMP insider trading (SEC Form 4) ───────────────────────────────────────


def _fmp_api_key() -> Optional[str]:
    """Resolve FMP_API_KEY, loading the project .env once like the backtester."""
    key = os.environ.get("FMP_API_KEY")
    if key:
        return key
    try:
        from screener.backtester.data import load_env_file
    except Exception:
        return None
    load_env_file()
    return os.environ.get("FMP_API_KEY")


def _aggregate_fmp_transactions(
    transactions: list[dict], window_days: int = _FMP_WINDOW_DAYS
) -> Optional[dict]:
    """Aggregate FMP insider rows into 6-month net buy/sell share counts.

    On SEC Form 4 the ``acquistionOrDisposition`` flag (FMP's spelling) only
    says whether shares were acquired (``A``) or disposed (``D``). An ``A`` row
    covers open-market purchases *and* grants, awards, option-exercises and
    gifts, so it cannot stand alone as a "buy" signal. We therefore key off
    the more specific ``transactionType`` SEC code:

    * a genuine **buy** is ``transactionType`` starting with ``P-`` (Purchase)
      *and* ``acquistionOrDisposition == "A"``;
    * a genuine **sell** is ``transactionType`` starting with ``S-`` (Sale)
      *and* ``acquistionOrDisposition == "D"``.

    Every other code (``A-Award``, ``G-Gift``, ``M-Exempt``,
    ``F-Payment of Exercise`` …) is skipped. Net shares > 0 ⇒ insiders bought
    more than they sold over the window. ``transactionType`` may be absent or
    ``None``; it is coerced to ``str`` (default ``""``) so a missing code just
    skips the row instead of raising.
    """
    if not transactions:
        return None
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=window_days)
    bought = sold = 0.0
    buy_trans = sell_trans = 0
    for txn in transactions:
        date_raw = txn.get("transactionDate") or txn.get("filingDate")
        ts = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(ts) or ts < cutoff:
            continue
        try:
            shares = float(txn.get("securitiesTransacted") or 0.0)
        except (TypeError, ValueError):
            continue
        disposition = txn.get("acquistionOrDisposition")
        txn_type = str(txn.get("transactionType") or "").upper()
        if txn_type.startswith("P-") and disposition == "A":
            bought += shares
            buy_trans += 1
        elif txn_type.startswith("S-") and disposition == "D":
            sold += shares
            sell_trans += 1
    if buy_trans == 0 and sell_trans == 0:
        return None
    return {
        "fmp_net_shares_6m": bought - sold,
        "fmp_buy_shares_6m": bought,
        "fmp_sell_shares_6m": sold,
        "fmp_buy_trans_6m": buy_trans,
        "fmp_sell_trans_6m": sell_trans,
    }


def _fetch_fmp_insider_one(
    name: str,
    symbol: str,
    *,
    api_key: str,
    cache_ttl: float | None,
    refresh: bool,
) -> Optional[dict]:
    def _fetch() -> Optional[dict]:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=_FMP_WINDOW_DAYS)
        truncated = False

        def _request_page(page: int) -> Optional[list]:
            query = urllib.parse.urlencode(
                {"symbol": symbol, "page": page, "apikey": api_key}
            )
            req = urllib.request.Request(
                f"{_FMP_INSIDER_URL}?{query}", headers=_SCREENER_HEADERS
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", "ignore"))
            return payload if isinstance(payload, list) else None

        # FMP paginates insider rows (newest first). Walk pages until one
        # is empty/non-list, the oldest row on a page predates the 182-day
        # window, or we hit a safety cap (no unbounded loop on bad data).
        collected: list[dict] = []
        expected_page_size: int | None = None
        for page in range(_FMP_MAX_PAGES):
            rows = _request_page(page)
            if not rows:
                break
            if expected_page_size is None:
                expected_page_size = len(rows)
            collected.extend(rows)
            oldest_raw = rows[-1].get("transactionDate") or rows[-1].get("filingDate")
            oldest = pd.to_datetime(oldest_raw, errors="coerce")
            if not pd.isna(oldest) and oldest < cutoff:
                break
            if (
                page == _FMP_MAX_PAGES - 1
                and expected_page_size is not None
                and len(rows) >= expected_page_size
                and not pd.isna(oldest)
                and oldest >= cutoff
            ):
                truncated = True
                logger.warning(
                    "FMP insider trading for %s may be truncated at %d pages",
                    symbol,
                    _FMP_MAX_PAGES,
                )

        transactions = collected or None
        if not transactions:
            return None
        agg = _aggregate_fmp_transactions(transactions)
        if agg is None:
            return None
        # Surface page-cap truncation to callers (not just a log line): the
        # 6m totals may be incomplete when the history was cut at the cap.
        return {"name": name, "fmp_symbol": symbol, "fmp_truncated": truncated, **agg}

    return _FMP_INSIDER_PROVIDER.fetch(
        ("insider_trading", name, symbol),
        _fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"insider trading {symbol}",
    )


def fetch_fmp_insiders(
    universe: pd.DataFrame,
    market: str,
    max_workers: int = 12,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch 6-month net insider buying from FMP for each US ticker.

    Returns an empty frame when no ``FMP_API_KEY`` is configured, so callers
    can fall back to the yfinance feed transparently.
    """
    if universe.empty:
        return pd.DataFrame()
    api_key = _fmp_api_key()
    if not api_key:
        return pd.DataFrame()

    jobs = [
        (str(row["name"]), _tv_to_yf(str(row["ticker"]), market))
        for _, row in universe.iterrows()
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _fetch_fmp_insider_one,
                n,
                s,
                api_key=api_key,
                cache_ttl=cache_ttl,
                refresh=refresh,
            )
            for n, s in jobs
        ]
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
        def _fetch() -> str:
            req = urllib.request.Request(
                self.base_url.format(symbol=symbol.upper()), headers=_SCREENER_HEADERS
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", "ignore")

        return call_with_resilience(
            "screener-in",
            f"company page {symbol}",
            _fetch,
            fallback="",
        )

    def fetch_pages(self, symbols):
        return {s.upper(): self.fetch_page(s) for s in symbols}


def _fetch_openscreener_one(
    name: str,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> Optional[dict]:
    def _fetch() -> Optional[dict]:
        try:
            from openscreener import Stock
        except ImportError:
            return None
        rows = Stock(name, scraper=_HttpScraper()).shareholding_quarterly()
        if not rows:
            return None
        if len(rows) < 2:
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

    return _OPENSCREENER_PROVIDER.fetch(
        ("shareholding_quarterly", name),
        _fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"shareholding {name}",
    )


def fetch_openscreener_promoters(
    universe: pd.DataFrame,
    max_workers: int = 6,
    cache_ttl: float | None = 7 * 86400,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch quarterly promoter % from screener.in for each Indian ticker."""
    if universe.empty:
        return pd.DataFrame()

    names = universe["name"].astype(str).tolist()
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _fetch_openscreener_one, n, cache_ttl=cache_ttl, refresh=refresh
            )
            for n in names
        ]
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
        # US: FMP (SEC Form 4) is the primary signal when available; fall back
        # to the yfinance feed per-row when FMP has no data for a ticker.
        yf_net = pd.to_numeric(insiders.get("yf_net_shares_6m"), errors="coerce")
        if "fmp_net_shares_6m" in insiders.columns:
            fmp_net = pd.to_numeric(insiders.get("fmp_net_shares_6m"), errors="coerce")
            net = fmp_net.where(fmp_net.notna() & (fmp_net != 0.0), yf_net)
        else:
            net = yf_net
        mask = net > 0
        if min_yf_net_pct is not None:
            pct = pd.to_numeric(insiders.get("yf_net_pct_6m"), errors="coerce")
            mask = mask & (pct >= min_yf_net_pct)

    return insiders[mask.fillna(False)].copy()
