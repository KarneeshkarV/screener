"""Data acquisition for earnings backtest.

Fetches earnings dates, price bars, volume, analyst recommendations,
and options data. Uses yfinance for US and jugaad_data for India (NSE).

Designed for batch processing under tight RAM constraints (~2 GB).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import yfinance as yf

from screener.backtester.data import (
    YFinancePriceFetcher,
    _configure_yfinance,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".screener" / "earnings_backtest"
YFINANCE_TIMEOUT_SECONDS = 5
EARNINGS_CACHE_DAYS = 30
SENTIMENT_CACHE_DAYS = 1
MAX_WORKERS = 12
_YFINANCE_TIMEOUT_PATCHED = False

# Indian companies report a fiscal quarter's results ~45-60 days after the
# period-end (e.g. "Mar 2024" results are announced in May 2024). screener.in
# only exposes the fiscal PERIOD-END label, not the announcement date, so a
# period-end keyed event would be applied before it was ever public. We add a
# conservative filing lag to the period-end as a point-in-time floor. The value
# 60 is deliberately the CONSERVATIVE UPPER bound of the 45-60 day window: it is
# chosen so the synthetic point-in-time date never precedes the real
# announcement even for late (day 46-60) filers, common at March/year-end
# results. Using the lower bound (45) would leak EPS for late filers, since a
# backtest as_of between day 46 and the real announcement would trade on results
# that were not yet public. The real NSE announcement date (when available) is
# preferred over this estimate.
INDIA_EARNINGS_FILING_LAG_DAYS = 60


def _install_yfinance_timeout_patch() -> None:
    """Cap yfinance's internal request timeout for scrape/API calls."""
    global _YFINANCE_TIMEOUT_PATCHED
    if _YFINANCE_TIMEOUT_PATCHED:
        return
    try:
        import yfinance.data as yf_data

        original_get = yf_data.YfData.get
        original_cache_get = yf_data.YfData.cache_get

        def capped_get(self, url, params=None, timeout=30):
            timeout = min(
                float(timeout or YFINANCE_TIMEOUT_SECONDS), YFINANCE_TIMEOUT_SECONDS
            )
            return original_get(self, url, params=params, timeout=timeout)

        def capped_cache_get(self, url, params=None, timeout=30):
            timeout = min(
                float(timeout or YFINANCE_TIMEOUT_SECONDS), YFINANCE_TIMEOUT_SECONDS
            )
            return original_cache_get(self, url, params=params, timeout=timeout)

        yf_data.YfData.get = capped_get
        yf_data.YfData.cache_get = capped_cache_get
        _YFINANCE_TIMEOUT_PATCHED = True
    except Exception as exc:
        logger.debug("yfinance_timeout_patch_failed", extra={"error": str(exc)})


def _safe_key(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _json_cache_path(kind: str, key: str) -> Path:
    return CACHE_DIR / kind / f"{_safe_key(key)}.json"


def _read_json_cache(path: Path, max_age_days: int) -> tuple[bool, Any]:
    if not path.exists():
        return False, None
    age = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
    if age > max_age_days:
        return False, None
    try:
        payload = json.loads(path.read_text())
        return True, payload.get("value")
    except Exception as exc:
        logger.debug(
            "json_cache_read_failed", extra={"path": str(path), "error": str(exc)}
        )
        return False, None


def _write_json_cache(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"value": value}, allow_nan=True))
    except Exception as exc:
        logger.debug(
            "json_cache_write_failed", extra={"path": str(path), "error": str(exc)}
        )


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _earnings_to_records(ed: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in ed.iterrows():
        # iterrows() types the index label as Hashable; it is a datetime here.
        ts = pd.Timestamp(cast(Any, idx)).tz_localize(None).normalize()
        records.append(
            {
                "earnings_date": ts.date().isoformat(),
                "eps_estimate": _jsonable(row.get("EPS Estimate", float("nan"))),
                "reported_eps": _jsonable(row.get("Reported EPS", float("nan"))),
                "surprise_pct": _jsonable(row.get("Surprise(%)", float("nan"))),
            }
        )
    return records


def _earnings_from_records(records: list[dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not records:
        return None
    df = pd.DataFrame(records)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.set_index("earnings_date")
    df = df.rename(
        columns={
            "eps_estimate": "EPS Estimate",
            "reported_eps": "Reported EPS",
            "surprise_pct": "Surprise(%)",
        }
    )
    return df


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
    cache_path = _json_cache_path("earnings_yf", f"{ticker}_{years}")
    hit, cached = _read_json_cache(cache_path, EARNINGS_CACHE_DAYS)
    if hit:
        return _earnings_from_records(cached or [])

    _install_yfinance_timeout_patch()
    _configure_yfinance()
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            _write_json_cache(cache_path, [])
            return None
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
        cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
        ed = ed[ed.index >= cutoff]
        _write_json_cache(cache_path, _earnings_to_records(ed))
        return ed if not ed.empty else None
    except Exception as exc:
        logger.debug(
            "earnings_dates_failed", extra={"ticker": ticker, "error": str(exc)}
        )
        _write_json_cache(cache_path, [])
        return None


def fetch_earnings_dates_nse() -> Optional[pd.DataFrame]:
    """Fetch earnings result dates from NSE corporate announcements via jugaad_data."""
    cache_path = _json_cache_path("earnings_nse", "corporate_announcements")
    hit, cached = _read_json_cache(cache_path, SENTIMENT_CACHE_DAYS)
    if hit:
        if not cached:
            return None
        df = pd.DataFrame(cached)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df

    try:
        from jugaad_data.nse import NSELive

        nse = NSELive()
        announcements = nse.corporate_announcements()
        if not announcements:
            _write_json_cache(cache_path, [])
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
            _write_json_cache(cache_path, [])
            return None
        df = pd.DataFrame(rows)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.strftime(
            "%Y-%m-%d"
        )
        _write_json_cache(cache_path, df.to_dict("records"))
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df

    except Exception as exc:
        logger.warning("nse_earnings_fetch_failed", extra={"error": str(exc)})
        _write_json_cache(cache_path, [])
        return None


def _earnings_rows_for_ticker(ticker: str, years: int) -> list[dict[str, Any]]:
    ed = fetch_earnings_dates_yf(ticker, years=years)
    if ed is None:
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in ed.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "earnings_date": idx.date() if hasattr(idx, "date") else idx,
                "eps_estimate": row.get("EPS Estimate", float("nan")),
                "reported_eps": row.get("Reported EPS", float("nan")),
                "surprise_pct": row.get("Surprise(%)", float("nan")),
            }
        )
    return rows


def _fetch_yf_earnings_rows(
    tickers: list[str], years: int, batch_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_workers = min(MAX_WORKERS, max(1, batch_size), max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_earnings_rows_for_ticker, ticker, years): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                logger.debug(
                    "earnings_collect_error",
                    extra={"ticker": ticker, "error": str(exc)},
                )
    return rows


def fetch_earnings_dates_openscreener(
    ticker: str,
    years: int = 3,
    filing_lag_days: int = INDIA_EARNINGS_FILING_LAG_DAYS,
) -> Optional[pd.DataFrame]:
    """Return India quarterly result periods from screener.in via openscreener.

    screener.in keys each row on the fiscal PERIOD-END (e.g. ``"Mar 2024"`` →
    2024-03-31). Indian results are only announced ~45-60 days later, so the
    bare period-end leaks information into the backtest. We add
    ``filing_lag_days`` to the period-end as a point-in-time floor for when the
    result became public. The default is the conservative 60-day upper bound of
    that window, so the floor never precedes the real announcement even for late
    filers. Callers that have the actual NSE announcement date should prefer it
    (see :func:`collect_earnings_events`).
    """
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    cache_path = _json_cache_path(
        "earnings_openscreener", f"{symbol}_{years}_{filing_lag_days}"
    )
    hit, cached = _read_json_cache(cache_path, EARNINGS_CACHE_DAYS)
    if hit and cached:
        return _earnings_from_records(cached or [])

    try:
        from openscreener import Stock
        from screener.insiders import _HttpScraper

        payload = Stock(symbol, scraper=_HttpScraper()).fetch("quarterly_results")
        if not isinstance(payload, dict):
            _write_json_cache(cache_path, [])
            return None
        quarterly = payload.get("quarterly_results")
        if not isinstance(quarterly, list) or not quarterly:
            _write_json_cache(cache_path, [])
            return None

        cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
        records: list[dict[str, Any]] = []
        for item in quarterly:
            if not isinstance(item, dict):
                continue
            label = item.get("date")
            if not label:
                continue
            try:
                period_end = pd.to_datetime(
                    str(label), format="%b %Y"
                ) + pd.offsets.MonthEnd(0)
            except Exception:
                continue
            if period_end < cutoff:
                continue
            # Apply the filing lag: the result is not public until up to ~60 days
            # after the fiscal period-end. Use that as the (estimated) event
            # date so the backtest never acts on it before it was announced.
            announce_date = period_end + pd.Timedelta(days=filing_lag_days)
            records.append(
                {
                    "earnings_date": announce_date.date().isoformat(),
                    "period_end": period_end.date().isoformat(),
                    "eps_estimate": None,
                    "reported_eps": _jsonable(item.get("eps")),
                    "surprise_pct": None,
                }
            )

        _write_json_cache(cache_path, records)
        return _earnings_from_records(records)
    except Exception as exc:
        logger.debug(
            "openscreener_earnings_failed",
            extra={"ticker": ticker, "error": str(exc)},
        )
        return None


def _openscreener_earnings_rows_for_ticker(
    ticker: str, years: int
) -> list[dict[str, Any]]:
    ed = fetch_earnings_dates_openscreener(ticker, years=years)
    if ed is None:
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in ed.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "earnings_date": idx.date() if hasattr(idx, "date") else idx,
                "period_end": row.get("period_end"),
                "eps_estimate": row.get("EPS Estimate", float("nan")),
                "reported_eps": row.get("Reported EPS", float("nan")),
                "surprise_pct": row.get("Surprise(%)", float("nan")),
            }
        )
    return rows


def _fetch_openscreener_earnings_rows(
    tickers: list[str], years: int, batch_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_workers = min(2, max(1, batch_size), max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(
                _openscreener_earnings_rows_for_ticker, ticker, years
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                rows.extend(future.result())
                time.sleep(0.1)
            except Exception as exc:
                logger.debug(
                    "openscreener_earnings_collect_error",
                    extra={"ticker": ticker, "error": str(exc)},
                )
    return rows


# ── Batch earnings collector ────────────────────────────────────────────


def collect_earnings_events(
    tickers: list[str],
    years: int = 3,
    batch_size: int = 50,
    market: str = "us",
) -> pd.DataFrame:
    """Collect earnings dates for all *tickers*.

    For India: uses jugaad_data (NSE announcements) only.
    For US: uses yfinance only.
    """
    rows: list[dict] = []

    if market == "india":
        # NSE-announced (ticker, fiscal-quarter) pairs already covered by a real
        # announcement date, so the openscreener period-end+lag estimate for the
        # same result is not double-counted.
        nse_quarters: set[tuple[str, pd.Period]] = set()

        # Try NSE corporate announcements first (broader coverage). These carry
        # the real announcement (``sort_date``) — already point-in-time.
        nse_events = fetch_earnings_dates_nse()
        if nse_events is not None and not nse_events.empty:
            # Only keep tickers in our universe
            ticker_set = set(tickers)
            filtered = nse_events[nse_events["ticker"].isin(ticker_set)]
            # Convert to unified format
            cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
            filtered = filtered[filtered["earnings_date"] >= cutoff]
            for _, row in filtered.iterrows():
                ann = pd.Timestamp(row["earnings_date"])
                # Map the announcement back to the fiscal quarter it reports on:
                # the quarter that ended most recently BEFORE the announcement
                # (results are filed after the quarter closes). Rolling back to
                # the prior quarter-end is stable across the realistic 30-90d
                # filing-delay range; subtracting a fixed 45d drifts into the
                # NEXT quarter once the delay exceeds 45d, which broke dedup and
                # double-counted the result against the openscreener estimate.
                reported_quarter = (ann + pd.offsets.QuarterEnd(-1)).to_period("Q")
                nse_quarters.add((str(row["ticker"]), reported_quarter))
                rows.append(
                    {
                        "ticker": row["ticker"],
                        "earnings_date": row["earnings_date"],
                        "eps_estimate": float("nan"),
                        "reported_eps": float("nan"),
                        "surprise_pct": float("nan"),
                    }
                )
        else:
            logger.warning("india_nse_earnings_unavailable")

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(
                "openscreener_earnings_batch",
                extra={"batch": f"{i}-{i + len(batch)}", "size": len(batch)},
            )
            for osc_row in _fetch_openscreener_earnings_rows(batch, years, batch_size):
                # Drop openscreener rows whose fiscal quarter is already covered
                # by a real NSE announcement for the same ticker (dedup).
                pe = osc_row.get("period_end")
                if pe is not None:
                    quarter = pd.Timestamp(pe).to_period("Q")
                    if (str(osc_row["ticker"]), quarter) in nse_quarters:
                        continue
                rows.append({k: v for k, v in osc_row.items() if k != "period_end"})
    else:
        # US: yfinance
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            logger.info(
                "earnings_batch",
                extra={"batch": f"{i}-{i + len(batch)}", "size": len(batch)},
            )
            rows.extend(_fetch_yf_earnings_rows(batch, years, batch_size))

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
    For India: returns None; this avoids Yahoo lookups and keeps India on
    NSE/OpenScreener sources.
    """
    if market == "india":
        return None

    cache_path = _json_cache_path("analyst", f"{market}_{ticker}")
    hit, cached = _read_json_cache(cache_path, SENTIMENT_CACHE_DAYS)
    if hit:
        return cast("dict[Any, Any] | None", cached)

    _install_yfinance_timeout_patch()
    try:
        _configure_yfinance()
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            _write_json_cache(cache_path, None)
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
        result = _jsonable(result)
        _write_json_cache(cache_path, result)
        return cast("dict[Any, Any] | None", result)
    except Exception as exc:
        logger.debug(
            "analyst_sentiment_error", extra={"ticker": ticker, "error": str(exc)}
        )
        _write_json_cache(cache_path, None)
        return None


# ── Options / IV sentiment ──────────────────────────────────────────────


def fetch_iv_sentiment_yf(ticker: str) -> Optional[dict]:
    """Compute put/call ratio and IV percentile from yfinance (US only)."""
    cache_path = _json_cache_path("iv_yf", ticker)
    hit, cached = _read_json_cache(cache_path, SENTIMENT_CACHE_DAYS)
    if hit:
        return cast("dict[Any, Any] | None", cached)

    _install_yfinance_timeout_patch()
    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            _write_json_cache(cache_path, None)
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

        result = {
            "pc_ratio": round(pc_ratio, 4),
            "median_iv": round(median_iv, 2),
            "total_calls": total_calls,
            "total_puts": total_puts,
        }
        _write_json_cache(cache_path, _jsonable(result))
        return result
    except Exception as exc:
        logger.debug("iv_sentiment_error", extra={"ticker": ticker, "error": str(exc)})
        _write_json_cache(cache_path, None)
        return None


def fetch_iv_sentiment_nse(symbol: str) -> Optional[dict]:
    """Compute put/call ratio and IV from NSE option chain via jugaad_data.

    *symbol* is the NSE symbol (e.g. 'RELIANCE'), NOT the yfinance ticker.
    Uses openInterest for P/C ratio (more stable than volume).
    NSE option chain does include impliedVolatility per strike.
    """
    cache_path = _json_cache_path("iv_nse", symbol)
    hit, cached = _read_json_cache(cache_path, SENTIMENT_CACHE_DAYS)
    if hit:
        return cast("dict[Any, Any] | None", cached)

    try:
        from jugaad_data.nse import NSELive

        nse = NSELive()
        oc = nse.equities_option_chain(symbol)
        if not oc or "records" not in oc:
            _write_json_cache(cache_path, None)
            return None

        data = oc["records"].get("data", [])
        if not data:
            _write_json_cache(cache_path, None)
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

        result = {
            "pc_ratio": round(pc_ratio, 4),
            "median_iv": round(median_iv, 2) if not np.isnan(median_iv) else None,
            "total_calls": total_ce_vol,
            "total_puts": total_pe_vol,
        }
        _write_json_cache(cache_path, _jsonable(result))
        return result
    except Exception as exc:
        logger.debug(
            "nse_iv_sentiment_error", extra={"symbol": symbol, "error": str(exc)}
        )
        _write_json_cache(cache_path, None)
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
