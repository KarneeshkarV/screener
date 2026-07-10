"""Earnings-date acquisition for the earnings backtest.

Fetches point-in-time earnings/result dates from the three supported sources:
yfinance (US), NSE corporate announcements (India, via ``jugaad_data``), and
screener.in quarterly results (India, via ``openscreener``), plus the batch
collector that unifies them.

Split out of :mod:`screener.earnings_backtest.data`, which re-exports every
public name here for backwards compatibility. Cross-function and seam
dependencies are looked up through the ``data`` module (``data.<name>``) so the
test suite's ``monkeypatch.setattr(data, ...)`` patches keep taking effect.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Optional, cast

import pandas as pd
import yfinance as yf

from screener.earnings_backtest import data

logger = logging.getLogger(__name__)

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


def _earnings_to_records(ed: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in ed.iterrows():
        # iterrows() types the index label as Hashable; it is a datetime here.
        ts = pd.Timestamp(cast(Any, idx)).tz_localize(None).normalize()
        records.append(
            {
                "earnings_date": ts.date().isoformat(),
                "eps_estimate": data._jsonable(row.get("EPS Estimate", float("nan"))),
                "reported_eps": data._jsonable(row.get("Reported EPS", float("nan"))),
                "surprise_pct": data._jsonable(row.get("Surprise(%)", float("nan"))),
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


# ── Earnings dates (yfinance) ────────────────────────────────────────────


def fetch_earnings_dates_yf(
    ticker: str,
    years: int = 3,
) -> Optional[pd.DataFrame]:
    """Return yfinance earnings_dates for *ticker*."""

    def _fetch() -> list[dict[str, Any]]:
        data._configure_yfinance()
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                return []
            ed = ed.copy()
            ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
            cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
            ed = ed[ed.index >= cutoff]
            return _earnings_to_records(ed) if not ed.empty else []
        except Exception as exc:
            logger.debug(
                "earnings_dates_failed", extra={"ticker": ticker, "error": str(exc)}
            )
            return []

    records = data.cached_json_call(
        "earnings_yf",
        (ticker, years),
        ttl_seconds=data.EARNINGS_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )
    return _earnings_from_records(records)


def fetch_earnings_dates_nse() -> Optional[pd.DataFrame]:
    """Fetch earnings result dates from NSE corporate announcements via jugaad_data."""

    def _fetch() -> list[dict[str, Any]]:
        try:
            from jugaad_data.nse import NSELive

            nse = NSELive()
            announcements = nse.corporate_announcements()
            if not announcements:
                return []

            rows: list[dict[str, Any]] = []
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
                return []
            df = pd.DataFrame(rows)
            df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.strftime(
                "%Y-%m-%d"
            )
            return cast("list[dict[str, Any]]", df.to_dict("records"))
        except Exception as exc:
            logger.warning("nse_earnings_fetch_failed", extra={"error": str(exc)})
            return []

    cached = data.cached_json_call(
        "earnings_nse",
        "corporate_announcements",
        ttl_seconds=data.SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )
    if not cached:
        return None
    df = pd.DataFrame(cached)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def _earnings_rows_for_ticker(ticker: str, years: int) -> list[dict[str, Any]]:
    ed = data.fetch_earnings_dates_yf(ticker, years=years)
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
    max_workers = min(data.MAX_WORKERS, max(1, batch_size), max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(data._earnings_rows_for_ticker, ticker, years): ticker
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

    def _fetch() -> list[dict[str, Any]]:
        from openscreener import Stock
        from screener.insiders import _HttpScraper

        payload = Stock(symbol, scraper=_HttpScraper()).fetch("quarterly_results")
        if not isinstance(payload, dict):
            return []
        quarterly = payload.get("quarterly_results")
        if not isinstance(quarterly, list) or not quarterly:
            return []

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
                    "reported_eps": data._jsonable(item.get("eps")),
                    "surprise_pct": None,
                }
            )
        return records

    try:
        records = data.cached_json_call(
            "earnings_openscreener",
            (symbol, years, filing_lag_days),
            ttl_seconds=data.EARNINGS_CACHE_DAYS * 86_400,
            refresh=False,
            fetch=_fetch,
        )
    except Exception as exc:
        logger.debug(
            "openscreener_earnings_failed",
            extra={"ticker": ticker, "error": str(exc)},
        )
        return None
    return _earnings_from_records(records)


def _openscreener_earnings_rows_for_ticker(
    ticker: str, years: int
) -> list[dict[str, Any]]:
    ed = data.fetch_earnings_dates_openscreener(ticker, years=years)
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
                data._openscreener_earnings_rows_for_ticker, ticker, years
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
        nse_events = data.fetch_earnings_dates_nse()
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
            for osc_row in data._fetch_openscreener_earnings_rows(
                batch, years, batch_size
            ):
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
            rows.extend(data._fetch_yf_earnings_rows(batch, years, batch_size))

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
