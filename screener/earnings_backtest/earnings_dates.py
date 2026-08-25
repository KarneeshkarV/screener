"""Earnings-date acquisition for the earnings backtest.

Fetches point-in-time earnings/result dates from the three supported sources:
yfinance (US), NSE corporate announcements (India, via ``jugaad_data``), and
screener.in quarterly results (India, via ``openscreener``), plus the batch
collector that unifies them.

The compatibility facade re-exports the public names from this canonical
module; this module never imports the facade back.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from screener import _optional
from screener.backtester.data import _configure_yfinance, call_yfinance_with_timeout
from screener.cache import cached_json_call
from screener.earnings_backtest.common import (
    EARNINGS_CACHE_DAYS,
    MAX_WORKERS,
    SENTIMENT_CACHE_DAYS,
    jsonable,
)

if TYPE_CHECKING:
    import yfinance as yf
else:
    yf = _optional.load("yfinance")

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
                "eps_estimate": jsonable(row.get("EPS Estimate", float("nan"))),
                "reported_eps": jsonable(row.get("Reported EPS", float("nan"))),
                "surprise_pct": jsonable(row.get("Surprise(%)", float("nan"))),
            }
        )
    return records


def _earnings_from_records(records: list[dict[str, Any]]) -> pd.DataFrame | None:
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
) -> pd.DataFrame | None:
    """Return yfinance earnings_dates for *ticker*."""

    def _fetch() -> list[dict[str, Any]]:
        _configure_yfinance()

        def _request() -> list[dict[str, Any]]:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                return []
            ed = ed.copy()
            ed.index = pd.to_datetime(ed.index).tz_localize(None).normalize()
            cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
            ed = ed[ed.index >= cutoff]
            return _earnings_to_records(ed) if not ed.empty else []

        try:
            return call_yfinance_with_timeout(_request)
        except Exception as exc:
            logger.debug(
                "earnings_dates_failed", extra={"ticker": ticker, "error": str(exc)}
            )
            return []

    records = cached_json_call(
        "earnings_yf",
        (ticker, years),
        ttl_seconds=EARNINGS_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )
    return _earnings_from_records(records)


def fetch_earnings_dates_nse() -> pd.DataFrame | None:
    """Fetch earnings result dates from NSE corporate announcements via jugaad_data."""

    def _fetch() -> list[dict[str, Any]]:
        try:
            NSELive = _optional.load("jugaad_data.nse").NSELive

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

    cached = cached_json_call(
        "earnings_nse",
        "corporate_announcements",
        ttl_seconds=SENTIMENT_CACHE_DAYS * 86_400,
        refresh=False,
        fetch=_fetch,
    )
    if not cached:
        return None
    df = pd.DataFrame(cached)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


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
) -> pd.DataFrame | None:
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
        Stock = _optional.load("openscreener").Stock

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
                    "reported_eps": jsonable(item.get("eps")),
                    "surprise_pct": None,
                }
            )
        return records

    try:
        records = cached_json_call(
            "earnings_openscreener",
            (symbol, years, filing_lag_days),
            ttl_seconds=EARNINGS_CACHE_DAYS * 86_400,
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


# ── Earnings dates + EPS surprise (Financial Modeling Prep) ──────────────


def fetch_earnings_dates_fmp(
    ticker: str,
    years: int = 3,
) -> pd.DataFrame | None:
    """Return FMP historical earnings for *ticker* with a computed EPS surprise.

    FMP's ``historical/earning_calendar`` endpoint reports the real
    announcement ``date`` together with ``eps`` and ``epsEstimated``, so an EPS
    surprise ``(eps - epsEstimated) / |epsEstimated| * 100`` is computable — the
    piece the India NSE/openscreener path lacks. FMP uses the same ``.NS`` /
    ``.BO`` suffixes as this codebase, so *ticker* is passed through unchanged.

    Requires ``FMP_API_KEY`` (resolved via :func:`screener.fmp.resolve_api_key`,
    the shared key helper); returns ``None`` when the key is unset or the
    endpoint yields nothing usable. Only the lower ``years`` cutoff is applied,
    mirroring :func:`fetch_earnings_dates_yf`.
    """
    from screener import fmp

    def _fetch() -> list[dict[str, Any]]:
        api_key = fmp.resolve_api_key()
        if not api_key:
            return []
        try:
            client = fmp.FmpClient(api_key, base_url=fmp.FMP_V3_BASE_URL)
            payload = client.get(f"historical/earning_calendar/{ticker}")
        except Exception as exc:
            logger.debug(
                "fmp_earnings_failed", extra={"ticker": ticker, "error": str(exc)}
            )
            return []
        if not isinstance(payload, list) or not payload:
            return []

        cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))
        records: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            raw = item.get("date")
            if not raw:
                continue
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.isna(ts):
                continue
            ed = ts.normalize()
            if ed < cutoff:
                continue
            eps = item.get("eps")
            eps_est = item.get("epsEstimated")
            surprise = None
            if eps is not None and eps_est is not None and eps_est != 0:
                surprise = (eps - eps_est) / abs(eps_est) * 100.0
            records.append(
                {
                    "earnings_date": ed.date().isoformat(),
                    "eps_estimate": jsonable(eps_est),
                    "reported_eps": jsonable(eps),
                    "surprise_pct": jsonable(surprise),
                }
            )
        return records

    try:
        records = cached_json_call(
            "earnings_fmp",
            (ticker, years),
            ttl_seconds=EARNINGS_CACHE_DAYS * 86_400,
            refresh=False,
            fetch=_fetch,
        )
    except Exception as exc:
        logger.debug(
            "fmp_earnings_cache_failed", extra={"ticker": ticker, "error": str(exc)}
        )
        return None
    return _earnings_from_records(records)


def _fmp_earnings_rows_for_ticker(ticker: str, years: int) -> list[dict[str, Any]]:
    ed = fetch_earnings_dates_fmp(ticker, years=years)
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


def _fetch_fmp_earnings_rows(
    tickers: list[str], years: int, batch_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_workers = min(MAX_WORKERS, max(1, batch_size), max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_fmp_earnings_rows_for_ticker, ticker, years): ticker
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                logger.debug(
                    "fmp_earnings_collect_error",
                    extra={"ticker": ticker, "error": str(exc)},
                )
    return rows


def _reported_quarter(announcement: pd.Timestamp) -> pd.Period:
    """Map an announcement date to the fiscal quarter it reports on.

    Results are filed after a quarter closes, so the reported quarter is the one
    that ended most recently BEFORE the announcement. Rolling back to the prior
    quarter-end is stable across the realistic 30-90d filing-delay range (see the
    NSE dedup rationale in :func:`collect_earnings_events`).
    """
    return (announcement + pd.offsets.QuarterEnd(-1)).to_period("Q")


def _collect_india_fmp_events(
    tickers: list[str], years: int, batch_size: int
) -> list[dict[str, Any]]:
    """India earnings with FMP EPS surprise, reconciled against NSE dates.

    Reconciliation policy (documented in the PR): NSE corporate announcements
    carry the authoritative point-in-time announcement date but no EPS surprise;
    FMP carries both a date and a computable surprise. For each fiscal quarter we
    PREFER the real NSE announcement date and ENRICH it with the FMP surprise;
    quarters covered only by FMP are added using FMP's own (already
    point-in-time) announcement date. openscreener is deliberately not used in
    this mode — it has no surprise and only an estimated date that FMP
    supersedes.
    """
    cutoff = pd.Timestamp(date.today() - timedelta(days=years * 365))

    # FMP surprise rows keyed by (ticker, fiscal quarter reported on).
    fmp_by_quarter: dict[tuple[str, pd.Period], dict[str, Any]] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info(
            "fmp_earnings_batch",
            extra={"batch": f"{i}-{i + len(batch)}", "size": len(batch)},
        )
        for fmp_row in _fetch_fmp_earnings_rows(batch, years, batch_size):
            quarter = _reported_quarter(pd.Timestamp(fmp_row["earnings_date"]))
            fmp_by_quarter[(str(fmp_row["ticker"]), quarter)] = fmp_row

    rows: list[dict[str, Any]] = []
    matched_fmp: set[tuple[str, pd.Period]] = set()

    nse_events = fetch_earnings_dates_nse()
    if nse_events is not None and not nse_events.empty:
        ticker_set = set(tickers)
        filtered = nse_events[nse_events["ticker"].isin(ticker_set)]
        filtered = filtered[filtered["earnings_date"] >= cutoff]
        for _, row in filtered.iterrows():
            ann = pd.Timestamp(row["earnings_date"])
            key = (str(row["ticker"]), _reported_quarter(ann))
            fmp_match = fmp_by_quarter.get(key)
            if fmp_match is not None:
                matched_fmp.add(key)
            rows.append(
                {
                    "ticker": row["ticker"],
                    "earnings_date": row["earnings_date"],
                    "eps_estimate": (
                        fmp_match["eps_estimate"] if fmp_match else float("nan")
                    ),
                    "reported_eps": (
                        fmp_match["reported_eps"] if fmp_match else float("nan")
                    ),
                    "surprise_pct": (
                        fmp_match["surprise_pct"] if fmp_match else float("nan")
                    ),
                }
            )
    else:
        logger.warning("india_nse_earnings_unavailable")

    # FMP-only quarters (no matching NSE announcement): keep FMP's own date.
    for key, fmp_row in fmp_by_quarter.items():
        if key in matched_fmp:
            continue
        rows.append(
            {
                "ticker": fmp_row["ticker"],
                "earnings_date": fmp_row["earnings_date"],
                "eps_estimate": fmp_row["eps_estimate"],
                "reported_eps": fmp_row["reported_eps"],
                "surprise_pct": fmp_row["surprise_pct"],
            }
        )
    return rows


# ── Batch earnings collector ────────────────────────────────────────────


def collect_earnings_events(
    tickers: list[str],
    years: int = 3,
    batch_size: int = 50,
    market: str = "us",
    surprise_source: str | None = None,
) -> pd.DataFrame:
    """Collect earnings dates for all *tickers*.

    For India: uses jugaad_data (NSE announcements) + screener.in (openscreener)
    by default. Pass ``surprise_source="fmp"`` to instead source India earnings
    from Financial Modeling Prep (NSE dates enriched with FMP EPS surprise; see
    :func:`_collect_india_fmp_events`) — the only India source that carries a
    computable surprise, as required by the PEAD backtest. The default keeps the
    existing India behaviour bit-for-bit unchanged.

    For US: uses yfinance only (already carries EPS surprise); ``surprise_source``
    is ignored.
    """
    rows: list[dict] = []

    if market == "india" and surprise_source == "fmp":
        rows = _collect_india_fmp_events(tickers, years, batch_size)
    elif market == "india":
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


def events_to_dates_map(events: pd.DataFrame) -> dict[str, list[date]]:
    """Group an events frame into ``ticker -> sorted unique earnings dates``."""
    if events is None or events.empty:
        return {}
    if "ticker" not in events.columns or "earnings_date" not in events.columns:
        return {}
    out: dict[str, list[date]] = {}
    for ticker, group in events.groupby("ticker", sort=False):
        dates: list[date] = []
        for raw in group["earnings_date"]:
            ts = pd.Timestamp(raw)
            if pd.isna(ts):
                continue
            dates.append(ts.normalize().date())
        if dates:
            out[str(ticker)] = sorted(set(dates))
    return out


def load_earnings_dates_map(
    tickers: list[str],
    market: str,
    *,
    years: int = 5,
    collect_fn: Any | None = None,
) -> dict[str, list[date]]:
    """Collect historical earnings dates for *tickers* as a blackout map.

    Uses :func:`collect_earnings_events` (or an injectable *collect_fn*) which
    already caches per-source results on disk. Returns ``ticker -> list[date]``
    for tickers that have at least one known event; missing tickers are omitted.
    """
    if not tickers:
        return {}
    fn = collect_fn if collect_fn is not None else collect_earnings_events
    try:
        events = fn(list(tickers), years=years, market=market)
    except Exception as exc:
        logger.warning(
            "earnings_dates_map_failed",
            extra={"market": market, "n_tickers": len(tickers), "error": str(exc)},
        )
        return {}
    return events_to_dates_map(events)


def next_earnings_date(
    earnings_dates: list[date] | pd.DatetimeIndex | pd.Series | None,
    as_of: date,
) -> date | None:
    """Return the earliest earnings date on or after *as_of*, else ``None``."""
    if earnings_dates is None:
        return None
    as_of_ts = pd.Timestamp(as_of).normalize()
    upcoming: list[date] = []
    for raw in earnings_dates:
        ts = pd.Timestamp(raw)
        if pd.isna(ts):
            continue
        ts = ts.normalize()
        if ts >= as_of_ts:
            upcoming.append(ts.date())
    return min(upcoming) if upcoming else None


def fetch_next_earnings_dates(
    symbols: list[str],
    market: str,
    *,
    as_of: date | None = None,
    yf_fetcher: Any | None = None,
    nse_fetcher: Any | None = None,
) -> dict[str, date | None]:
    """Map screen symbols to their next known earnings date (or ``None``).

    Market-aware: yfinance for US, NSE corporate announcements for India.
    Provider failures are logged and yield ``None`` for affected symbols so a
    live screen never aborts on earnings data.
    """
    as_of_d = as_of or date.today()
    result: dict[str, date | None] = {sym: None for sym in symbols}
    if not symbols:
        return result

    if market == "india":
        fetch_nse = nse_fetcher if nse_fetcher is not None else fetch_earnings_dates_nse
        try:
            nse_df = fetch_nse()
        except Exception as exc:
            logger.warning("nse_next_earnings_failed", extra={"error": str(exc)})
            return result
        if nse_df is None or nse_df.empty:
            return result
        # Build yf-style ticker -> next date, then map back to input symbols.
        by_yf: dict[str, date | None] = {}
        for ticker, group in nse_df.groupby("ticker", sort=False):
            by_yf[str(ticker)] = next_earnings_date(group["earnings_date"], as_of_d)
        from screener.symbols import tv_to_yf

        for sym in symbols:
            yf_sym = tv_to_yf(str(sym), "india")
            result[sym] = by_yf.get(yf_sym)
            if result[sym] is None:
                # Also try bare / .NS variants for name-only screen rows.
                bare = str(sym).replace(".NS", "").replace(".BO", "")
                result[sym] = by_yf.get(f"{bare}.NS") or by_yf.get(bare)
        return result

    # US (and other non-india): per-ticker yfinance earnings_dates (cached).
    fetch_yf = yf_fetcher if yf_fetcher is not None else fetch_earnings_dates_yf
    from screener.symbols import tv_to_yf

    def _fetch_one(sym: str) -> tuple[str, date | None]:
        yf_sym = tv_to_yf(str(sym), market)
        try:
            ed = fetch_yf(yf_sym)
        except Exception as exc:
            logger.warning(
                "yf_next_earnings_failed",
                extra={"ticker": yf_sym, "error": str(exc)},
            )
            return sym, None
        if ed is None or (isinstance(ed, pd.DataFrame) and ed.empty):
            return sym, None
        if isinstance(ed, pd.DataFrame):
            return sym, next_earnings_date(list(ed.index), as_of_d)
        return sym, next_earnings_date(ed, as_of_d)

    max_workers = min(MAX_WORKERS, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym, earnings_date in executor.map(_fetch_one, symbols):
            result[sym] = earnings_date
    return result
