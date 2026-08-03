"""Fundamental data adapters for backtests.

The backtester evaluates entry expressions against per-ticker OHLCV frames. This
module enriches those frames with dated fundamental columns so expressions can
remain ordinary Pine-like formulas while fundamentals stay point-in-time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import os
from typing import Any, Protocol, cast

import pandas as pd
import requests

from screener.backtester.data import load_env_file
from screener.financials import as_percent as _as_percent
from screener.financials import first_number, pct_change
from screener.providers import CachedProvider, ProviderSpec
from screener.provider_utils import fmp_get


INDIA_FUNDAMENTAL_FILING_LAG_DAYS = 60
DEFAULT_FUNDAMENTAL_FIELDS: tuple[str, ...] = (
    "pe_ttm",
    "pb_ttm",
    "roe_ttm",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "revenue_up_3q",
    "market_cap",
)


_FMP_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="fmp", namespace="backtester_fmp_fundamentals", ttl_seconds=86400
    )
)
_OPENSCREENER_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="openscreener",
        namespace="backtester_openscreener_fundamentals",
        ttl_seconds=86400,
    )
)
_YFINANCE_FUNDAMENTALS_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="yfinance",
        namespace="backtester_yfinance_fundamentals",
        ttl_seconds=86400,
    )
)


class FundamentalFetcher(Protocol):
    markets: frozenset[str]

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """Return yf-style ticker -> dated fundamental frame indexed by date."""


def _increased_last_n_quarters(
    rows: list[dict[str, Any]], current_pos: int, field: str, quarters: int
) -> float | None:
    end = current_pos + quarters
    if end > len(rows):
        return None
    values = [first_number(row, field) for row in rows[current_pos:end]]
    if any(value is None for value in values):
        return None
    numeric = cast(list[float], values)
    return 1.0 if all(a > b for a, b in zip(numeric, numeric[1:])) else 0.0


def _first_revenue(row: Mapping[str, Any]) -> float | None:
    return first_number(
        row,
        "revenue",
        "sales",
        "net_sales",
        "net sales",
        "sales_ttm",
        "income",
    )


def _increased_last_n_revenues(
    rows: list[dict[str, Any]], current_pos: int, quarters: int
) -> float | None:
    end = current_pos + quarters
    if end > len(rows):
        return None
    values = [_first_revenue(row) for row in rows[current_pos:end]]
    if any(value is None for value in values):
        return None
    numeric = cast(list[float], values)
    return 1.0 if all(a > b for a, b in zip(numeric, numeric[1:])) else 0.0


def _rows(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _row_by_date(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_date = row.get("date")
        if isinstance(row_date, str) and row_date:
            out.setdefault(row_date, row)
    return out


def _effective_date(row: Mapping[str, Any], lag_days: int) -> pd.Timestamp | None:
    raw = row.get("acceptedDate") or row.get("fillingDate") or row.get("date")
    if raw is None:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    ts = cast(pd.Timestamp, ts).tz_localize(None).normalize()
    if lag_days > 0:
        ts += pd.Timedelta(days=lag_days)
    return ts


_FMP_TIMEOUT = 30.0


def _fmp_get(
    session: requests.Session,
    path: str,
    params: Mapping[str, Any],
    api_key: str,
) -> object:
    return fmp_get(
        path, params, api_key, session=session, headers={}, timeout=_FMP_TIMEOUT
    )


def _fetch_fmp_sections(
    symbol: str,
    *,
    api_key: str,
    session: requests.Session,
    limit: int,
    fields: tuple[str, ...] = DEFAULT_FUNDAMENTAL_FIELDS,
) -> dict[str, Any]:
    params = {"period": "quarter", "limit": limit}
    payload: dict[str, Any] = {
        "income": _fmp_get(session, f"income-statement/{symbol}", params, api_key),
    }
    if any(field in fields for field in ("debt_to_equity",)):
        payload["balance"] = _fmp_get(
            session, f"balance-sheet-statement/{symbol}", params, api_key
        )
    if any(
        field in fields for field in ("pe_ttm", "pb_ttm", "roe_ttm", "debt_to_equity")
    ):
        payload["ratios"] = _fmp_get(session, f"ratios/{symbol}", params, api_key)
    if any(field in fields for field in ("pe_ttm", "pb_ttm")):
        payload["key_metrics"] = _fmp_get(
            session, f"key-metrics/{symbol}", params, api_key
        )
    if "market_cap" in fields:
        payload["enterprise_values"] = _fmp_get(
            session, f"enterprise-values/{symbol}", params, api_key
        )
    return payload


def _normalize_fmp_payload(
    payload: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    lag_days: int,
) -> pd.DataFrame:
    income_rows = _rows(payload, "income")
    if not income_rows:
        return pd.DataFrame(columns=list(fields), index=pd.DatetimeIndex([]))

    ratio_by_date = _row_by_date(_rows(payload, "ratios"))
    metrics_by_date = _row_by_date(_rows(payload, "key_metrics"))
    balance_by_date = _row_by_date(_rows(payload, "balance"))
    ev_by_date = _row_by_date(_rows(payload, "enterprise_values"))
    income_by_date = _row_by_date(income_rows)

    records: list[dict[str, Any]] = []
    for current_pos, row in enumerate(income_rows):
        row_date = row.get("date")
        if not isinstance(row_date, str) or not row_date:
            continue
        effective = _effective_date(row, lag_days)
        if effective is None:
            continue

        ratio = ratio_by_date.get(row_date, {})
        metrics = metrics_by_date.get(row_date, {})
        balance = balance_by_date.get(row_date, {})
        enterprise = ev_by_date.get(row_date, {})

        ts = pd.Timestamp(row_date)
        prior_target = (ts - pd.DateOffset(years=1)).date().isoformat()
        prior_income = income_by_date.get(prior_target, {})
        revenue_growth = pct_change(
            first_number(row, "revenue"),
            first_number(prior_income, "revenue"),
        )
        eps_growth = pct_change(
            first_number(row, "eps", "epsdiluted"),
            first_number(prior_income, "eps", "epsdiluted"),
        )

        record = {
            "effective_date": effective,
            "pe_ttm": first_number(
                metrics,
                "peRatioTTM",
                "peRatio",
                "priceEarningsRatioTTM",
                "priceEarningsRatio",
            )
            or first_number(ratio, "priceEarningsRatio", "priceEarningsRatioTTM"),
            "pb_ttm": first_number(metrics, "pbRatioTTM", "pbRatio", "priceToBookRatio")
            or first_number(ratio, "priceToBookRatio", "priceBookValueRatio"),
            "roe_ttm": _as_percent(
                first_number(ratio, "returnOnEquity", "returnOnEquityTTM")
            ),
            "debt_to_equity": first_number(
                ratio, "debtEquityRatio", "debtToEquity", "debtToEquityRatio"
            )
            or first_number(balance, "totalDebtToEquity"),
            "revenue_growth_yoy": revenue_growth,
            "eps_growth_yoy": eps_growth,
            "revenue_up_3q": _increased_last_n_quarters(
                income_rows, current_pos, "revenue", 3
            ),
            "market_cap": first_number(enterprise, "marketCapitalization", "marketCap"),
        }
        records.append({key: record.get(key) for key in ("effective_date", *fields)})

    if not records:
        return pd.DataFrame(columns=list(fields), index=pd.DatetimeIndex([]))

    frame = pd.DataFrame(records)
    frame.index = pd.to_datetime(frame.pop("effective_date"), errors="coerce")
    frame = frame[frame.index.notna()]
    frame.index = cast(pd.DatetimeIndex, frame.index).tz_localize(None).normalize()
    for col in fields:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.reindex(columns=list(fields))


class FMPFundamentalFetcher:
    """Fetch dated US fundamentals from FMP and normalize expression columns."""

    markets = frozenset({"us"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        fields: Iterable[str] | None = None,
        lag_days: int = 1,
        refresh: bool = False,
        cache_ttl: float | None = 86400,
        session: requests.Session | None = None,
        limit: int = 120,
        max_workers: int = 8,
    ) -> None:
        load_env_file()
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required to use FMP fundamentals")
        self.fields = tuple(dict.fromkeys(fields or DEFAULT_FUNDAMENTAL_FIELDS))
        self.lag_days = max(int(lag_days), 0)
        self.refresh = bool(refresh)
        self.cache_ttl = cache_ttl
        self.session = session or requests.Session()
        self.limit = max(int(limit), 1)
        self.max_workers = max(int(max_workers), 1)

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        ticker_list = [t for t in dict.fromkeys(tickers) if t]

        def fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
            symbol = ticker.split(".", 1)[0].upper()

            def fetch_payload(symbol: str = symbol) -> dict[str, Any]:
                return _fetch_fmp_sections(
                    symbol,
                    api_key=cast(str, self.api_key),
                    session=self.session
                    if self.max_workers == 1
                    else requests.Session(),
                    limit=self.limit,
                    fields=self.fields,
                )

            fallback: dict[str, Any] = {}
            payload: dict[str, Any] = _FMP_PROVIDER.fetch(
                ("us", symbol, self.limit, self.fields),
                fetch_payload,
                refresh=self.refresh,
                fallback=fallback,
                ttl_seconds=self.cache_ttl,
                operation=f"backtest fundamentals {symbol}",
            )
            frame = _normalize_fmp_payload(
                payload,
                fields=self.fields,
                lag_days=self.lag_days,
            )
            if not frame.empty:
                frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
            return ticker, frame

        out: dict[str, pd.DataFrame] = {}
        if len(ticker_list) <= 1 or self.max_workers == 1:
            for ticker in ticker_list:
                key, frame = fetch_one(ticker)
                out[key] = frame
            return out

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(ticker_list))
        ) as pool:
            future_to_ticker = {
                pool.submit(fetch_one, ticker): ticker for ticker in ticker_list
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    key, frame = future.result()
                except Exception:
                    out[ticker] = pd.DataFrame(columns=list(self.fields))
                else:
                    out[key] = frame
        return out


def _parse_india_period_end(label: Any) -> pd.Timestamp | None:
    if not label:
        return None
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return pd.to_datetime(str(label), format=fmt) + pd.offsets.MonthEnd(0)
        except (TypeError, ValueError):
            continue
    ts = pd.to_datetime(label, errors="coerce")
    if pd.isna(ts):
        return None
    return cast(pd.Timestamp, ts).tz_localize(None).normalize() + pd.offsets.MonthEnd(0)


def _fetch_openscreener_quarterly(symbol: str) -> dict[str, Any]:  # pragma: no cover
    from openscreener import Stock
    from screener.insiders import _HttpScraper

    payload = Stock(symbol, scraper=_HttpScraper()).fetch("quarterly_results")
    return payload if isinstance(payload, dict) else {}


def _fetch_yfinance_quarterly_revenue(
    ticker: str,
) -> dict[str, Any]:  # pragma: no cover
    import yfinance as yf

    stock = yf.Ticker(ticker)
    statement = stock.quarterly_financials
    if statement is None or statement.empty:
        statement = stock.quarterly_income_stmt
    if statement is None or statement.empty:
        return {}
    revenue = pd.Series(dtype=float)
    for row_name in ("Total Revenue", "Operating Revenue"):
        if row_name in statement.index:
            revenue = pd.to_numeric(statement.loc[row_name], errors="coerce").dropna()
            break
    if revenue.empty:
        return {}

    records: list[dict[str, Any]] = []
    for raw_date, value in revenue.items():
        period_end = pd.Timestamp(cast(Any, raw_date)).tz_localize(None).normalize()
        records.append(
            {
                "date": period_end.strftime("%b %Y"),
                "sales": float(value),
            }
        )
    return {"quarterly_results": records}


def _normalize_openscreener_payload(
    payload: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    lag_days: int,
) -> pd.DataFrame:
    quarterly = _rows(payload, "quarterly_results")
    dated_rows: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in quarterly:
        period_end = _parse_india_period_end(row.get("date"))
        if period_end is not None:
            dated_rows.append((period_end, row))
    if not dated_rows:
        return pd.DataFrame(columns=list(fields), index=pd.DatetimeIndex([]))

    dated_rows.sort(key=lambda item: item[0], reverse=True)
    sorted_rows = [row for _period_end, row in dated_rows]
    records: list[dict[str, Any]] = []
    for pos, (period_end, _row) in enumerate(dated_rows):
        effective = period_end + pd.Timedelta(days=lag_days)
        record: dict[str, Any] = {"effective_date": effective}
        if "revenue_up_3q" in fields:
            record["revenue_up_3q"] = _increased_last_n_revenues(sorted_rows, pos, 3)
        records.append({key: record.get(key) for key in ("effective_date", *fields)})

    frame = pd.DataFrame(records)
    frame.index = pd.to_datetime(frame.pop("effective_date"), errors="coerce")
    frame = frame[frame.index.notna()]
    frame.index = cast(pd.DatetimeIndex, frame.index).tz_localize(None).normalize()
    for col in fields:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.reindex(columns=list(fields))


class OpenScreenerFundamentalFetcher:
    """Fetch India quarterly fundamentals from openscreener/screener.in."""

    markets = frozenset({"india"})

    def __init__(
        self,
        *,
        fields: Iterable[str] | None = None,
        lag_days: int = INDIA_FUNDAMENTAL_FILING_LAG_DAYS,
        refresh: bool = False,
        cache_ttl: float | None = 86400,
        max_workers: int = 8,
    ) -> None:
        self.fields = tuple(dict.fromkeys(fields or ("revenue_up_3q",)))
        self.lag_days = max(int(lag_days), 0)
        self.refresh = bool(refresh)
        self.cache_ttl = cache_ttl
        self.max_workers = max(int(max_workers), 1)

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        ticker_list = [t for t in dict.fromkeys(tickers) if t]

        def fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
            symbol = ticker.replace(".NS", "").replace(".BO", "").upper()

            fallback: dict[str, Any] = {}

            def fetch_payload(symbol: str = symbol) -> dict[str, Any]:
                return _fetch_openscreener_quarterly(symbol)

            payload: dict[str, Any] = _OPENSCREENER_PROVIDER.fetch(
                ("india", symbol),
                fetch_payload,
                refresh=self.refresh,
                fallback=fallback,
                ttl_seconds=self.cache_ttl,
                operation=f"backtest fundamentals {symbol}",
            )
            frame = _normalize_openscreener_payload(
                payload,
                fields=self.fields,
                lag_days=self.lag_days,
            )
            if frame.empty:

                def fetch_yf_payload(ticker: str = ticker) -> dict[str, Any]:
                    return _fetch_yfinance_quarterly_revenue(ticker)

                yf_payload: dict[str, Any] = _YFINANCE_FUNDAMENTALS_PROVIDER.fetch(
                    ("india", ticker),
                    fetch_yf_payload,
                    refresh=self.refresh,
                    fallback=fallback,
                    ttl_seconds=self.cache_ttl,
                    operation=f"backtest yfinance fundamentals {ticker}",
                )
                frame = _normalize_openscreener_payload(
                    yf_payload,
                    fields=self.fields,
                    lag_days=self.lag_days,
                )
            if not frame.empty:
                frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
            return ticker, frame

        out: dict[str, pd.DataFrame] = {}
        if len(ticker_list) <= 1 or self.max_workers == 1:
            for ticker in ticker_list:
                key, frame = fetch_one(ticker)
                out[key] = frame
            return out

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(ticker_list))
        ) as pool:
            future_to_ticker = {
                pool.submit(fetch_one, ticker): ticker for ticker in ticker_list
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    key, frame = future.result()
                except Exception:
                    out[ticker] = pd.DataFrame(columns=list(self.fields))
                else:
                    out[key] = frame
        return out


class YFinanceFundamentalFetcher:
    """Fetch India quarterly revenue fundamentals from yfinance."""

    markets = frozenset({"india"})

    def __init__(
        self,
        *,
        fields: Iterable[str] | None = None,
        lag_days: int = INDIA_FUNDAMENTAL_FILING_LAG_DAYS,
        refresh: bool = False,
        cache_ttl: float | None = 86400,
        max_workers: int = 8,
    ) -> None:
        self.fields = tuple(dict.fromkeys(fields or ("revenue_up_3q",)))
        self.lag_days = max(int(lag_days), 0)
        self.refresh = bool(refresh)
        self.cache_ttl = cache_ttl
        self.max_workers = max(int(max_workers), 1)

    def fetch(
        self,
        tickers: Iterable[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        ticker_list = [t for t in dict.fromkeys(tickers) if t]

        def fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
            fallback: dict[str, Any] = {}

            def fetch_payload(ticker: str = ticker) -> dict[str, Any]:
                return _fetch_yfinance_quarterly_revenue(ticker)

            payload: dict[str, Any] = _YFINANCE_FUNDAMENTALS_PROVIDER.fetch(
                ("india", ticker),
                fetch_payload,
                refresh=self.refresh,
                fallback=fallback,
                ttl_seconds=self.cache_ttl,
                operation=f"backtest yfinance fundamentals {ticker}",
            )
            frame = _normalize_openscreener_payload(
                payload,
                fields=self.fields,
                lag_days=self.lag_days,
            )
            if not frame.empty:
                frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
            return ticker, frame

        out: dict[str, pd.DataFrame] = {}
        if len(ticker_list) <= 1 or self.max_workers == 1:
            for ticker in ticker_list:
                key, frame = fetch_one(ticker)
                out[key] = frame
            return out

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(ticker_list))
        ) as pool:
            future_to_ticker = {
                pool.submit(fetch_one, ticker): ticker for ticker in ticker_list
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    key, frame = future.result()
                except Exception:
                    out[ticker] = pd.DataFrame(columns=list(self.fields))
                else:
                    out[key] = frame
        return out


@dataclass(frozen=True)
class _FundamentalProvider:
    fetcher_type: type[Any]
    default_lag_days: int


_FUNDAMENTAL_PROVIDERS = {
    "fmp": _FundamentalProvider(FMPFundamentalFetcher, 1),
    "openscreener": _FundamentalProvider(
        OpenScreenerFundamentalFetcher, INDIA_FUNDAMENTAL_FILING_LAG_DAYS
    ),
    "yfinance": _FundamentalProvider(
        YFinanceFundamentalFetcher, INDIA_FUNDAMENTAL_FILING_LAG_DAYS
    ),
}
_FUNDAMENTAL_PROVIDER_ALIASES = {"open-screener": "openscreener", "yf": "yfinance"}


def resolve_fundamental_provider(provider: str | None) -> str | None:
    """Normalize a provider name without constructing its adapter."""
    resolved = (provider or "").strip().lower()
    if not resolved:
        return None
    resolved = _FUNDAMENTAL_PROVIDER_ALIASES.get(resolved, resolved)
    if resolved not in _FUNDAMENTAL_PROVIDERS:
        raise ValueError(f"Unknown fundamentals provider: {provider}")
    return resolved


def fundamental_filing_lag_days(provider: str | None) -> int:
    """Return the default reporting lag for a normalized provider."""
    resolved = resolve_fundamental_provider(provider)
    return _FUNDAMENTAL_PROVIDERS[resolved].default_lag_days if resolved else 1


def build_fundamental_fetcher(
    provider: str | None,
    *,
    market: str,
    fields: Iterable[str] | None = None,
    lag_days: int = 1,
    refresh: bool = False,
) -> FundamentalFetcher | None:
    """Construct the one market-compatible adapter for a provider."""
    resolved = resolve_fundamental_provider(provider)
    if resolved is None:
        return None
    definition = _FUNDAMENTAL_PROVIDERS[resolved]
    if market not in definition.fetcher_type.markets:
        supported = ", ".join(sorted(definition.fetcher_type.markets))
        raise ValueError(
            f"--fundamentals-provider {resolved} currently supports only -m {supported}."
        )
    return cast(
        FundamentalFetcher,
        definition.fetcher_type(fields=fields, lag_days=lag_days, refresh=refresh),
    )


def merge_fundamentals_into_bars(
    bars_by_tv: dict[str, pd.DataFrame],
    fundamentals_by_yf: Mapping[str, pd.DataFrame],
    yf_by_tv: Mapping[str, str],
) -> dict[str, pd.DataFrame]:
    """Forward-fill dated fundamentals onto each ticker's OHLCV frame."""
    out: dict[str, pd.DataFrame] = {}
    for tv_symbol, bars in bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv_symbol] = bars
            continue
        fundamentals = fundamentals_by_yf.get(yf_by_tv.get(tv_symbol, ""))
        if fundamentals is None or fundamentals.empty:
            out[tv_symbol] = bars
            continue
        aligned = fundamentals.sort_index().reindex(bars.index, method="ffill")
        out[tv_symbol] = bars.join(aligned, how="left")
    return out
