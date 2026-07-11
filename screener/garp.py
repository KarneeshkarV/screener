"""GARP screen helpers for India and US markets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from screener.cache import cached_json_call
from screener.financials import first_number, pct_change, to_number
from screener.fmp import resolve_api_key
from screener.markets import get_market
from screener.parallel import parallel_map
from screener.provider_utils import fmp_get
from screener.providers import CachedProvider, ProviderSpec
from screener.scanner import scan
from screener.symbols import tv_to_nse, tv_to_yf


INDIA_MIN_CRORE = 1000.0
US_MIN_USD = 1_000_000_000.0

# FMP US fundamentals: 24h cache, "fmp" circuit breaker. ``cache_ttl`` is
# overridden per-call below to honour the screen's --cache-ttl flag.
_FMP_US_PROVIDER = CachedProvider(
    ProviderSpec(provider="fmp", namespace="garp_fmp_us", ttl_seconds=86400)
)


class GarpThresholds(BaseModel):
    market_cap_min: float = Field(ge=0.0)
    sales_min: float = Field(ge=0.0)
    peg_max: float = Field(default=2.0, gt=0.0)
    sales_growth_5y_min: float = Field(default=15.0)
    operating_profit_growth_min: float = Field(default=10.0)
    eps_growth_5y_min: float = Field(default=12.0)
    roe_5y_min: float = Field(default=15.0)
    roce_or_roic_min: float = Field(default=15.0)

    model_config = ConfigDict(frozen=True)


INDIA_THRESHOLDS = GarpThresholds(
    market_cap_min=INDIA_MIN_CRORE,
    sales_min=INDIA_MIN_CRORE,
)
US_THRESHOLDS = GarpThresholds(market_cap_min=US_MIN_USD, sales_min=US_MIN_USD)


class GarpFundamentals(BaseModel):
    """Canonical provider-independent fundamentals used by GARP scoring."""

    name: str = ""
    description: str = ""
    market_cap: float | None = None
    sales: float | None = None
    peg: float | None = None
    sales_growth_5y: float | None = None
    operating_profit_growth: float | None = None
    eps_growth_5y: float | None = None
    roe_5y: float | None = None
    roce_or_roic: float | None = None
    expected_quarterly_profit: float | None = None
    profit_3q_back: float | None = None
    quarterly_profit_growth: float | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __getitem__(self, key: str) -> object:
        """Keep internal adapter consumers source-compatible with row indexing."""
        return getattr(self, key)


NormalizedGarpRow = GarpFundamentals


class GarpFundamentalsAdapter(Protocol):
    """Load one symbol and return the normalized row shape used by GARP."""

    @property
    def thresholds(self) -> GarpThresholds: ...

    def load_row(
        self,
        symbol: str,
        description: str | None,
        *,
        cache_ttl: float | None,
        refresh: bool,
    ) -> NormalizedGarpRow | None: ...


def _cagr(latest: float | None, oldest: float | None, years: float) -> float | None:
    if latest is None or oldest is None or latest <= 0 or oldest <= 0 or years <= 0:
        return None
    return float(((latest / oldest) ** (1.0 / years) - 1.0) * 100.0)


def _series_from_statement(statement: pd.DataFrame, row_names: list[str]) -> pd.Series:
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    for name in row_names:
        if name in statement.index:
            statement_row = cast(pd.Series, statement.loc[name])
            return pd.to_numeric(statement_row, errors="coerce").dropna()
    return pd.Series(dtype=float)


def _average_ratio(
    numerator: pd.Series, denominator: pd.Series, periods: int
) -> float | None:
    if numerator.empty or denominator.empty:
        return None
    values = []
    for col in list(numerator.index)[:periods]:
        den = to_number(denominator.get(col))
        num = to_number(numerator.get(col))
        if num is not None and den not in (None, 0):
            values.append((num / den) * 100.0)
    if not values:
        return None
    return float(sum(values) / len(values))


def _coerce_garp_fundamentals(
    row: GarpFundamentals | Mapping[str, object],
) -> GarpFundamentals | None:
    if isinstance(row, GarpFundamentals):
        return row
    try:
        return GarpFundamentals.model_validate(row)
    except ValidationError:
        return None


def _passes_garp(
    row: GarpFundamentals | Mapping[str, object], thresholds: GarpThresholds
) -> bool:
    fundamentals = _coerce_garp_fundamentals(row)
    if fundamentals is None:
        return False
    market_cap = fundamentals.market_cap
    sales = fundamentals.sales
    peg = fundamentals.peg
    sales_growth = fundamentals.sales_growth_5y
    operating_growth = fundamentals.operating_profit_growth
    eps_growth = fundamentals.eps_growth_5y
    roe = fundamentals.roe_5y
    capital_return = fundamentals.roce_or_roic
    quarterly_growth = fundamentals.quarterly_profit_growth
    required = (
        market_cap,
        sales,
        peg,
        sales_growth,
        operating_growth,
        eps_growth,
        roe,
        capital_return,
        quarterly_growth,
    )
    if any(value is None for value in required):
        return False
    assert all(value is not None for value in required)
    return (
        market_cap > thresholds.market_cap_min
        and sales > thresholds.sales_min
        and 0 < peg < thresholds.peg_max
        and sales_growth > thresholds.sales_growth_5y_min
        and operating_growth > thresholds.operating_profit_growth_min
        and eps_growth > thresholds.eps_growth_5y_min
        and roe > thresholds.roe_5y_min
        and capital_return > thresholds.roce_or_roic_min
        and quarterly_growth > 0
    )


def add_garp_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    if scored.empty:
        scored["garp_score"] = []
        return scored

    def pct(col: str) -> pd.Series:
        return pd.to_numeric(scored[col], errors="coerce").rank(pct=True).fillna(0)

    peg = pd.to_numeric(scored["peg"], errors="coerce")
    # A negative (loss-making) or zero PEG is not a value signal: NaN it before
    # ranking so it flows through as a missing factor (fillna(0)) instead of
    # ranking lowest and earning a top inv_peg.
    peg = peg.where(peg > 0)
    inv_peg = (1 - peg.rank(pct=True)).fillna(0)
    scored["garp_score"] = (
        30 * inv_peg
        + 20 * pct("eps_growth_5y")
        + 15 * pct("sales_growth_5y")
        + 15 * pct("roe_5y")
        + 10 * pct("roce_or_roic")
        + 10 * pct("quarterly_profit_growth")
    ).round(2)
    return scored.sort_values("garp_score", ascending=False)


def load_garp_universe(
    market: str,
    universe_size: int,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    from tradingview_screener import col

    market_meta = get_market(market)
    if market == "india":
        filters = [
            col("type") == "stock",
            col("close") >= market_meta.screen_min_close,
            col("market_cap_basic") >= INDIA_MIN_CRORE,
        ]
    else:
        filters = [
            col("type") == "stock",
            col("close") >= market_meta.screen_min_close,
            col("market_cap_basic") >= US_MIN_USD,
        ]
    _total, df = scan(
        market=market,
        filters=filters,
        limit=universe_size,
        order_by="volume",
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return df


def _fetch_india_sections(symbol: str) -> dict[str, Any]:
    from openscreener import Stock

    stock = Stock(symbol)
    return {
        "ratios": stock.fetch("ratios") or {},
        "profit_loss": stock.fetch("profit_loss") or {},
        "quarterly_results": stock.fetch("quarterly_results") or {},
    }


def _india_row(
    symbol: str, description: str | None, payload: dict[str, Any]
) -> NormalizedGarpRow:
    ratios = cast(
        dict[str, Any],
        payload.get("ratios") if isinstance(payload.get("ratios"), dict) else {},
    )
    profit_loss = cast(
        dict[str, Any],
        payload.get("profit_loss")
        if isinstance(payload.get("profit_loss"), dict)
        else {},
    )
    metrics = {**profit_loss, **ratios}
    quarterly = cast(
        dict[str, Any],
        payload.get("quarterly_results")
        if isinstance(payload.get("quarterly_results"), dict)
        else {},
    )
    expected_q_np = first_number(
        ratios,
        "expected_quarterly_net_profit",
        "expected_quarterly_profit",
        "expected_net_profit",
    )
    np_3q_back = first_number(
        quarterly,
        "net_profit_3quarters_back",
        "net profit 3quarters back",
        "net_profit_3q_back",
    )
    return GarpFundamentals(
        name=symbol,
        description=description or "",
        market_cap=first_number(metrics, "market_capitalization", "market_cap"),
        sales=first_number(metrics, "sales", "sales_ttm", "revenue"),
        peg=first_number(metrics, "peg_ratio", "peg"),
        sales_growth_5y=first_number(metrics, "sales_growth_5years", "sales_growth_5y"),
        operating_profit_growth=first_number(
            metrics, "operating_profit_growth", "opm_growth"
        ),
        eps_growth_5y=first_number(metrics, "eps_growth_5years", "eps_growth_5y"),
        roe_5y=first_number(
            metrics, "average_return_on_equity_5years", "average_roe_5y"
        ),
        roce_or_roic=first_number(
            metrics,
            "average_return_on_capital_employed_3years",
            "average_roce_3y",
            "roce_percent",
        ),
        expected_quarterly_profit=expected_q_np,
        profit_3q_back=np_3q_back,
        quarterly_profit_growth=pct_change(expected_q_np, np_3q_back),
    )


class OpenScreenerGarpAdapter:
    """India fundamentals adapter backed by the OpenScreener section payloads."""

    @property
    def thresholds(self) -> GarpThresholds:
        return INDIA_THRESHOLDS

    def load_row(
        self,
        symbol: str,
        description: str | None,
        *,
        cache_ttl: float | None,
        refresh: bool,
    ) -> NormalizedGarpRow | None:
        payload = cached_json_call(
            "garp_india",
            ("india", symbol),
            ttl_seconds=cache_ttl,
            refresh=refresh,
            fetch=lambda: _fetch_india_sections(symbol),
        )
        if not isinstance(payload, dict):
            return None
        return _coerce_garp_fundamentals(_india_row(symbol, description, payload))


def _universe_items(universe: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (str(row["name"]), str(row.get("description") or ""))
        for _, row in universe.iterrows()
        if row.get("name")
    ]


def _screen_garp_with_adapter(
    universe: pd.DataFrame,
    *,
    adapter: GarpFundamentalsAdapter,
    limit: int,
    workers: int,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    rows = [
        row
        for row in parallel_map(
            lambda item: adapter.load_row(
                item[0],
                item[1],
                cache_ttl=cache_ttl,
                refresh=refresh,
            ),
            _universe_items(universe),
            max_workers=max(1, workers),
            on_error="skip",
        )
        if row is not None and _passes_garp(row, adapter.thresholds)
    ]
    return add_garp_score(pd.DataFrame(row.model_dump() for row in rows)).head(limit)


def screen_india_garp(
    universe: pd.DataFrame,
    *,
    limit: int,
    workers: int,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    return _screen_garp_with_adapter(
        universe,
        adapter=OpenScreenerGarpAdapter(),
        limit=limit,
        workers=workers,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )


def _us_row(symbol: str, description: str | None) -> NormalizedGarpRow:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    income = ticker.income_stmt
    estimates = ticker.earnings_estimate

    revenue = _series_from_statement(income, ["Total Revenue"])
    operating = _series_from_statement(
        income, ["Operating Income", "Operating Income As Reported"]
    )
    net_income = _series_from_statement(
        income, ["Net Income", "Net Income Common Stockholders"]
    )
    equity = _series_from_statement(income, ["Stockholders Equity", "Total Equity"])
    ebit = _series_from_statement(income, ["EBIT", "Operating Income"])
    tax_rate = _series_from_statement(income, ["Tax Rate For Calcs"])
    debt = pd.Series(dtype=float)
    try:
        balance = ticker.balance_sheet
        debt = _series_from_statement(balance, ["Total Debt"])
        equity = _series_from_statement(
            balance, ["Stockholders Equity", "Total Stockholder Equity"]
        )
    except Exception:
        balance = pd.DataFrame()

    quarterly_eps_growth = None
    expected_eps = None
    year_ago_eps = None
    if estimates is not None and not estimates.empty and "0q" in estimates.index:
        expected_eps = to_number(estimates.loc["0q"].get("avg"))
        year_ago_eps = to_number(estimates.loc["0q"].get("yearAgoEps"))
        quarterly_eps_growth = pct_change(expected_eps, year_ago_eps)

    latest_revenue = to_number(revenue.iloc[0]) if not revenue.empty else None
    oldest_revenue = (
        to_number(revenue.iloc[min(len(revenue) - 1, 4)]) if len(revenue) else None
    )
    latest_op = to_number(operating.iloc[0]) if not operating.empty else None
    old_op = (
        to_number(operating.iloc[min(len(operating) - 1, 1)])
        if len(operating)
        else None
    )
    latest_ni = to_number(net_income.iloc[0]) if not net_income.empty else None
    old_ni = (
        to_number(net_income.iloc[min(len(net_income) - 1, 4)])
        if len(net_income)
        else None
    )

    tax = to_number(tax_rate.iloc[0]) if not tax_rate.empty else 0.21
    nopat = ebit * (1.0 - float(tax or 0.21))
    invested_capital = debt.add(equity, fill_value=0)
    roic = _average_ratio(nopat, invested_capital, 3)

    return GarpFundamentals(
        name=symbol,
        description=description or info.get("shortName") or "",
        market_cap=to_number(info.get("marketCap")),
        sales=latest_revenue,
        peg=to_number(info.get("trailingPegRatio") or info.get("pegRatio")),
        sales_growth_5y=_cagr(latest_revenue, oldest_revenue, 4),
        operating_profit_growth=pct_change(latest_op, old_op),
        eps_growth_5y=_cagr(latest_ni, old_ni, 4),
        roe_5y=_average_ratio(net_income, equity, 5),
        roce_or_roic=roic,
        expected_quarterly_profit=expected_eps,
        profit_3q_back=year_ago_eps,
        quarterly_profit_growth=quarterly_eps_growth,
    )


# ── FMP fundamentals (US) ───────────────────────────────────────────────────
#
# The yfinance path above costs ~4 HTTP round-trips per ticker. When an
# FMP_API_KEY is configured we source the same inputs from FMP instead and
# cache the per-symbol payload on disk; yfinance remains the fallback when no
# key is set or FMP has no statement data for a symbol.


def _fmp_get(path: str, params: dict[str, Any], api_key: str) -> Any:
    return fmp_get(path, params, api_key)


def _fetch_fmp_us_sections(symbol: str, api_key: str) -> dict[str, Any] | None:
    return {
        "profile": _fmp_get(f"profile/{symbol}", {}, api_key),
        "ratios_ttm": _fmp_get(f"ratios-ttm/{symbol}", {}, api_key),
        "income_annual": _fmp_get(
            f"income-statement/{symbol}",
            {"period": "annual", "limit": 5},
            api_key,
        ),
        "balance_annual": _fmp_get(
            f"balance-sheet-statement/{symbol}",
            {"period": "annual", "limit": 5},
            api_key,
        ),
        "income_quarterly": _fmp_get(
            f"income-statement/{symbol}",
            {"period": "quarter", "limit": 5},
            api_key,
        ),
        # FMP sorts estimates descending by date (farthest future first),
        # so a small limit would drop the nearest upcoming quarter.
        "estimates_quarterly": _fmp_get(
            f"analyst-estimates/{symbol}",
            {"period": "quarter", "limit": 40},
            api_key,
        ),
    }


def _fetch_fmp_us_cached(
    symbol: str,
    api_key: str,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> dict[str, Any] | None:
    return _FMP_US_PROVIDER.fetch(
        ("us", symbol),
        lambda: _fetch_fmp_us_sections(symbol, api_key),
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=f"garp fundamentals {symbol}",
    )


def _fmp_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def _fmp_series(statements: list[dict[str, Any]], field: str) -> pd.Series:
    """Build a newest-first series keyed by statement date (FMP order)."""
    data: dict[str, float] = {}
    for entry in statements:
        date = entry.get("date")
        value = to_number(entry.get(field))
        if date and value is not None and str(date) not in data:
            data[str(date)] = value
    return pd.Series(data, dtype=float)


def _fmp_quarterly_eps(
    estimates: list[dict[str, Any]], quarterly_income: list[dict[str, Any]]
) -> tuple[float | None, float | None]:
    """Mirror yfinance's earnings_estimate ``0q`` row.

    Expected EPS is the average analyst estimate for the first unreported
    quarter; year-ago EPS is the actual EPS from the reported quarter ending
    closest to one year before that estimate date.
    """
    expected_eps: float | None = None
    expected_ts: pd.Timestamp | None = None
    latest_reported = (
        pd.to_datetime(cast(str, quarterly_income[0].get("date")), errors="coerce")
        if quarterly_income
        else pd.NaT
    )
    if not pd.isna(latest_reported):
        upcoming: list[tuple[pd.Timestamp, float]] = []
        for entry in estimates:
            ts = pd.to_datetime(cast(str, entry.get("date")), errors="coerce")
            eps = first_number(entry, "estimatedEpsAvg", "epsAvg")
            if not pd.isna(ts) and ts > latest_reported and eps is not None:
                upcoming.append((ts, eps))
        if upcoming:
            expected_ts, expected_eps = min(upcoming)

    # Pair by date (estimate date minus one year), not by list position:
    # fiscal calendars shift and FMP statement lists can have gaps.
    year_ago_eps: float | None = None
    if expected_ts is not None:
        target = expected_ts - pd.Timedelta(days=365)
        best: tuple[float, float] | None = None
        for entry in quarterly_income:
            ts = pd.to_datetime(cast(str, entry.get("date")), errors="coerce")
            eps = to_number(entry.get("eps"))
            if pd.isna(ts) or eps is None:
                continue
            delta = abs(float((ts - target).days))
            if delta <= 60 and (best is None or delta < best[0]):
                best = (delta, eps)
        if best is not None:
            year_ago_eps = best[1]
    return expected_eps, year_ago_eps


def _fmp_us_row(
    symbol: str, description: str | None, payload: dict[str, Any]
) -> NormalizedGarpRow | None:
    """Map an FMP payload to the same row shape ``_us_row`` produces.

    Returns ``None`` when FMP has no annual statements for the symbol so the
    caller can fall back to yfinance.
    """
    if not isinstance(payload, dict):
        return None
    income = _fmp_list(payload, "income_annual")
    if not income:
        return None
    profile_rows = _fmp_list(payload, "profile")
    profile = profile_rows[0] if profile_rows else {}
    ratios_rows = _fmp_list(payload, "ratios_ttm")
    ratios = ratios_rows[0] if ratios_rows else {}
    balance = _fmp_list(payload, "balance_annual")
    quarterly = _fmp_list(payload, "income_quarterly")
    estimates = _fmp_list(payload, "estimates_quarterly")

    revenue = _fmp_series(income, "revenue")
    operating = _fmp_series(income, "operatingIncome")
    net_income = _fmp_series(income, "netIncome")
    equity = _fmp_series(balance, "totalStockholdersEquity")
    debt = _fmp_series(balance, "totalDebt")
    ebit = operating

    tax: float | None = None
    tax_expense = to_number(income[0].get("incomeTaxExpense"))
    pretax = to_number(income[0].get("incomeBeforeTax"))
    if tax_expense is not None and pretax not in (None, 0):
        tax = tax_expense / float(pretax or 1.0)
    nopat = ebit * (1.0 - float(tax or 0.21))
    invested_capital = debt.add(equity, fill_value=0)
    roic = _average_ratio(nopat, invested_capital, 3)

    latest_revenue = to_number(revenue.iloc[0]) if not revenue.empty else None
    oldest_revenue = (
        to_number(revenue.iloc[min(len(revenue) - 1, 4)]) if len(revenue) else None
    )
    latest_op = to_number(operating.iloc[0]) if not operating.empty else None
    old_op = (
        to_number(operating.iloc[min(len(operating) - 1, 1)])
        if len(operating)
        else None
    )
    latest_ni = to_number(net_income.iloc[0]) if not net_income.empty else None
    old_ni = (
        to_number(net_income.iloc[min(len(net_income) - 1, 4)])
        if len(net_income)
        else None
    )

    expected_eps, year_ago_eps = _fmp_quarterly_eps(estimates, quarterly)

    return GarpFundamentals(
        name=symbol,
        description=description or str(profile.get("companyName") or ""),
        market_cap=first_number(profile, "mktCap", "marketCap"),
        sales=latest_revenue,
        peg=first_number(ratios, "priceEarningsToGrowthRatioTTM", "pegRatioTTM"),
        sales_growth_5y=_cagr(latest_revenue, oldest_revenue, 4),
        operating_profit_growth=pct_change(latest_op, old_op),
        eps_growth_5y=_cagr(latest_ni, old_ni, 4),
        roe_5y=_average_ratio(net_income, equity, 5),
        roce_or_roic=roic,
        expected_quarterly_profit=expected_eps,
        profit_3q_back=year_ago_eps,
        quarterly_profit_growth=pct_change(expected_eps, year_ago_eps),
    )


class YFinanceGarpAdapter:
    """US fundamentals fallback adapter backed by yfinance."""

    @property
    def thresholds(self) -> GarpThresholds:
        return US_THRESHOLDS

    def load_row(
        self,
        symbol: str,
        description: str | None,
        *,
        cache_ttl: float | None,
        refresh: bool,
    ) -> NormalizedGarpRow | None:
        del cache_ttl, refresh
        return _coerce_garp_fundamentals(_us_row(symbol, description))


@dataclass(frozen=True)
class FmpGarpAdapter:
    """US fundamentals adapter backed by cached FMP payloads."""

    api_key: str

    @property
    def thresholds(self) -> GarpThresholds:
        return US_THRESHOLDS

    def load_row(
        self,
        symbol: str,
        description: str | None,
        *,
        cache_ttl: float | None,
        refresh: bool,
    ) -> NormalizedGarpRow | None:
        payload = _fetch_fmp_us_cached(
            symbol, self.api_key, cache_ttl=cache_ttl, refresh=refresh
        )
        if not isinstance(payload, dict):
            return None
        row = _fmp_us_row(symbol, description, payload)
        return _coerce_garp_fundamentals(row) if row is not None else None


@dataclass(frozen=True)
class UsGarpFundamentalsAdapter:
    """US source-selection adapter: FMP first, yfinance fallback."""

    fmp: FmpGarpAdapter | None = None
    yfinance: YFinanceGarpAdapter = field(default_factory=YFinanceGarpAdapter)

    @property
    def thresholds(self) -> GarpThresholds:
        return US_THRESHOLDS

    def load_row(
        self,
        symbol: str,
        description: str | None,
        *,
        cache_ttl: float | None,
        refresh: bool,
    ) -> NormalizedGarpRow | None:
        if self.fmp is not None:
            row = self.fmp.load_row(
                symbol, description, cache_ttl=cache_ttl, refresh=refresh
            )
            if row is not None:
                return row
        return self.yfinance.load_row(
            symbol, description, cache_ttl=cache_ttl, refresh=refresh
        )


def _us_fundamentals_adapter() -> UsGarpFundamentalsAdapter:
    api_key = resolve_api_key()
    return UsGarpFundamentalsAdapter(FmpGarpAdapter(api_key) if api_key else None)


def screen_us_garp(
    universe: pd.DataFrame,
    *,
    limit: int,
    workers: int,
    cache_ttl: float | None = 86400,
    refresh: bool = False,
) -> pd.DataFrame:
    return _screen_garp_with_adapter(
        universe,
        adapter=_us_fundamentals_adapter(),
        limit=limit,
        workers=workers,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )


def load_garp_fundamentals(
    symbol: str,
    market: str,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> GarpFundamentals | None:
    """Load validated per-symbol fundamentals through the canonical adapters.

    Composes the same fetch + map path the market screens use, for one symbol:

    * India — the cached openscreener sections (``garp_india`` namespace) mapped
      through :func:`_india_row`.
    * US — the cached FMP payload mapped through :func:`_fmp_us_row` when an
      ``FMP_API_KEY`` is configured, falling back to the yfinance
      :func:`_us_row` when no key is set or FMP has no statement data.

    Returns ``None`` when no fundamentals are available. Cache namespaces,
    keys, TTLs and network calls are identical to the screen path.
    """
    if market == "india":
        sym = tv_to_nse(symbol, strip_suffix=True)
        return OpenScreenerGarpAdapter().load_row(
            sym, "", cache_ttl=cache_ttl, refresh=refresh
        )
    yf_sym = tv_to_yf(symbol, market)
    return _us_fundamentals_adapter().load_row(
        yf_sym, "", cache_ttl=cache_ttl, refresh=refresh
    )


def load_garp_row(
    symbol: str,
    market: str,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> dict[str, Any] | None:
    """Compatibility dict serialization of :func:`load_garp_fundamentals`."""
    fundamentals = load_garp_fundamentals(
        symbol,
        market,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    normalized = (
        _coerce_garp_fundamentals(fundamentals) if fundamentals is not None else None
    )
    return normalized.model_dump() if normalized is not None else None


def run_garp_screen(
    market: str,
    universe_size: int,
    *,
    limit: int,
    workers: int,
    cache_ttl: float | None,
    refresh: bool,
    on_universe: Callable[[pd.DataFrame], None] = lambda _df: None,
) -> pd.DataFrame | None:
    """Run the full GARP pipeline and return the scored results.

    Loads the liquid universe, enriches it with market-specific fundamentals
    and applies the GARP filter + score. ``on_universe`` is called with the
    loaded universe before enrichment so the command layer can emit its
    progress line (and route it to stdout/stderr as needed). Returns ``None``
    when the base universe scan yields nothing (distinct from an empty result
    after filtering), leaving rendering to the caller.
    """
    universe = load_garp_universe(
        market,
        int(universe_size),
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    if universe.empty:
        return None

    on_universe(universe)
    if market == "india":
        return screen_india_garp(
            universe,
            limit=int(limit),
            workers=int(workers),
            cache_ttl=cache_ttl,
            refresh=refresh,
        )
    return screen_us_garp(
        universe,
        limit=int(limit),
        workers=int(workers),
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
