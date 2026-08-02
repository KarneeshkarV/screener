import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from tradingview_screener import Query
import pandas as pd

from screener.markets import TV_MARKETS
from screener.providers import CachedProvider, FrameWithMeta, ProviderSpec


LOG = logging.getLogger(__name__)

# The scan payload is a frame plus TradingView's total match count, which the
# frame cannot carry, so the entry is a parquet plus a JSON sidecar that expire
# together. Routing it through the seam (instead of hand-wiring cache lookup +
# resilience here) is what gives the scanner stale-serve on a provider outage.
SCANNER_PROVIDER: CachedProvider = CachedProvider(
    ProviderSpec(
        provider="tradingview",
        namespace="tradingview_scanner",
        ttl_seconds=900,
        kind="frame_meta",
    )
)


MARKETS = TV_MARKETS

DEFAULT_COLUMNS = [
    "name",
    "description",
    "close",
    "change",
    "volume",
    "market_cap_basic",
]

SETUP_SCORE_COLUMNS = [
    "EMA5",
    "EMA20",
    "EMA100",
    "EMA200",
    "RSI",
]

DETAIL_COLUMNS = [
    "price_earnings_ttm",
    "return_on_equity",
    "dividend_yield_recent",
    "debt_to_equity",
    "RSI",
]


@dataclass(frozen=True)
class ScannerPlan:
    market: str
    filters: list[Any]
    columns: list[str]
    order_by: str
    query_order_by: str
    fetch_limit: int


def build_scanner_plan(
    *,
    market: str,
    filters: list[Any],
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
) -> ScannerPlan:
    columns = list(DEFAULT_COLUMNS)
    if detail:
        columns.extend(DETAIL_COLUMNS)

    if order_by == "setup_score":
        columns.extend(c for c in SETUP_SCORE_COLUMNS if c not in columns)
        fetch_limit = max(limit * 10, 500)
        query_order_by = "volume"
    else:
        fetch_limit = max(limit * 3, 100)
        query_order_by = order_by

    return ScannerPlan(
        market=market,
        filters=filters,
        columns=columns,
        order_by=order_by,
        query_order_by=query_order_by,
        fetch_limit=fetch_limit,
    )


class TradingViewScannerAdapter:
    """Adapter for TradingView query construction, cache keys, and resilience."""

    def fetch(
        self,
        plan: ScannerPlan,
        *,
        cache_ttl: float | None = 900,
        refresh: bool = False,
    ) -> tuple[int, pd.DataFrame]:
        query = (
            Query()
            .set_markets(MARKETS[plan.market])
            .select(*plan.columns)
            .where(*plan.filters)
            .order_by(plan.query_order_by, ascending=False)
            .limit(plan.fetch_limit)
        )

        return get_scanner_data_cached(
            query,
            key_parts=(
                "scanner",
                plan.market,
                # Query.where() joins filters with AND, so their order does not
                # change TradingView semantics — sort so semantically identical
                # filter lists hash to the same cache key.
                sorted(repr(f) for f in plan.filters),
                plan.columns,
                plan.order_by,
                plan.fetch_limit,
            ),
            columns=plan.columns,
            cache_ttl=cache_ttl,
            refresh=refresh,
        )


TRADINGVIEW_SCANNER = TradingViewScannerAdapter()


def get_scanner_data_cached(
    query: Query,
    *,
    key_parts: object,
    columns: list[str],
    operation: str = "scanner data",
    cache_ttl: float | None = 900,
    refresh: bool = False,
) -> tuple[int, pd.DataFrame]:
    def fetch() -> FrameWithMeta:
        count, df = query.get_scanner_data()
        return df, {"count": int(count)}

    result: FrameWithMeta | None = SCANNER_PROVIDER.fetch(
        key_parts,
        fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=operation,
    )
    if result is None:
        # No live data and no cache entry to fall back on. A stale entry would
        # already have been served (with its own warning) by the provider.
        LOG.warning(
            "tradingview scan failed for %s; returning empty results "
            "(not cached) - rerun with --refresh once connectivity is back",
            operation,
        )
        return 0, pd.DataFrame(columns=columns)
    frame, meta = result
    return int(meta.get("count", 0)), frame


def _percentile(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True).fillna(0)


def _log_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return _percentile(values.add(1).map(math.log))


def _add_setup_score(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    ema5 = pd.to_numeric(df["EMA5"], errors="coerce")
    ema20 = pd.to_numeric(df["EMA20"], errors="coerce")
    ema100 = pd.to_numeric(df["EMA100"], errors="coerce")
    ema200 = pd.to_numeric(df["EMA200"], errors="coerce")
    change = pd.to_numeric(df["change"], errors="coerce")
    rsi = pd.to_numeric(df["RSI"], errors="coerce")

    dollar_volume = pd.to_numeric(df["volume"], errors="coerce") * close
    liquidity = _log_percentile(dollar_volume)
    market_cap = _log_percentile(df["market_cap_basic"])

    trend_spread = (
        ((ema5 - ema20) / close)
        + ((ema20 - ema100) / close)
        + ((ema100 - ema200) / close)
    ).clip(lower=0, upper=0.35)
    trend_strength = _percentile(trend_spread)

    momentum = ((change.clip(lower=-5, upper=10) + 5) / 15).fillna(0)
    rsi_quality = (1 - ((rsi - 60).abs() / 40)).clip(lower=0, upper=1).fillna(0)
    price_quality = _percentile(close.clip(lower=0, upper=200))

    extension = ((close - ema20) / ema20).fillna(0)
    overextension_penalty = ((extension - 0.12).clip(lower=0) / 0.25).clip(upper=1)

    setup_score = (
        25 * liquidity
        + 30 * trend_strength
        + 15 * momentum
        + 15 * market_cap
        + 10 * rsi_quality
        + 5 * price_quality
        - 15 * overextension_penalty
    ).round(2)
    return df.assign(setup_score=setup_score)


def _dedupe_listings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "description" not in df.columns:
        return df

    deduped = df.copy()
    fallback = deduped["name"] if "name" in deduped.columns else deduped["ticker"]
    company = (
        deduped["description"]
        .fillna("")
        .where(
            deduped["description"].fillna("").str.strip() != "",
            fallback.fillna(""),
        )
    )
    deduped["_listing_key"] = company.map(
        lambda value: re.sub(r"[^a-z0-9]+", "", str(value).lower())
    )
    deduped = deduped.drop_duplicates("_listing_key", keep="first")
    return deduped.drop(columns=["_listing_key"])


def shape_scan_results(
    df: pd.DataFrame,
    *,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
) -> pd.DataFrame:
    """Shape raw scanner rows after Adapter fetch without provider access."""
    shaped = df
    if order_by == "setup_score" and not shaped.empty:
        shaped = _add_setup_score(shaped)
        shaped = shaped.sort_values("setup_score", ascending=False)
        hidden_score_columns = [
            col
            for col in SETUP_SCORE_COLUMNS
            if not detail or col not in DETAIL_COLUMNS
        ]
        shaped = shaped.drop(columns=hidden_score_columns)
    if not shaped.empty:
        shaped = _dedupe_listings(shaped).head(limit)
    return shaped


def scan(
    market: str,
    filters: list,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
    cache_ttl: float | None = 900,
    refresh: bool = False,
) -> tuple[int, pd.DataFrame]:
    plan = build_scanner_plan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
    )
    count, df = TRADINGVIEW_SCANNER.fetch(
        plan,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return count, shape_scan_results(df, limit=limit, order_by=order_by, detail=detail)
