import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tradingview_screener import Query

from screener.markets import TV_MARKETS
from screener.providers import CachedProvider, FrameWithMeta, ProviderSpec
from screener.scoring import (
    OUTPUT_SCORE_COLUMN,
    ScoreSpec,
    apply_score,
    default_scorer,
    get_scorer,
)
from screener.scoring.components import log_percentile

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

# Backward-compatible alias: columns the legacy EMA setup score fetches.
SETUP_SCORE_COLUMNS = list(get_scorer("ema").columns)

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
    scorer: ScoreSpec | None = None


def build_scanner_plan(
    *,
    market: str,
    filters: list[Any],
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
    scorer: ScoreSpec | None = None,
) -> ScannerPlan:
    columns = list(DEFAULT_COLUMNS)
    if detail:
        columns.extend(c for c in DETAIL_COLUMNS if c not in columns)

    active_scorer: ScoreSpec | None = None
    if order_by == OUTPUT_SCORE_COLUMN:
        active_scorer = scorer if scorer is not None else default_scorer()
        columns.extend(c for c in active_scorer.columns if c not in columns)
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
        scorer=active_scorer,
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


# Re-export for correctness tests that still import from scanner.
_log_percentile = log_percentile


def _add_setup_score(
    df: pd.DataFrame,
    scorer: ScoreSpec | None = None,
    *,
    market: str | None = None,
) -> pd.DataFrame:
    """Apply a ranking recipe and write ``setup_score``.

    Defaults to the EMA trend setup for backward compatibility with tests and
    call sites that still invoke this helper directly. ``market`` is only read
    by bar-derived recipes, which resolve price history for the scanned rows.
    """
    active = scorer if scorer is not None else default_scorer()
    return apply_score(df, active, market=market)


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


def _helper_columns_to_drop(
    scorer: ScoreSpec,
    *,
    detail: bool,
) -> list[str]:
    """Drop score-only helper columns; keep DEFAULT / DETAIL display columns."""
    keep = set(DEFAULT_COLUMNS)
    if detail:
        keep.update(DETAIL_COLUMNS)
    keep.add(OUTPUT_SCORE_COLUMN)
    return [col for col in scorer.columns if col not in keep]


def shape_scan_results(
    df: pd.DataFrame,
    *,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
    scorer: ScoreSpec | None = None,
    market: str | None = None,
) -> pd.DataFrame:
    """Shape raw scanner rows after Adapter fetch without provider access.

    Scoring happens here, *after* the TradingView filters have already cut the
    field, so a bar-derived recipe only ever fetches price history for the rows
    the scan returned rather than for the whole market.
    """
    shaped = df
    if order_by == OUTPUT_SCORE_COLUMN and not shaped.empty:
        active = scorer if scorer is not None else default_scorer()
        shaped = _add_setup_score(shaped, active, market=market)
        shaped = shaped.sort_values(OUTPUT_SCORE_COLUMN, ascending=False)
        drop_cols = _helper_columns_to_drop(active, detail=detail)
        shaped = shaped.drop(columns=[c for c in drop_cols if c in shaped.columns])
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
    scorer: ScoreSpec | None = None,
) -> tuple[int, pd.DataFrame]:
    plan = build_scanner_plan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
        scorer=scorer,
    )
    count, df = TRADINGVIEW_SCANNER.fetch(
        plan,
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return count, shape_scan_results(
        df,
        limit=limit,
        order_by=order_by,
        detail=detail,
        scorer=plan.scorer,
        market=market,
    )
