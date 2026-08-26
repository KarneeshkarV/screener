import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tradingview_screener import Query

from screener.cache import frame_meta_paths
from screener.markets import TV_MARKETS
from screener.providers import CachedProvider, FrameWithMeta, ProviderSpec
from screener.resilience import RetryConfig
from screener.scoring import (
    DEFAULT_PRICE_ADJUSTMENT,
    OUTPUT_SCORE_COLUMN,
    PriceAdjustment,
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
# Named separately from the spec because the sidecar path is resolved from it
# directly: tests swap SCANNER_PROVIDER for a FakeProvider whose ``spec`` is
# None, and reading the namespace through the provider would break under the
# double.
SCANNER_NAMESPACE = "tradingview_scanner"

SCANNER_PROVIDER: CachedProvider = CachedProvider(
    ProviderSpec(
        provider="tradingview",
        namespace=SCANNER_NAMESPACE,
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
        # Over-fetch so the rows lost before ``.head(limit)`` still leave
        # more than ``limit`` to rank. Three cuts happen after the fetch: the
        # recipe's eligibility floor, names whose price fetch came back empty,
        # and ``_dedupe_listings`` collapsing NSE/BSE dual listings. What that
        # headroom costs depends on the recipe. A snapshot recipe ranks on
        # columns the one TradingView request already returned, so spare rows
        # are free. A bar-derived recipe downloads daily OHLCV per surviving
        # ticker, so every spare row is another network fetch. 5x covers the
        # three drops. The snapshot path can afford 10x.
        if active_scorer.bar_score is not None:
            fetch_limit = max(limit * 5, 200)
        else:
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
        timeout: float | None = None,
        retries: int | None = None,
        strict: bool = False,
    ) -> tuple[int, pd.DataFrame, datetime]:
        query = (
            Query()
            .set_markets(MARKETS[plan.market])
            .select(*plan.columns)
            .where(*plan.filters)
            .order_by(plan.query_order_by, ascending=False)
            .limit(plan.fetch_limit)
        )

        return _scanner_entry(
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
            timeout=timeout,
            retries=retries,
            strict=strict,
        )


TRADINGVIEW_SCANNER = TradingViewScannerAdapter()


# Metadata sidecar key holding the wall-clock time the payload was fetched
# from TradingView. It rides the frame_meta JSON so it survives the parquet
# round-trip: a cached frame's ``as_of`` stays its original fetch time.
FETCHED_AT_META_KEY = "fetched_at"


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _fetched_at_from_meta(
    meta: dict[str, Any], sidecar: Path | None = None
) -> datetime:
    """Recover the fetch timestamp from a frame_meta sidecar.

    Falls back to the sidecar file's mtime, NOT to "now". Entries written
    before this key existed carry no timestamp, and "now" would have called
    them fresh. That is safe on the TTL-gated path, where a served entry really
    is within the TTL, but wrong on the stale path: ``_read_stale`` ignores the
    TTL by design, so an arbitrarily old legacy frame would have reported
    ``as_of=now`` and looked freshly fetched to a caller sizing real orders.
    The sidecar is written in the same breath as the payload, so its mtime is
    the fetch time for both paths.

    A naive timestamp is refused for the same reason: comparing it against an
    aware "now" either raises or silently comes out wrong, and a cache file is
    not a trustworthy enough source to guess an offset for.
    """
    raw = meta.get(FETCHED_AT_META_KEY)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            return parsed.astimezone(UTC)
    if sidecar is not None:
        try:
            return datetime.fromtimestamp(sidecar.stat().st_mtime, UTC)
        except OSError:
            pass
    return _now_utc()


def get_scanner_data_cached(
    query: Query,
    *,
    key_parts: object,
    columns: list[str],
    operation: str = "scanner data",
    cache_ttl: float | None = 900,
    refresh: bool = False,
    timeout: float | None = None,
    retries: int | None = None,
    strict: bool = False,
) -> tuple[int, pd.DataFrame]:
    """Scanner fetch without the fetch timestamp; see :func:`_scanner_entry`."""
    count, df, _as_of = _scanner_entry(
        query,
        key_parts=key_parts,
        columns=columns,
        operation=operation,
        cache_ttl=cache_ttl,
        refresh=refresh,
        timeout=timeout,
        retries=retries,
        strict=strict,
    )
    return count, df


def _scanner_entry(
    query: Query,
    *,
    key_parts: object,
    columns: list[str],
    operation: str = "scanner data",
    cache_ttl: float | None = 900,
    refresh: bool = False,
    timeout: float | None = None,
    retries: int | None = None,
    strict: bool = False,
) -> tuple[int, pd.DataFrame, datetime]:
    def fetch() -> FrameWithMeta:
        # Forward only an explicit timeout: requests treats ``timeout=None``
        # the same as absent, and passing nothing keeps stub queries that
        # declare plain ``get_scanner_data(self)`` working.
        request_kwargs: dict[str, Any] = (
            {"timeout": timeout} if timeout is not None else {}
        )
        count, df = query.get_scanner_data(**request_kwargs)
        return df, {
            "count": int(count),
            FETCHED_AT_META_KEY: _now_utc().isoformat(),
        }

    # The real wall-clock budget for a hung socket is attempts x timeout, so
    # the caller can cap either side independently.
    retry = RetryConfig(attempts=retries) if retries is not None else None
    result: FrameWithMeta | None = SCANNER_PROVIDER.fetch(
        key_parts,
        fetch,
        refresh=refresh,
        fallback=None,
        ttl_seconds=cache_ttl,
        operation=operation,
        retry=retry,
        strict=strict,
    )
    if result is None:
        # No live data and no cache entry to fall back on. A stale entry would
        # already have been served (with its own warning) by the provider -
        # unless strict mode raised first.
        LOG.warning(
            "tradingview scan failed for %s; returning empty results "
            "(not cached) - rerun with --refresh once connectivity is back",
            operation,
        )
        # Nothing was fetched, so there is no honest fetch time; "now" marks
        # the moment this empty payload was assembled.
        return 0, pd.DataFrame(columns=columns), _now_utc()
    frame, meta = result
    _, meta_path = frame_meta_paths(SCANNER_NAMESPACE, key_parts)
    return (
        int(meta.get("count", 0)),
        frame,
        _fetched_at_from_meta(meta, meta_path),
    )


# Re-export for correctness tests that still import from scanner.
_log_percentile = log_percentile


def _add_setup_score(
    df: pd.DataFrame,
    scorer: ScoreSpec | None = None,
    *,
    market: str | None = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
) -> pd.DataFrame:
    """Apply a ranking recipe and write ``setup_score``.

    Defaults to the EMA trend setup for backward compatibility with tests and
    call sites that still invoke this helper directly. ``market`` and
    ``refresh`` are only read by bar-derived recipes, which resolve price
    history for the scanned rows; ``refresh`` bypasses the on-disk bar cache so
    ``--refresh`` does not rank fresh snapshot rows on stale price history.
    ``price_adjustment`` is bar-only too. It must match the backtest's
    ``--price-adjustment`` so both sides score the same closes.
    """
    active = scorer if scorer is not None else default_scorer()
    return apply_score(
        df,
        active,
        market=market,
        refresh=refresh,
        price_adjustment=price_adjustment,
    )


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
    helpers = list(scorer.columns)
    # A bar-derived recipe also writes its raw value under ``aux_column`` (the
    # number the backtester ranks on) beside the 0-100 ``setup_score``. That is
    # a diagnostic, not a display column, so it rides along only under
    # ``--detail``; otherwise it would push the table past the width at which
    # ``display`` starts hiding ``description``.
    aux = scorer.bar_score.aux_column if scorer.bar_score is not None else None
    if aux is not None and not detail:
        helpers.append(aux)
    return [col for col in helpers if col not in keep]


def shape_scan_results(
    df: pd.DataFrame,
    *,
    limit: int = 50,
    order_by: str = "volume",
    detail: bool = False,
    scorer: ScoreSpec | None = None,
    market: str | None = None,
    refresh: bool = False,
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
) -> pd.DataFrame:
    """Shape raw scanner rows after Adapter fetch without provider access.

    Scoring happens here, *after* the TradingView filters have already cut the
    field, so a bar-derived recipe only ever fetches price history for the rows
    the scan returned rather than for the whole market.
    """
    shaped = df
    if order_by == OUTPUT_SCORE_COLUMN and not shaped.empty:
        active = scorer if scorer is not None else default_scorer()
        shaped = _add_setup_score(
            shaped,
            active,
            market=market,
            refresh=refresh,
            price_adjustment=price_adjustment,
        )
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
    price_adjustment: PriceAdjustment = DEFAULT_PRICE_ADJUSTMENT,
    timeout: float | None = None,
    retries: int | None = None,
    strict: bool = False,
) -> tuple[int, pd.DataFrame, datetime]:
    """Run one scan and return ``(total_matches, frame, as_of)``.

    ``as_of`` is when the payload was fetched from TradingView, not when this
    call returned - a cache hit reports the original fetch time. ``timeout``
    caps each underlying request (forwarded to ``requests.post``);
    ``retries`` overrides the resilience retry attempts for this scan.
    ``strict=True`` raises :class:`StaleDataError` instead of serving stale
    cache when the live fetch fails.
    """
    plan = build_scanner_plan(
        market=market,
        filters=filters,
        limit=limit,
        order_by=order_by,
        detail=detail,
        scorer=scorer,
    )
    count, df, as_of = TRADINGVIEW_SCANNER.fetch(
        plan,
        cache_ttl=cache_ttl,
        refresh=refresh,
        timeout=timeout,
        retries=retries,
        strict=strict,
    )
    shaped = shape_scan_results(
        df,
        limit=limit,
        order_by=order_by,
        detail=detail,
        scorer=plan.scorer,
        market=market,
        refresh=refresh,
        price_adjustment=price_adjustment,
    )
    return count, shaped, as_of
