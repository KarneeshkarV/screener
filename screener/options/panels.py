"""Daily options panels built on the repository's canonical snapshot store."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from screener.cache import append_panel_snapshot, panel_path, read_frame
from screener.options import contract_store
from screener.options.metrics import compute_chain_metrics
from screener.options.models import OptionChain, OptionsMarket
from screener.options.nse_bhavcopy import BhavcopyFetcher, load_bhavcopy_chains
from screener.options.provider import OptionsProvider, default_us_provider
from screener.parallel import parallel_map
from screener.symbols import tv_to_nse
from screener.unusual_volume.nse_client import is_trading_day

LOG = logging.getLogger(__name__)

OPTIONS_PANEL_NAMES = {
    "us": "options_metrics_us",
    "india": "options_metrics_india",
}
PANEL_DEDUPE_KEYS = ["as_of", "SYMBOL"]

# Additive panel columns derived from *multiple* intraday snapshots in the
# contract store (Phase 3.4). They are NaN on the legacy EOD path and whenever
# only a single snapshot exists for a session, and populated only when the
# store holds two or more snapshots for the underlying that day. Kept in sync
# with ``PANEL_DERIVED_EXPRESSION_FIELDS`` in :mod:`screener.options.backtest`.
INTRADAY_PANEL_FIELDS: tuple[str, ...] = (
    "oi_change_intraday",
    "iv_change_intraday",
    "pcr_volume_intraday",
)
ProgressCallback = Callable[[date, int], None]
ErrorCallback = Callable[[date, Exception], None]


@dataclass(frozen=True)
class SnapshotResult:
    panel: pd.DataFrame
    chains: tuple[OptionChain, ...]
    requested: int
    missing: tuple[str, ...]


def metrics_row(chain: OptionChain) -> dict[str, object]:
    """Flatten one chain's metrics into the stable panel schema."""
    metrics = compute_chain_metrics(chain)
    row = metrics.model_dump()
    row["as_of"] = pd.Timestamp(metrics.as_of.date())
    row["SYMBOL"] = row.pop("underlying")
    row["support_strikes"] = json.dumps(list(metrics.support_strikes))
    row["resistance_strikes"] = json.dumps(list(metrics.resistance_strikes))
    row["front_expiry"] = (
        metrics.front_expiry.isoformat() if metrics.front_expiry else None
    )
    row["next_expiry"] = (
        metrics.next_expiry.isoformat() if metrics.next_expiry else None
    )
    row["options_volume"] = metrics.call_volume + metrics.put_volume
    # Intraday-derived columns are part of the panel schema on every path; they
    # stay null unless a store reduction fills them from ≥2 snapshots.
    for field in INTRADAY_PANEL_FIELDS:
        row[field] = None
    return row


def _intraday_fields(chains: list[OptionChain]) -> dict[str, float | None]:
    """Derive the Phase 3.4 intraday columns from a session's snapshot chains.

    ``chains`` must be ascending in ``as_of``. With fewer than two snapshots the
    session has no intraday delta to measure, so every field is ``None``:

    - ``oi_change_intraday``: total (call+put) OI at the last snapshot minus the
      session-open (first) snapshot — net intraday OI build/unwind.
    - ``iv_change_intraday``: last-snapshot ``median_iv`` minus the open's, i.e.
      the IV move versus session open (``None`` if either IV is unavailable).
    - ``pcr_volume_intraday``: mean of each snapshot's put/call *volume* ratio
      across the session — a rolling intraday put/call volume ratio.
    """
    empty: dict[str, float | None] = dict.fromkeys(INTRADAY_PANEL_FIELDS, None)
    if len(chains) < 2:
        return empty
    snapshots = [compute_chain_metrics(chain) for chain in chains]
    first, last = snapshots[0], snapshots[-1]
    oi_change = (last.call_oi + last.put_oi) - (first.call_oi + first.put_oi)
    iv_change = (
        last.median_iv - first.median_iv
        if last.median_iv is not None and first.median_iv is not None
        else None
    )
    ratios = [m.pcr_volume for m in snapshots if m.pcr_volume is not None]
    pcr_volume_intraday = sum(ratios) / len(ratios) if ratios else None
    return {
        "oi_change_intraday": oi_change,
        "iv_change_intraday": iv_change,
        "pcr_volume_intraday": pcr_volume_intraday,
    }


def _store_daily_row(chains: list[OptionChain], *, day: date) -> dict[str, object]:
    """Reduce one underlying's session snapshots to a single daily panel row.

    The base row is the metrics of the *last* snapshot of the session — the same
    reduction the EOD panel performs — with ``as_of`` pinned to the session date
    and the intraday-derived columns layered on top.
    """
    ordered = sorted(chains, key=lambda chain: chain.as_of)
    row = metrics_row(ordered[-1])
    row["as_of"] = pd.Timestamp(day)
    row.update(_intraday_fields(ordered))
    return row


def store_panel_rows(
    market: str,
    day: date,
    *,
    symbols: set[str] | None = None,
    root: Path | None = None,
) -> list[dict[str, object]]:
    """Daily panel rows reduced from the contract store for one session date.

    Returns one row per stored underlying (optionally filtered to ``symbols``,
    compared upper-cased). An empty list means the store holds nothing for that
    ``(market, day)`` and the caller should fall back to the legacy path.
    """
    wanted = {symbol.upper() for symbol in symbols} if symbols else None
    om = cast(OptionsMarket, market)
    rows: list[dict[str, object]] = []
    for underlying in contract_store.stored_underlyings(market, day=day, root=root):
        if wanted is not None and underlying.upper() not in wanted:
            continue
        frame = contract_store.load_contracts(
            underlying, market=market, day=day, root=root
        )
        if frame is None or frame.empty:
            continue
        chains = contract_store.frame_to_chains(frame, market=om)
        if not chains:
            continue
        rows.append(_store_daily_row(chains, day=day))
    return rows


def _history_metrics(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("as_of").copy()
    iv = pd.to_numeric(cast(Any, ordered.get("median_iv")), errors="coerce")
    if not isinstance(iv, pd.Series):
        iv = pd.Series(np.nan, index=ordered.index, dtype=float)
    history_days = iv.notna().cumsum()
    expanding_min = iv.expanding(min_periods=2).min()
    expanding_max = iv.expanding(min_periods=2).max()
    spread = expanding_max - expanding_min
    ordered["iv_rank"] = ((iv - expanding_min) / spread * 100.0).where(spread > 0)
    ordered["iv_history_days"] = history_days.astype(int)

    percentiles: list[float] = []
    seen: list[float] = []
    for raw in iv.tolist():
        if pd.isna(raw):
            percentiles.append(float("nan"))
            continue
        seen.append(float(raw))
        if len(seen) < 2:
            percentiles.append(float("nan"))
            continue
        less_or_equal = sum(value <= float(raw) for value in seen)
        percentiles.append(less_or_equal / len(seen) * 100.0)
    ordered["iv_percentile"] = percentiles

    volume = pd.to_numeric(cast(Any, ordered.get("options_volume")), errors="coerce")
    if not isinstance(volume, pd.Series):
        volume = pd.Series(np.nan, index=ordered.index, dtype=float)
    baseline = volume.shift(1).rolling(20, min_periods=5).mean()
    ordered["options_volume_avg_20"] = baseline
    ordered["unusual_options_ratio"] = (volume / baseline).where(baseline > 0)

    for side in ("call", "put"):
        oi_col = f"{side}_oi"
        change_col = f"{side}_oi_change"
        oi = pd.to_numeric(cast(Any, ordered.get(oi_col)), errors="coerce")
        if not isinstance(oi, pd.Series):
            oi = pd.Series(np.nan, index=ordered.index, dtype=float)
        snapshot_change = oi.diff()
        supplied = pd.to_numeric(cast(Any, ordered.get(change_col)), errors="coerce")
        if not isinstance(supplied, pd.Series):
            supplied = pd.Series(np.nan, index=ordered.index, dtype=float)
        ordered[change_col] = supplied.fillna(snapshot_change)

    call_change = pd.to_numeric(ordered["call_oi_change"], errors="coerce")
    put_change = pd.to_numeric(ordered["put_oi_change"], errors="coerce")
    snapshot_ratio = (put_change / call_change).where(call_change != 0)
    supplied_ratio = pd.to_numeric(
        cast(Any, ordered.get("oi_chg_ratio")), errors="coerce"
    )
    if not isinstance(supplied_ratio, pd.Series):
        supplied_ratio = pd.Series(np.nan, index=ordered.index, dtype=float)
    ordered["oi_chg_ratio"] = supplied_ratio.fillna(snapshot_ratio)
    ordered["history_days"] = np.arange(1, len(ordered) + 1)
    return ordered


def enrich_panel_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal expanding/trailing fields to a metrics panel."""
    if frame.empty:
        return frame.copy()
    required = {"as_of", "SYMBOL"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"options panel missing columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["as_of"] = pd.to_datetime(
        normalized["as_of"], errors="coerce"
    ).dt.normalize()
    normalized = normalized[normalized["as_of"].notna()]
    normalized["SYMBOL"] = normalized["SYMBOL"].astype(str).str.upper()
    normalized = normalized.drop_duplicates(PANEL_DEDUPE_KEYS, keep="last")
    pieces = [
        _history_metrics(group)
        for _symbol, group in normalized.groupby("SYMBOL", sort=True)
    ]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(PANEL_DEDUPE_KEYS)
        .reset_index(drop=True)
    )


def append_metrics_rows(panel_name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Merge rows, recompute causal history fields, and atomically persist."""
    existing = read_frame(panel_path(panel_name))
    combined = (
        pd.concat([existing, rows], ignore_index=True)
        if existing is not None and not existing.empty
        else rows.copy()
    )
    if combined.empty:
        return existing if existing is not None else pd.DataFrame()
    enriched = enrich_panel_history(combined)
    return append_panel_snapshot(
        panel_name,
        enriched,
        dedupe_keys=PANEL_DEDUPE_KEYS,
    )


def append_chains(
    chains: Mapping[str, OptionChain] | Iterable[OptionChain], *, market: str
) -> pd.DataFrame:
    """Compute and persist one daily row for each supplied chain."""
    if market not in OPTIONS_PANEL_NAMES:
        raise ValueError(f"unsupported options market: {market}")
    values = chains.values() if isinstance(chains, Mapping) else chains
    rows = pd.DataFrame([metrics_row(chain) for chain in values])
    return append_metrics_rows(OPTIONS_PANEL_NAMES[market], rows)


def read_options_panel(market: str) -> pd.DataFrame:
    """Read a market options panel, returning an empty frame when absent."""
    try:
        name = OPTIONS_PANEL_NAMES[market]
    except KeyError:
        raise ValueError(f"unsupported options market: {market}") from None
    frame = read_frame(panel_path(name))
    return frame if frame is not None else pd.DataFrame()


def show_symbol(market: str, symbol: str) -> pd.DataFrame:
    """Return all accumulated rows for one normalized underlying."""
    panel = read_options_panel(market)
    if panel.empty or "SYMBOL" not in panel.columns:
        return pd.DataFrame(columns=panel.columns)
    normalized = (
        tv_to_nse(symbol, strip_suffix=True)
        if market == "india"
        else symbol.strip().upper()
    )
    rows = panel[panel["SYMBOL"].astype(str).str.upper() == normalized].copy()
    return rows.sort_values("as_of").reset_index(drop=True)


def build_india_panel(
    start: date,
    end: date,
    *,
    symbols: set[str] | None = None,
    refresh: bool = False,
    fetcher: BhavcopyFetcher | None = None,
    trading_day: Callable[[date], bool] = is_trading_day,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
) -> pd.DataFrame:
    """Backfill point-in-time India metrics from daily UDiff bhavcopies."""
    if end < start:
        raise ValueError("end must be on or after start")
    normalized_symbols = (
        {tv_to_nse(symbol, strip_suffix=True) for symbol in symbols}
        if symbols
        else None
    )
    rows: list[dict[str, object]] = []
    cursor = start
    loaded_days = 0
    while cursor <= end:
        if not trading_day(cursor):
            cursor += timedelta(days=1)
            continue
        # Merge per (day, underlying): prefer contract-store rows when present,
        # then fill any requested underlyings the store did not cover from the
        # bhavcopy. Skipping the bhavcopy entirely when *any* store row exists
        # collapses the panel universe to the recorder watchlist for that day.
        store_rows = store_panel_rows("india", cursor, symbols=normalized_symbols)
        covered = {
            str(row["SYMBOL"]).upper() for row in store_rows if row.get("SYMBOL")
        }
        day_rows: list[dict[str, object]] = list(store_rows)
        store_covers_request = (
            normalized_symbols is not None
            and bool(normalized_symbols)
            and normalized_symbols <= covered
        )
        if not store_covers_request:
            try:
                chains = load_bhavcopy_chains(
                    cursor,
                    symbols=normalized_symbols,
                    refresh=refresh,
                    fetcher=fetcher,
                )
            except Exception as exc:  # noqa: BLE001 - archive gaps degrade per day
                LOG.warning("options bhavcopy unavailable for %s: %s", cursor, exc)
                if on_error is not None:
                    on_error(cursor, exc)
            else:
                for chain in chains.values():
                    symbol = str(chain.underlying).upper()
                    if symbol in covered:
                        continue  # store row wins on conflict
                    day_rows.append(metrics_row(chain))
        if day_rows:
            rows.extend(day_rows)
            loaded_days += 1
            if on_progress is not None:
                on_progress(cursor, len(day_rows))
        cursor += timedelta(days=1)
    LOG.info("built India options rows from %d trading days", loaded_days)
    return append_metrics_rows(OPTIONS_PANEL_NAMES["india"], pd.DataFrame(rows))


def snapshot_us(
    tickers: Iterable[str],
    *,
    provider: OptionsProvider | None = None,
    refresh: bool = False,
    max_workers: int = 4,
) -> SnapshotResult:
    """Fetch bounded US live snapshots and append successful normalized rows."""
    symbols = tuple(
        dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip())
    )
    source = provider or default_us_provider()

    def load(symbol: str) -> OptionChain | None:
        try:
            return source.fetch_chain(symbol, "us", refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - one symbol must not abort a batch
            LOG.warning("US options snapshot failed for %s: %s", symbol, exc)
            return None

    fetched = parallel_map(
        load,
        symbols,
        max_workers=min(max(int(max_workers), 1), 8),
        drop_none=False,
    )
    chains = tuple(chain for chain in fetched if chain is not None)
    found = {chain.underlying for chain in chains}
    panel = append_chains(chains, market="us") if chains else read_options_panel("us")
    return SnapshotResult(
        panel=panel,
        chains=chains,
        requested=len(symbols),
        missing=tuple(symbol for symbol in symbols if symbol not in found),
    )


__all__ = [
    "INTRADAY_PANEL_FIELDS",
    "OPTIONS_PANEL_NAMES",
    "PANEL_DEDUPE_KEYS",
    "SnapshotResult",
    "append_chains",
    "append_metrics_rows",
    "build_india_panel",
    "enrich_panel_history",
    "metrics_row",
    "read_options_panel",
    "show_symbol",
    "snapshot_us",
    "store_panel_rows",
]
