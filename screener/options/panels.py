"""Daily options panels built on the repository's canonical snapshot store."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
import json
import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from screener.cache import append_panel_snapshot, panel_path, read_frame
from screener.options.metrics import compute_chain_metrics
from screener.options.models import OptionChain
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
    return row


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
            rows.extend(metrics_row(chain) for chain in chains.values())
            loaded_days += 1
            if on_progress is not None:
                on_progress(cursor, len(chains))
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
]
