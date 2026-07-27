"""First-class options contract store — every snapshot as a time series.

The daily options panel (:mod:`screener.options.panels`) collapses a chain to
one metrics row per underlying per day. This store instead persists *every*
observed contract snapshot, timestamped, so intraday chains accumulate into a
backtestable history. Partition:
``~/.screener/contracts/{market}/{YYYY-MM-DD}/{underlying}.parquet`` (one file
per underlying per session date), where the date is the **observed** snapshot
timestamp (PIT) in the market's timezone.

Two timestamps are stored per row:

* ``snapshot_ts`` — when the chain was **observed/ingested** (wall clock at
  record time). This is the point-in-time key used for partitioning, chain
  reconstruction, and freshness.
* ``quote_ts`` — the **venue quote** timestamp from the provider payload
  (e.g. CBOE delayed feed stamp). Delayed feeds must not be replayed as if
  actionable at the venue stamp.

Dedupe keys are ``(underlying, expiry, strike, right, quote_ts)``: re-recording
the same still-delayed venue quote is a no-op (first observation's
``snapshot_ts`` is kept via ``keep="first"``), so a no-op leaves the file mtime
intact. Older parquet files that lack ``quote_ts`` are loaded with
``quote_ts = snapshot_ts`` (or NaT) for backward compatibility.

Writes reuse the atomic temp-file + ``os.replace`` discipline and the POSIX
advisory lock of :mod:`screener.cache`. The temp file and parent directory are
fsync'd for crash durability.

Snapshots are enriched on ingest: any contract missing implied volatility has
it inverted from its mark price, and missing greeks are filled from the IV —
reusing the same dependency-free Black-Scholes inversion
(:mod:`screener.options.greeks`) the legacy NSE bhavcopy panel uses. Enrichment
is strictly snapshot-local (causal): it only ever reads the snapshot's own
spot, price, strike and expiry.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
from typing import Any, Optional, cast
from zoneinfo import ZoneInfo

import pandas as pd

from screener.cache import _file_lock
from screener.markets import get_market
from screener.options.greeks import black_scholes_greeks, implied_volatility
from screener.options.models import OptionChain, OptionContract, OptionsMarket


CONTRACT_STORE_ROOT = Path.home() / ".screener" / "contracts"

# Snapshot schema (one row per observed contract). Partition path encodes the
# market and session date; ``market`` is also stored so a loaded frame is
# self-describing when reconstructing chains.
CONTRACT_COLUMNS: tuple[str, ...] = (
    "underlying",
    "market",
    "expiry",
    "strike",
    "right",
    "lot_size",
    "bid",
    "ask",
    "last",
    "previous_close",
    "settle",
    "oi",
    "oi_change",
    "volume",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "spot",
    "snapshot_ts",
    "quote_ts",
    "source",
    "contract_symbol",
)

# Dedupe on contract identity + venue quote time so a re-fetched delayed
# payload (same quote_ts, new wall-clock observation) does not duplicate rows.
# Prefer first observation so snapshot_ts (PIT) and mtime stay stable.
DEDUPE_KEYS: tuple[str, ...] = (
    "underlying",
    "expiry",
    "strike",
    "right",
    "quote_ts",
)
_GREEK_COLUMNS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "rho")

# Flat risk-free rates for IV/greeks inversion, matching the bhavcopy panel's
# convention (India ~91-day T-bill). Overridable per run via the environment.
_RISK_FREE_RATE_ENV = {
    "india": "SCREENER_INDIA_RISK_FREE_RATE",
    "us": "SCREENER_US_RISK_FREE_RATE",
}
_DEFAULT_RISK_FREE_RATE = {"india": 0.065, "us": 0.045}


def _default_root() -> Path:
    """Store root, resolved through the cache registry so tests can override it."""
    from screener import cache

    return cache.cache_area_path("contracts")


def _risk_free_rate(market: str) -> float:
    env = _RISK_FREE_RATE_ENV.get(market)
    default = _DEFAULT_RISK_FREE_RATE.get(market, 0.0)
    raw = os.environ.get(env) if env else None
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _naive_utc(value: datetime) -> pd.Timestamp:
    """Coerce a (possibly tz-aware) datetime to a naive-UTC pandas Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _session_date(as_of: datetime, market: str) -> date:
    """Local session date of a snapshot timestamp in the market's timezone."""
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(get_market(market).timezone)).date()


def contract_path(
    underlying: str,
    *,
    market: str,
    day: date,
    root: Optional[Path] = None,
) -> Path:
    """Store path for one (market, session date, underlying) snapshot series."""
    safe = underlying.replace("/", "_").replace(":", "_").upper()
    base = Path(root) if root is not None else _default_root()
    return base / market / day.isoformat() / f"{safe}.parquet"


def _mark_price(row: pd.Series) -> float | None:
    """Best available mark for IV inversion: mid, else last/settle/prev close."""
    bid = row.get("bid")
    ask = row.get("ask")
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)
    for column in ("last", "settle", "previous_close"):
        value = row.get(column)
        if pd.notna(value) and value > 0:
            return float(value)
    return None


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Align a loaded frame to ``CONTRACT_COLUMNS``; backfill missing ``quote_ts``.

    Pre-``quote_ts`` parquet files treated venue and observation time as one
    column (``snapshot_ts``). Fill ``quote_ts`` from ``snapshot_ts`` so dedupe
    and schema consumers keep working; true observed time cannot be recovered.
    """
    out = frame.copy()
    if "quote_ts" not in out.columns:
        if "snapshot_ts" in out.columns:
            out["quote_ts"] = out["snapshot_ts"]
        else:
            out["quote_ts"] = pd.NaT
    for column in CONTRACT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.reindex(columns=list(CONTRACT_COLUMNS))


def chain_to_frame(
    chain: OptionChain,
    *,
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Flatten a normalized chain into one snapshot row per contract.

    ``chain.as_of`` is the venue quote timestamp (``quote_ts``). ``observed_at``
    is the point-in-time observation wall clock (``snapshot_ts``). When
    ``observed_at`` is omitted, both columns use ``chain.as_of`` — appropriate
    for historical/EOD ingest where venue time equals observation time.
    Live recorders should pass ``observed_at=datetime.now(timezone.utc)``
    (or a test clock) so delayed venue stamps are not treated as actionable.
    """
    quote_ts = _naive_utc(chain.as_of)
    snapshot_ts = _naive_utc(observed_at if observed_at is not None else chain.as_of)
    rows: list[dict[str, object]] = []
    for contract in chain.contracts:
        rows.append(
            {
                "underlying": chain.underlying,
                "market": chain.market,
                "expiry": pd.Timestamp(contract.expiry),
                "strike": float(contract.strike),
                "right": contract.right,
                "lot_size": contract.lot_size,
                "bid": contract.bid,
                "ask": contract.ask,
                "last": contract.last,
                "previous_close": contract.previous_close,
                "settle": contract.settle,
                "oi": contract.oi,
                "oi_change": contract.oi_change,
                "volume": contract.volume,
                "iv": contract.iv,
                "delta": contract.delta,
                "gamma": contract.gamma,
                "theta": contract.theta,
                "vega": contract.vega,
                "rho": contract.rho,
                "spot": chain.spot,
                "snapshot_ts": snapshot_ts,
                "quote_ts": quote_ts,
                "source": contract.source,
                "contract_symbol": contract.symbol,
            }
        )
    frame = pd.DataFrame(rows, columns=list(CONTRACT_COLUMNS))
    return frame


def enrich_contracts(frame: pd.DataFrame, *, market: str) -> pd.DataFrame:
    """Fill missing IV (inverted from the mark) and missing greeks (from IV).

    Snapshot-local and causal: only the row's own spot/price/strike/expiry are
    used. Rows whose IV cannot be identified are left untouched.
    """
    if frame.empty:
        return frame
    rate = _risk_free_rate(market)
    enriched = frame.copy()
    for position, row in enriched.iterrows():
        spot = row.get("spot")
        strike = row.get("strike")
        expiry = row.get("expiry")
        if pd.isna(spot) or pd.isna(strike) or pd.isna(expiry) or spot <= 0:
            continue
        as_of_day = pd.Timestamp(row["snapshot_ts"]).date()
        expiry_day = pd.Timestamp(expiry).date()
        if expiry_day <= as_of_day:
            continue
        time_years = (expiry_day - as_of_day).days / 365.25
        if time_years <= 0:
            continue
        right = str(row["right"])
        iv = row.get("iv")
        if pd.isna(iv) or iv is None or float(iv) <= 0:
            mark = _mark_price(row)
            if mark is None:
                continue
            iv = implied_volatility(
                mark,
                float(spot),
                float(strike),
                time_years,
                rate,
                right,  # type: ignore[arg-type]
            )
            if iv is None:
                continue
            enriched.at[position, "iv"] = iv
        if any(pd.isna(row.get(column)) for column in _GREEK_COLUMNS):
            greeks = black_scholes_greeks(
                float(spot),
                float(strike),
                time_years,
                rate,
                float(iv),
                right,  # type: ignore[arg-type]
            )
            if greeks is not None:
                for column in _GREEK_COLUMNS:
                    if pd.isna(row.get(column)):
                        enriched.at[position, column] = greeks[column]
    return enriched


def load_contracts(
    underlying: str,
    *,
    market: str,
    day: date,
    root: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load one underlying's snapshot series for one session date."""
    path = contract_path(underlying, market=market, day=day, root=root)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    return _normalize_frame(frame)


def append_snapshot(
    chain: OptionChain,
    *,
    market: str,
    root: Optional[Path] = None,
    enrich: bool = True,
    observed_at: datetime | None = None,
) -> pd.DataFrame:
    """Append one chain snapshot to the store (idempotent), return merged frame.

    ``observed_at`` is the PIT wall-clock for ``snapshot_ts`` and session-date
    partitioning. Live recorders must pass it (``record_pass`` stamps
    ``datetime.now(timezone.utc)``); when omitted it falls back to
    ``chain.as_of`` so historical/EOD ingest partitions under the chain's own
    session. Venue time from ``chain.as_of`` is stored separately as
    ``quote_ts``.

    Serialized with a POSIX file lock and written via a unique temp file plus
    atomic ``os.replace`` so concurrent recorders can't lose rows. Re-appending
    the same venue quote is a no-op (deduped on ``DEDUPE_KEYS``, first
    observation kept) and leaves the file mtime untouched.
    """
    observed = observed_at or chain.as_of
    frame = chain_to_frame(chain, observed_at=observed)
    if frame.empty:
        return frame
    if enrich:
        frame = enrich_contracts(frame, market=market)
    day = _session_date(observed, market)
    path = contract_path(chain.underlying, market=market, day=day, root=root)
    with _file_lock(path):
        existing = load_contracts(chain.underlying, market=market, day=day, root=root)
        merged = (
            pd.concat([existing, frame], ignore_index=True)
            if existing is not None and not existing.empty
            else frame.copy()
        )
        merged = merged.drop_duplicates(subset=list(DEDUPE_KEYS), keep="first")
        merged = merged.sort_values(list(DEDUPE_KEYS)).reset_index(drop=True)
        if existing is not None and merged.equals(existing):
            return merged
        _atomic_write(path, merged)
    return merged


def _atomic_write(path: Path, frame: pd.DataFrame) -> None:
    """Write parquet via temp file + replace, fsync'ing for crash durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, compression="zstd")
        # Ensure data hits stable storage before the rename becomes durable.
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # Durably record the directory entry for the renamed file. Some
        # filesystems (network mounts, etc.) disallow directory fsync.
        with contextlib.suppress(OSError):
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def load_range(
    underlying: str,
    *,
    market: str,
    start: date,
    end: date,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Concatenate one underlying's snapshots across a session-date range."""
    frames: list[pd.DataFrame] = []
    for day in _present_session_dates(market, root=root):
        if day < start or day > end:
            continue
        frame = load_contracts(underlying, market=market, day=day, root=root)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(CONTRACT_COLUMNS))
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(list(DEDUPE_KEYS)).reset_index(drop=True)


def frame_to_chains(frame: pd.DataFrame, *, market: OptionsMarket) -> list[OptionChain]:
    """Reconstruct one :class:`OptionChain` per distinct PIT snapshot timestamp.

    Groups by ``snapshot_ts`` (observed/ingest time), not venue ``quote_ts``, so
    delayed quotes are replayed only as of when they were actually captured.
    """
    if frame.empty:
        return []
    normalized = _normalize_frame(frame) if "quote_ts" not in frame.columns else frame
    chains: list[OptionChain] = []
    for (underlying, snapshot_ts), group in normalized.groupby(
        ["underlying", "snapshot_ts"], sort=True
    ):
        contracts = tuple(_row_to_contract(row) for _, row in group.iterrows())
        spot_series = group["spot"].dropna()
        chains.append(
            OptionChain(
                underlying=str(underlying),
                market=market,
                spot=float(spot_series.iloc[0]) if not spot_series.empty else None,
                as_of=pd.Timestamp(cast(Any, snapshot_ts))
                .to_pydatetime()
                .replace(tzinfo=timezone.utc),
                source=str(group["source"].iloc[0]),
                contracts=contracts,
            )
        )
    return chains


def _row_to_contract(row: pd.Series) -> OptionContract:
    def _opt(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    return OptionContract(
        symbol=str(row["contract_symbol"]),
        underlying=str(row["underlying"]),
        expiry=pd.Timestamp(row["expiry"]).date(),
        strike=float(row["strike"]),
        right=str(row["right"]),  # type: ignore[arg-type]
        oi=float(row["oi"]) if pd.notna(row["oi"]) else 0.0,
        oi_change=_opt(row.get("oi_change")),
        volume=float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
        iv=_opt(row.get("iv")),
        bid=_opt(row.get("bid")),
        ask=_opt(row.get("ask")),
        last=_opt(row.get("last")),
        previous_close=_opt(row.get("previous_close")),
        settle=_opt(row.get("settle")),
        delta=_opt(row.get("delta")),
        gamma=_opt(row.get("gamma")),
        theta=_opt(row.get("theta")),
        vega=_opt(row.get("vega")),
        rho=_opt(row.get("rho")),
        lot_size=_opt(row.get("lot_size")),
        as_of=pd.Timestamp(row["snapshot_ts"])
        .to_pydatetime()
        .replace(tzinfo=timezone.utc),
        source=str(row["source"]),
    )


def stored_underlyings(
    market: str, *, day: date, root: Optional[Path] = None
) -> list[str]:
    """Underlyings with a stored snapshot file for one (market, session date).

    Returns the partition file stems (upper-cased underlyings) present on disk,
    ascending; an empty list when nothing has been recorded for that session.
    Callers use a non-empty result as the signal to take the store-derived
    daily-panel path instead of the legacy EOD path.
    """
    base = (Path(root) if root is not None else _default_root()) / market
    day_dir = base / day.isoformat()
    if not day_dir.is_dir():
        return []
    return sorted(path.stem for path in day_dir.glob("*.parquet"))


def _present_session_dates(market: str, *, root: Optional[Path] = None) -> list[date]:
    """Session dates with at least one stored file, ascending."""
    base = (Path(root) if root is not None else _default_root()) / market
    if not base.is_dir():
        return []
    days: list[date] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        try:
            days.append(date.fromisoformat(child.name))
        except ValueError:
            continue
    return sorted(days)


@dataclass(frozen=True)
class StoreHealth:
    """Freshness + gap summary for one market's contract store."""

    market: str
    last_snapshot: Optional[datetime]
    age: Optional[timedelta]
    sessions_present: list[date] = field(default_factory=list)
    missing_sessions: list[date] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        return self.age is None or self.age > timedelta(days=1)

    def summary(self) -> str:
        if self.last_snapshot is None:
            return f"{self.market}: no snapshots recorded"
        age_hours = (self.age.total_seconds() / 3600.0) if self.age else 0.0
        gaps = (
            f", {len(self.missing_sessions)} missing session(s)"
            if self.missing_sessions
            else ""
        )
        return (
            f"{self.market}: last snapshot {self.last_snapshot:%Y-%m-%d %H:%M} "
            f"({age_hours:.1f}h ago), {len(self.sessions_present)} session(s){gaps}"
        )


def _max_snapshot_ts_for_day(
    market: str,
    day: date,
    *,
    root: Optional[Path] = None,
) -> Optional[pd.Timestamp]:
    """Maximum stored ``snapshot_ts`` for one session partition (column-only read)."""
    base = (Path(root) if root is not None else _default_root()) / market
    day_dir = base / day.isoformat()
    if not day_dir.is_dir():
        return None
    latest: Optional[pd.Timestamp] = None
    for path in day_dir.glob("*.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["snapshot_ts"])
        except (OSError, pd.errors.ParserError, ValueError, KeyError):
            continue
        if frame.empty or "snapshot_ts" not in frame.columns:
            continue
        day_max = frame["snapshot_ts"].max()
        if pd.isna(day_max):
            continue
        ts = pd.Timestamp(day_max)
        if latest is None or ts > latest:
            latest = ts
    return latest


def store_health(
    market: str,
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> StoreHealth:
    """Report last-snapshot age and missing business days for one market.

    Freshness uses the maximum stored ``snapshot_ts`` (PIT observation time),
    not parquet file mtimes — rewriting an old partition must not make a dead
    recorder look fresh. Only the newest session-date partitions are scanned
    (newest-first until a timestamp is found) to keep the check cheap.

    Gap detection is a heuristic: it flags business days between the first and
    last recorded session that have no partition (it does not consult an
    exchange holiday calendar), which is enough to notice a silently dead cron.
    """
    present = _present_session_dates(market, root=root)
    if not present:
        return StoreHealth(market=market, last_snapshot=None, age=None)
    latest_ts: Optional[pd.Timestamp] = None
    # Newest session dates first: under normal partition-by-observation-date
    # the global max snapshot_ts lives on the newest day with data.
    for day in reversed(present):
        day_max = _max_snapshot_ts_for_day(market, day, root=root)
        if day_max is None:
            continue
        if latest_ts is None or day_max > latest_ts:
            latest_ts = day_max
        # Once we have a value from a day, older days cannot hold a newer
        # observation under normal partitioning — stop scanning.
        break
    last_snapshot: Optional[datetime] = None
    if latest_ts is not None:
        ts = pd.Timestamp(latest_ts)
        if ts.tzinfo is None:
            last_snapshot = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            last_snapshot = ts.tz_convert("UTC").to_pydatetime()
    reference = now or datetime.now(timezone.utc)
    age = (reference - last_snapshot) if last_snapshot else None
    expected = pd.bdate_range(present[0], present[-1]).date
    present_set = set(present)
    missing = [day for day in expected if day not in present_set]
    return StoreHealth(
        market=market,
        last_snapshot=last_snapshot,
        age=age,
        sessions_present=present,
        missing_sessions=missing,
    )


__all__ = [
    "CONTRACT_COLUMNS",
    "CONTRACT_STORE_ROOT",
    "DEDUPE_KEYS",
    "StoreHealth",
    "append_snapshot",
    "chain_to_frame",
    "contract_path",
    "enrich_contracts",
    "frame_to_chains",
    "load_contracts",
    "load_range",
    "store_health",
    "stored_underlyings",
]
