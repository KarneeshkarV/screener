"""First-class options contract store — every snapshot as a time series.

The daily options panel (:mod:`screener.options.panels`) collapses a chain to
one metrics row per underlying per day. This store instead persists *every*
observed contract snapshot, timestamped, so intraday chains accumulate into a
backtestable history. Partition:
``~/.screener/contracts/{market}/{YYYY-MM-DD}/{underlying}.parquet`` (one file
per underlying per session date), where the date is the snapshot timestamp in
the market's timezone.

Writes reuse the atomic temp-file + ``os.replace`` discipline and the POSIX
advisory lock of :mod:`screener.cache`, and appends dedupe on
``(underlying, expiry, strike, right, snapshot_ts)`` so re-running a recorder
pass never duplicates a snapshot (idempotent, and a no-op leaves the file mtime
intact for the freshness check ``screener cache status`` relies on).

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
    "source",
    "contract_symbol",
)

DEDUPE_KEYS: tuple[str, ...] = (
    "underlying",
    "expiry",
    "strike",
    "right",
    "snapshot_ts",
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


def chain_to_frame(chain: OptionChain) -> pd.DataFrame:
    """Flatten a normalized chain into one snapshot row per contract."""
    snapshot_ts = _naive_utc(chain.as_of)
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
        return pd.read_parquet(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def append_snapshot(
    chain: OptionChain,
    *,
    market: str,
    root: Optional[Path] = None,
    enrich: bool = True,
) -> pd.DataFrame:
    """Append one chain snapshot to the store (idempotent), return merged frame.

    Serialized with a POSIX file lock and written via a unique temp file plus
    atomic ``os.replace`` so concurrent recorders can't lose rows. Re-appending
    the same snapshot is a no-op (deduped on ``DEDUPE_KEYS``) and leaves the
    file mtime untouched.
    """
    frame = chain_to_frame(chain)
    if frame.empty:
        return frame
    if enrich:
        frame = enrich_contracts(frame, market=market)
    day = _session_date(chain.as_of, market)
    path = contract_path(chain.underlying, market=market, day=day, root=root)
    with _file_lock(path):
        existing = load_contracts(chain.underlying, market=market, day=day, root=root)
        merged = (
            pd.concat([existing, frame], ignore_index=True)
            if existing is not None and not existing.empty
            else frame.copy()
        )
        merged = merged.drop_duplicates(subset=list(DEDUPE_KEYS), keep="last")
        merged = merged.sort_values(list(DEDUPE_KEYS)).reset_index(drop=True)
        if existing is not None and merged.equals(existing):
            return merged
        _atomic_write(path, merged)
    return merged


def _atomic_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.to_parquet(tmp, compression="zstd")
        os.replace(tmp, path)
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
    """Reconstruct one :class:`OptionChain` per distinct snapshot timestamp."""
    if frame.empty:
        return []
    chains: list[OptionChain] = []
    for (underlying, snapshot_ts), group in frame.groupby(
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


def store_health(
    market: str,
    *,
    root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> StoreHealth:
    """Report last-snapshot age and missing business days for one market.

    Gap detection is a heuristic: it flags business days between the first and
    last recorded session that have no partition (it does not consult an
    exchange holiday calendar), which is enough to notice a silently dead cron.
    """
    present = _present_session_dates(market, root=root)
    if not present:
        return StoreHealth(market=market, last_snapshot=None, age=None)
    base = (Path(root) if root is not None else _default_root()) / market
    latest_mtime = 0.0
    for day in present:
        day_dir = base / day.isoformat()
        for path in day_dir.glob("*.parquet"):
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
    last_snapshot = (
        datetime.fromtimestamp(latest_mtime, tz=timezone.utc) if latest_mtime else None
    )
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
]
