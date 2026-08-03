"""Free market-level options/volatility series for US and India regimes."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from io import StringIO
from typing import Any, cast

import numpy as np
import pandas as pd
import requests

from screener.cache import append_panel_snapshot, panel_path, read_frame
from screener.providers import CachedProvider, ProviderSpec
from screener.unusual_volume.nse_client import (
    fetch_nse_text,
    nse_cached_json,
)

INDIA_VIX_ARCHIVE_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
)
INDIA_ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"
CBOE_DAILY_STATS_URL = (
    "https://www.cboe.com/us/options/market_statistics/daily/?dt={yyyy_mm_dd}"
)
FRED_VOLATILITY_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS,VXVCLS"

_INDIA_VIX_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="nse",
        namespace="options_india_vix_archive",
        ttl_seconds=-1,
        kind="frame",
    )
)
_CBOE_PCR_PROVIDER = CachedProvider(
    ProviderSpec(provider="cboe", namespace="options_cboe_market_pcr", ttl_seconds=-1)
)
_FRED_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="fred",
        namespace="options_fred_volatility",
        ttl_seconds=86400,
        kind="frame",
    )
)


def _vol_regime(value: float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value < 15:
        return "low"
    if value < 25:
        return "normal"
    return "high"


def parse_india_vix_archive(text: str, *, requested_date: date) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Index Name", "Index Date", "Closing Index Value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"NSE index archive missing columns: {sorted(missing)}")
    selected = frame[
        frame["Index Name"].astype(str).str.strip().str.upper().eq("INDIA VIX")
    ]
    if selected.empty:
        return pd.DataFrame()
    row = selected.iloc[-1]
    parsed_date = pd.to_datetime(row["Index Date"], dayfirst=True, errors="coerce")
    as_of = requested_date if pd.isna(parsed_date) else pd.Timestamp(parsed_date).date()
    value = pd.to_numeric(row["Closing Index Value"], errors="coerce")
    if pd.isna(value):
        return pd.DataFrame()
    vix = float(value)
    return pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp(as_of),
                "india_vix": vix,
                "vol_regime": _vol_regime(vix),
                "source": "nse_index_archive",
            }
        ]
    )


def _archive_text(url: str, operation: str) -> str | None:
    return fetch_nse_text(url, operation, timeout=20)


def fetch_india_vix_archive(
    d: date,
    *,
    refresh: bool = False,
    text_fetcher: Callable[[str, str], str | None] = _archive_text,
    cache_provider: Any = _INDIA_VIX_PROVIDER,
) -> pd.DataFrame:
    url = INDIA_VIX_ARCHIVE_URL.format(ddmmyyyy=d.strftime("%d%m%Y"))

    def load() -> pd.DataFrame:
        text = text_fetcher(url, f"India VIX archive {d}")
        if not text:
            raise RuntimeError(f"India VIX archive unavailable for {d}")
        return parse_india_vix_archive(text, requested_date=d)

    return cast(
        pd.DataFrame,
        cache_provider.fetch(
            d.isoformat(),
            load,
            refresh=refresh,
            fallback=pd.DataFrame(),
            operation=f"India VIX archive {d}",
        ),
    )


def parse_india_vix_live(raw: object, *, as_of: date) -> pd.DataFrame:
    if not isinstance(raw, dict):
        return pd.DataFrame()
    rows = raw.get("data")
    if not isinstance(rows, list):
        return pd.DataFrame()
    for item in rows:
        if not isinstance(item, dict):
            continue
        if str(item.get("index") or "").strip().upper() != "INDIA VIX":
            continue
        value = pd.to_numeric(cast(Any, item.get("last")), errors="coerce")
        if pd.isna(value):
            return pd.DataFrame()
        vix = float(value)
        return pd.DataFrame(
            [
                {
                    "as_of": pd.Timestamp(as_of),
                    "india_vix": vix,
                    "vol_regime": _vol_regime(vix),
                    "source": "nse_all_indices_live",
                }
            ]
        )
    return pd.DataFrame()


def fetch_india_vix_live(
    *,
    as_of: date | None = None,
    refresh: bool = False,
    raw_fetcher: Callable[..., object] = nse_cached_json,
) -> pd.DataFrame:
    day = as_of or date.today()
    raw = raw_fetcher(
        "nse_all_indices",
        ("india_vix", day.isoformat()),
        INDIA_ALL_INDICES_URL,
        "India VIX live",
        refresh=refresh,
    )
    return parse_india_vix_live(raw, as_of=day)


def build_india_vix_panel(
    start: date,
    end: date,
    *,
    refresh: bool = False,
    fetcher: Callable[..., pd.DataFrame] = fetch_india_vix_archive,
) -> pd.DataFrame:
    if end < start:
        raise ValueError("end must be on or after start")
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            try:
                frame = fetcher(cursor, refresh=refresh)
            except Exception:  # noqa: BLE001 - holidays/archive gaps are normal
                frame = pd.DataFrame()
            if frame is not None and not frame.empty:
                frames.append(frame)
        cursor += timedelta(days=1)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return append_panel_snapshot("india_vix", rows, dedupe_keys=["as_of"])


def parse_cboe_market_stats_html(html: str, *, as_of: date) -> dict[str, object]:
    """Extract the official page's embedded JSON ratio name/value records."""
    decoded = html.replace('\\"', '"')
    pairs = re.findall(
        r'"name":"([^"]*PUT/CALL RATIO)","value":"([^"]*)"',
        decoded,
        flags=re.IGNORECASE,
    )
    values = {
        name.upper(): pd.to_numeric(value, errors="coerce") for name, value in pairs
    }
    mapping = {
        "total_pcr": "TOTAL PUT/CALL RATIO",
        "index_pcr": "INDEX PUT/CALL RATIO",
        "etp_pcr": "EXCHANGE TRADED PRODUCTS PUT/CALL RATIO",
        "equity_pcr": "EQUITY PUT/CALL RATIO",
        "vix_pcr": "CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO",
        "spx_pcr": "SPX + SPXW PUT/CALL RATIO",
    }
    if "TOTAL PUT/CALL RATIO" not in values:
        raise ValueError("CBOE daily statistics page has no put/call ratios")
    row: dict[str, object] = {
        "as_of": as_of.isoformat(),
        "source_pcr": "cboe_daily_statistics",
    }
    for column, label in mapping.items():
        value = values.get(label)
        row[column] = float(value) if value is not None and pd.notna(value) else None
    return row


def fetch_cboe_market_pcr(
    d: date,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
    cache_provider: Any = _CBOE_PCR_PROVIDER,
) -> pd.DataFrame:
    url = CBOE_DAILY_STATS_URL.format(yyyy_mm_dd=d.isoformat())

    def load() -> dict[str, object]:
        request = session.get if session is not None else requests.get
        response = request(url, timeout=30)
        response.raise_for_status()
        return parse_cboe_market_stats_html(response.text, as_of=d)

    row = cache_provider.fetch(
        d.isoformat(),
        load,
        refresh=refresh,
        fallback={},
        ttl_seconds=3600 if d >= date.today() else -1,
        operation=f"CBOE daily market statistics {d}",
    )
    return pd.DataFrame([row]) if row else pd.DataFrame()


def parse_fred_volatility_csv(text: str, *, start: date, end: date) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text))
    if "observation_date" not in frame.columns:
        raise ValueError("FRED volatility CSV missing observation_date")
    frame["as_of"] = pd.to_datetime(frame.pop("observation_date"), errors="coerce")
    frame["vix"] = pd.to_numeric(cast(Any, frame.get("VIXCLS")), errors="coerce")
    frame["vix3m"] = pd.to_numeric(cast(Any, frame.get("VXVCLS")), errors="coerce")
    frame = frame[
        frame["as_of"].notna()
        & frame["as_of"].between(pd.Timestamp(start), pd.Timestamp(end))
    ][["as_of", "vix", "vix3m"]]
    frame = frame[frame[["vix", "vix3m"]].notna().any(axis=1)].copy()
    frame["vol_term_spread"] = frame["vix3m"] - frame["vix"]
    frame["vol_term_ratio"] = (frame["vix"] / frame["vix3m"]).where(
        frame["vix3m"].ne(0)
    )
    frame["vol_regime"] = frame["vix"].map(
        lambda value: _vol_regime(float(value)) if pd.notna(value) else None
    )
    frame["source_vol"] = "fred_vix_vix3m"
    return frame.reset_index(drop=True)


def fetch_us_volatility(
    start: date,
    end: date,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
    cache_provider: Any = _FRED_PROVIDER,
) -> pd.DataFrame:
    def load() -> pd.DataFrame:
        request = session.get if session is not None else requests.get
        response = request(FRED_VOLATILITY_URL, timeout=30)
        response.raise_for_status()
        return parse_fred_volatility_csv(response.text, start=start, end=end)

    full = cache_provider.fetch(
        ("VIXCLS", "VXVCLS", start.isoformat(), end.isoformat()),
        load,
        refresh=refresh,
        fallback=pd.DataFrame(),
        operation="FRED VIX/VIX3M",
    )
    return cast(pd.DataFrame, full)


def build_us_regime_panel(
    start: date,
    end: date,
    *,
    refresh: bool = False,
    pcr_fetcher: Callable[..., pd.DataFrame] = fetch_cboe_market_pcr,
    volatility_fetcher: Callable[..., pd.DataFrame] = fetch_us_volatility,
) -> pd.DataFrame:
    if end < start:
        raise ValueError("end must be on or after start")
    volatility = volatility_fetcher(start, end, refresh=refresh)
    pcr_frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            try:
                frame = pcr_fetcher(cursor, refresh=refresh)
            except Exception:  # noqa: BLE001 - market holiday/page gap
                frame = pd.DataFrame()
            if frame is not None and not frame.empty:
                pcr_frames.append(frame)
        cursor += timedelta(days=1)
    pcr = pd.concat(pcr_frames, ignore_index=True) if pcr_frames else pd.DataFrame()
    if not pcr.empty:
        pcr["as_of"] = pd.to_datetime(pcr["as_of"], errors="coerce").dt.normalize()
    if volatility is None:
        volatility = pd.DataFrame()
    if pcr.empty:
        rows = volatility.copy()
    elif volatility.empty:
        rows = pcr.copy()
    else:
        rows = volatility.merge(pcr, on="as_of", how="outer")
    if not rows.empty:
        pcr_source = (
            rows["source_pcr"]
            if "source_pcr" in rows.columns
            else pd.Series(None, index=rows.index, dtype=object)
        )
        vol_source = (
            rows["source_vol"]
            if "source_vol" in rows.columns
            else pd.Series(None, index=rows.index, dtype=object)
        )
        rows["source"] = np.select(
            [pcr_source.notna() & vol_source.notna(), pcr_source.notna()],
            ["cboe_daily_statistics+fred", "cboe_daily_statistics"],
            default="fred_vix_vix3m",
        )
    return append_panel_snapshot("pcr_market_us", rows, dedupe_keys=["as_of"])


def read_regime_panel(market: str) -> pd.DataFrame:
    name = (
        "india_vix"
        if market == "india"
        else "pcr_market_us"
        if market == "us"
        else None
    )
    if name is None:
        raise ValueError(f"unsupported options regime market: {market}")
    frame = read_frame(panel_path(name))
    return frame if frame is not None else pd.DataFrame()


__all__ = [
    "CBOE_DAILY_STATS_URL",
    "FRED_VOLATILITY_URL",
    "INDIA_ALL_INDICES_URL",
    "INDIA_VIX_ARCHIVE_URL",
    "build_india_vix_panel",
    "build_us_regime_panel",
    "fetch_cboe_market_pcr",
    "fetch_india_vix_archive",
    "fetch_india_vix_live",
    "fetch_us_volatility",
    "parse_cboe_market_stats_html",
    "parse_fred_volatility_csv",
    "parse_india_vix_archive",
    "parse_india_vix_live",
    "read_regime_panel",
]
