"""NSE participant-wise derivatives positioning and F&O market lots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from io import StringIO
import re
from typing import Any, cast

import pandas as pd

from screener.cache import append_panel_snapshot, panel_path, read_frame
from screener.providers import CachedProvider, ProviderSpec
from screener.unusual_volume.nse_client import fetch_nse_text, is_trading_day

PARTICIPANT_OI_URL = (
    "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"
)
MARKET_LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

_PARTICIPANT_PROVIDER = CachedProvider(
    ProviderSpec(
        provider="nse",
        namespace="options_participant_oi",
        ttl_seconds=-1,
        kind="frame",
    )
)
_LOTS_PROVIDER = CachedProvider(
    ProviderSpec(provider="nse", namespace="options_market_lots", ttl_seconds=86400)
)

TextFetcher = Callable[[str, str], str | None]


def _snake(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def parse_participant_oi_csv(text: str, *, as_of: date) -> pd.DataFrame:
    """Parse NSE's title-prefixed participant OI CSV into derived net fields."""
    frame = pd.read_csv(StringIO(text), skiprows=1)
    frame.columns = [_snake(column) for column in frame.columns]
    if "client_type" not in frame.columns:
        raise ValueError("participant OI CSV missing Client Type")
    frame["participant"] = frame.pop("client_type").astype(str).str.strip()
    frame = frame[frame["participant"].ne("")]
    for column in frame.columns:
        if column != "participant":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    def net(long_column: str, short_column: str) -> pd.Series:
        if long_column not in frame.columns or short_column not in frame.columns:
            return pd.Series(float("nan"), index=frame.index, dtype=float)
        return frame[long_column] - frame[short_column]

    frame["index_futures_net"] = net("future_index_long", "future_index_short")
    frame["stock_futures_net"] = net("future_stock_long", "future_stock_short")
    frame["index_call_net"] = net("option_index_call_long", "option_index_call_short")
    frame["index_put_net"] = net("option_index_put_long", "option_index_put_short")
    frame["stock_call_net"] = net("option_stock_call_long", "option_stock_call_short")
    frame["stock_put_net"] = net("option_stock_put_long", "option_stock_put_short")
    frame["total_net"] = net("total_long_contracts", "total_short_contracts")
    frame["as_of"] = pd.Timestamp(as_of)
    frame["source"] = "nse_participant_oi"
    return frame.reset_index(drop=True)


def _default_text_fetcher(url: str, operation: str) -> str | None:
    return fetch_nse_text(url, operation, timeout=20)


def fetch_participant_oi(
    d: date,
    *,
    refresh: bool = False,
    text_fetcher: TextFetcher = _default_text_fetcher,
    cache_provider: Any = _PARTICIPANT_PROVIDER,
) -> pd.DataFrame:
    """Fetch one immutable participant-wise OI archive date."""
    url = PARTICIPANT_OI_URL.format(ddmmyyyy=d.strftime("%d%m%Y"))

    def load() -> pd.DataFrame:
        text = text_fetcher(url, f"participant OI {d}")
        if not text:
            raise RuntimeError(f"participant OI unavailable for {d}")
        return parse_participant_oi_csv(text, as_of=d)

    return cast(
        pd.DataFrame,
        cache_provider.fetch(
            d.isoformat(),
            load,
            refresh=refresh,
            fallback=pd.DataFrame(),
            operation=f"participant OI {d}",
        ),
    )


def append_participant_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return append_panel_snapshot(
        "participant_oi", rows, dedupe_keys=["as_of", "participant"]
    )


def build_participant_panel(
    start: date,
    end: date,
    *,
    refresh: bool = False,
    fetcher: Callable[..., pd.DataFrame] = fetch_participant_oi,
    trading_day: Callable[[date], bool] = is_trading_day,
) -> pd.DataFrame:
    """Backfill participant positioning while skipping unavailable dates."""
    if end < start:
        raise ValueError("end must be on or after start")
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        if trading_day(cursor):
            try:
                frame = fetcher(cursor, refresh=refresh)
            except Exception:  # noqa: BLE001 - one archive gap is expected
                frame = pd.DataFrame()
            if frame is not None and not frame.empty:
                frames.append(frame)
        cursor += timedelta(days=1)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return append_participant_rows(rows)


def read_participant_panel() -> pd.DataFrame:
    frame = read_frame(panel_path("participant_oi"))
    return frame if frame is not None else pd.DataFrame()


def parse_market_lots(text: str) -> dict[str, float]:
    """Return symbol -> nearest listed expiry lot size from ``fo_mktlots.csv``."""
    frame = pd.read_csv(StringIO(text), skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(frame.columns) < 3:
        raise ValueError("F&O market lots CSV has no expiry columns")
    symbol_column = frame.columns[1]
    expiry_columns = list(frame.columns[2:])
    lots: dict[str, float] = {}
    for _index, row in frame.iterrows():
        symbol = str(row.get(symbol_column) or "").strip().upper()
        if not symbol or symbol in {"SYMBOL", "NAN"}:
            continue
        lot: float | None = None
        for column in expiry_columns:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value) and float(value) > 0:
                lot = float(value)
                break
        if lot is not None:
            lots[symbol] = lot
    return lots


def fetch_market_lots(
    *,
    refresh: bool = False,
    text_fetcher: TextFetcher = _default_text_fetcher,
    cache_provider: Any = _LOTS_PROVIDER,
) -> dict[str, float]:
    """Fetch/cache the current NSE F&O market-lot mapping."""

    def load() -> dict[str, float]:
        text = text_fetcher(MARKET_LOTS_URL, "F&O market lots")
        if not text:
            raise RuntimeError("F&O market lots unavailable")
        return parse_market_lots(text)

    payload = cache_provider.fetch(
        ("fo_mktlots", date.today().isoformat()),
        load,
        refresh=refresh,
        fallback={},
        operation="F&O market lots",
    )
    return {str(symbol): float(lot) for symbol, lot in payload.items()}


__all__ = [
    "MARKET_LOTS_URL",
    "PARTICIPANT_OI_URL",
    "append_participant_rows",
    "build_participant_panel",
    "fetch_market_lots",
    "fetch_participant_oi",
    "parse_market_lots",
    "parse_participant_oi_csv",
    "read_participant_panel",
]
