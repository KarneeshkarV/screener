"""Historical India option chains from NSE's immutable F&O UDiff bhavcopy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from screener.options.models import OptionChain, OptionContract
from screener.operator.fetch import CACHE_ROOT, FO_ARCHIVE_URL
from screener.resilience import call_with_resilience
from screener.unusual_volume.nse_client import (
    fo_bhavcopy_cache_path,
    read_fo_bhavcopy_raw,
)

OPTION_INSTRUMENT_TYPES = frozenset({"STO", "IDO"})
REQUIRED_COLUMNS = frozenset(
    {
        "FinInstrmTp",
        "TckrSymb",
        "XpryDt",
        "StrkPric",
        "OptnTp",
        "OpnIntrst",
        "ChngInOpnIntrst",
        "TtlTradgVol",
    }
)
BhavcopyFetcher = Callable[[date], pd.DataFrame]


def _as_number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _positive(value: object) -> float | None:
    number = _as_number(value)
    return number if number is not None and number > 0 else None


def _nonnegative(value: object) -> float:
    number = _as_number(value)
    return max(number or 0.0, 0.0)


def _row_date(row: Mapping[str, Any], fallback: date) -> date:
    parsed = pd.to_datetime(row.get("TradDt"), errors="coerce")
    if pd.isna(parsed):
        return fallback
    return pd.Timestamp(parsed).date()


def _last_price(row: Mapping[str, Any]) -> float | None:
    return (
        _positive(row.get("ClsPric"))
        or _positive(row.get("LastPric"))
        or _positive(row.get("SttlmPric"))
    )


def _contract_symbol(row: Mapping[str, Any], underlying: str, expiry: date) -> str:
    raw = str(row.get("FinInstrmNm") or "").strip()
    if raw and raw.lower() != "nan":
        return raw
    return (
        f"{underlying}-{expiry.isoformat()}-{float(row['StrkPric']):g}-"
        f"{str(row['OptnTp']).upper()}"
    )


def normalize_bhavcopy_options(
    frame: pd.DataFrame,
    *,
    as_of: date,
    symbols: set[str] | None = None,
    lot_sizes: Mapping[str, float] | None = None,
) -> dict[str, OptionChain]:
    """Normalize UDiff option rows into one multi-expiry chain per underlying."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"NSE F&O bhavcopy missing columns: {sorted(missing)}")
    if frame.empty:
        return {}

    options = frame[frame["FinInstrmTp"].isin(OPTION_INSTRUMENT_TYPES)].copy()
    options["TckrSymb"] = options["TckrSymb"].astype(str).str.strip().str.upper()
    if symbols is not None:
        wanted = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
        options = options[options["TckrSymb"].isin(wanted)]
    if options.empty:
        return {}

    options["_expiry"] = pd.to_datetime(options["XpryDt"], errors="coerce")
    options = options[options["_expiry"].notna()]
    chains: dict[str, OptionChain] = {}
    for underlying_raw, group in options.groupby("TckrSymb", sort=True):
        underlying = str(underlying_raw).strip().upper()
        contracts: list[OptionContract] = []
        spots = [
            value
            for value in (_positive(raw) for raw in group.get("UndrlygPric", []))
            if value is not None
        ]
        spot = float(pd.Series(spots).median()) if spots else None
        for raw_row in group.to_dict("records"):
            row = cast(dict[str, Any], raw_row)
            right_raw = str(row.get("OptnTp") or "").strip().upper()
            if right_raw not in {"CE", "PE"}:
                continue
            strike = _as_number(row.get("StrkPric"))
            if strike is None or strike < 0:
                continue
            expiry = pd.Timestamp(row["_expiry"]).date()
            embedded_lot = _positive(row.get("NewBrdLotQty"))
            mapped_lot = _positive((lot_sizes or {}).get(underlying))
            contract_as_of = _row_date(row, as_of)
            contracts.append(
                OptionContract(
                    symbol=_contract_symbol(row, underlying, expiry),
                    underlying=underlying,
                    expiry=expiry,
                    strike=strike,
                    right="call" if right_raw == "CE" else "put",
                    oi=_nonnegative(row.get("OpnIntrst")),
                    oi_change=_as_number(row.get("ChngInOpnIntrst")),
                    volume=_nonnegative(row.get("TtlTradgVol")),
                    last=_last_price(row),
                    previous_close=_positive(row.get("PrvsClsgPric")),
                    lot_size=embedded_lot or mapped_lot,
                    as_of=datetime.combine(
                        contract_as_of, time.min, tzinfo=timezone.utc
                    ),
                    source="nse_bhavcopy",
                )
            )
        if not contracts:
            continue
        chain_as_of = max(contract.as_of for contract in contracts)
        chains[underlying] = OptionChain(
            underlying=underlying,
            market="india",
            spot=spot,
            as_of=chain_as_of,
            source="nse_bhavcopy",
            contracts=tuple(contracts),
        )
    return chains


def _read_raw(d: date) -> pd.DataFrame:
    return read_fo_bhavcopy_raw(
        d,
        cache_root=CACHE_ROOT,
        archive_url_template=FO_ARCHIVE_URL,
        resilience_call=call_with_resilience,
    )


def load_bhavcopy_chains(
    d: date,
    *,
    symbols: set[str] | None = None,
    refresh: bool = False,
    fetcher: BhavcopyFetcher | None = None,
) -> dict[str, OptionChain]:
    """Load and normalize all NSE option chains stamped on ``d``.

    Archive bytes are immutable and use the existing NSE cache. ``refresh``
    removes only that date's decoded cache file before re-downloading.
    """
    loader = fetcher or _read_raw
    if refresh and fetcher is None:
        cache_path = fo_bhavcopy_cache_path(d, Path(CACHE_ROOT))
        cache_path.unlink(missing_ok=True)
    frame = loader(d)
    return normalize_bhavcopy_options(frame, as_of=d, symbols=symbols)


__all__ = [
    "BhavcopyFetcher",
    "OPTION_INSTRUMENT_TYPES",
    "REQUIRED_COLUMNS",
    "load_bhavcopy_chains",
    "normalize_bhavcopy_options",
]
