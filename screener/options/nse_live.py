"""NSE live equity/index option-chain normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

from screener.options.lot_history import historical_lot_sizes
from screener.options.models import (
    OptionChain,
    OptionContract,
    OptionRight,
    OptionsMarket,
)
from screener.options._parse import number as _number
from screener.options._parse import quote_pair
from screener.unusual_volume.nse_client import nse_cached_json

# Documented defaults for the India recorder watchlist when neither the
# live chain payload nor ``~/.screener/lot_sizes_history.csv`` supplies a lot.
# Prefer those sources when present; these are last-resort multipliers only.
# (NSE revises index lots periodically — keep history CSV current for accuracy.)
DEFAULT_INDIA_LOT_SIZES: dict[str, float] = {
    "NIFTY": 75.0,
    "BANKNIFTY": 35.0,
    "FINNIFTY": 65.0,
    "MIDCPNIFTY": 120.0,
    "NIFTYNXT50": 25.0,
}

_LEG_LOT_KEYS: tuple[str, ...] = (
    "marketLot",
    "market_lot",
    "lotSize",
    "boardLotQuantity",
    "NewBrdLotQty",
)


class RawFetcher(Protocol):
    def __call__(
        self, symbol: str, *, refresh: bool = False
    ) -> dict[str, Any] | None: ...


def _lot_from_mapping(payload: Mapping[str, Any] | None) -> float | None:
    """Extract a positive lot size from common NSE payload field names."""
    if payload is None:
        return None
    for key in _LEG_LOT_KEYS:
        value = _number(payload.get(key), nonnegative=True)
        if value is not None and value > 0:
            return value
    return None


def resolve_india_lot_size(
    underlying: str,
    *,
    as_of: date,
    leg: Mapping[str, Any] | None = None,
    records: Mapping[str, Any] | None = None,
    lot_sizes: Mapping[str, float] | None = None,
) -> float | None:
    """Resolve NSE F&O lot size for one underlying at ``as_of``.

    Precedence: embedded leg/record fields → caller ``lot_sizes`` map →
    :func:`~screener.options.lot_history.historical_lot_sizes` →
    :data:`DEFAULT_INDIA_LOT_SIZES`. Returns ``None`` only when no source has
    the symbol (callers should warn rather than silently assume 1.0).
    """
    embedded = _lot_from_mapping(leg) or _lot_from_mapping(records)
    if embedded is not None:
        return embedded
    symbol = underlying.strip().upper()
    if lot_sizes is not None and symbol in lot_sizes:
        lot = float(lot_sizes[symbol])
        if lot > 0:
            return lot
    history = historical_lot_sizes(as_of)
    if symbol in history:
        return history[symbol]
    return DEFAULT_INDIA_LOT_SIZES.get(symbol)


def _timestamp(raw: object, now: datetime) -> datetime:
    parsed = pd.to_datetime(cast(Any, raw), dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return now.astimezone(timezone.utc)
    ts = pd.Timestamp(parsed)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo("Asia/Kolkata"))
    return ts.tz_convert(timezone.utc).to_pydatetime()


def _expiry(
    row: Mapping[str, Any],
    leg: Mapping[str, Any],
    default: date,
) -> date:
    parsed = pd.to_datetime(
        cast(Any, leg.get("expiryDate") or row.get("expiryDate")),
        dayfirst=True,
        errors="coerce",
    )
    return default if pd.isna(parsed) else pd.Timestamp(parsed).date()


def _quote_pair(leg: Mapping[str, Any]) -> tuple[float | None, float | None]:
    return quote_pair(
        leg,
        bid_keys=("bidprice", "bidPrice"),
        ask_keys=("askPrice", "askprice"),
    )


def _contracts_from_records(
    records: list[Any],
    *,
    underlying: str,
    as_of: datetime,
    default_expiry: date,
    records_meta: Mapping[str, Any] | None = None,
    lot_sizes: Mapping[str, float] | None = None,
) -> list[OptionContract]:
    contracts: list[OptionContract] = []
    for raw_row in records:
        if not isinstance(raw_row, dict):
            continue
        row = cast(dict[str, Any], raw_row)
        legs: tuple[tuple[str, OptionRight], ...] = (("CE", "call"), ("PE", "put"))
        for key, right in legs:
            raw_leg = row.get(key)
            if not isinstance(raw_leg, dict):
                continue
            leg = cast(dict[str, Any], raw_leg)
            strike = _number(
                row.get("strikePrice") or leg.get("strikePrice"), nonnegative=True
            )
            if strike is None:
                strike = 0.0
            contract_expiry = _expiry(row, leg, default_expiry)
            bid, ask = _quote_pair(leg)
            iv_pct = _number(leg.get("impliedVolatility"), nonnegative=True)
            iv = iv_pct / 100.0 if iv_pct is not None else None
            identifier = str(leg.get("identifier") or "").strip()
            if not identifier:
                identifier = (
                    f"{underlying}-{contract_expiry.isoformat()}-{strike:g}-{key}"
                )
            lot_size = resolve_india_lot_size(
                underlying,
                as_of=as_of.date(),
                leg=leg,
                records=records_meta,
                lot_sizes=lot_sizes,
            )
            try:
                contracts.append(
                    OptionContract(
                        symbol=identifier,
                        underlying=underlying,
                        expiry=contract_expiry,
                        strike=strike,
                        right=right,
                        oi=_number(leg.get("openInterest"), nonnegative=True) or 0.0,
                        oi_change=_number(leg.get("changeinOpenInterest")),
                        volume=_number(leg.get("totalTradedVolume"), nonnegative=True)
                        or 0.0,
                        iv=iv,
                        bid=bid,
                        ask=ask,
                        last=_number(leg.get("lastPrice"), nonnegative=True),
                        lot_size=lot_size,
                        as_of=as_of,
                        source="nse_live",
                    )
                )
            except ValueError:
                continue
    return contracts


def _filtered_contracts(
    raw: dict[str, Any],
    underlying: str,
    as_of: datetime,
    *,
    records_meta: Mapping[str, Any] | None = None,
    lot_sizes: Mapping[str, float] | None = None,
) -> list[OptionContract]:
    filtered = raw.get("filtered")
    if not isinstance(filtered, dict):
        return []
    lot_size = resolve_india_lot_size(
        underlying,
        as_of=as_of.date(),
        records=records_meta,
        lot_sizes=lot_sizes,
    )
    contracts: list[OptionContract] = []
    legs: tuple[tuple[str, OptionRight], ...] = (("CE", "call"), ("PE", "put"))
    for key, right in legs:
        leg = filtered.get(key)
        if not isinstance(leg, dict):
            continue
        oi = _number(leg.get("totOI"), nonnegative=True)
        if oi is None:
            continue
        contracts.append(
            OptionContract(
                symbol=f"{underlying}-{as_of.date().isoformat()}-0-{key}",
                underlying=underlying,
                expiry=as_of.date(),
                strike=0.0,
                right=right,
                oi=oi,
                lot_size=lot_size,
                as_of=as_of,
                source="nse_live",
            )
        )
    return contracts


def parse_nse_chain(
    raw: dict[str, Any],
    *,
    symbol: str,
    expiry: date | None = None,
    now: datetime | None = None,
    lot_sizes: Mapping[str, float] | None = None,
) -> OptionChain | None:
    """Normalize NSE's records/filtered option-chain payload."""
    current = now or datetime.now(timezone.utc)
    records_obj = raw.get("records")
    records = records_obj if isinstance(records_obj, dict) else {}
    as_of = _timestamp(records.get("timestamp"), current)
    underlying = symbol.strip().upper()
    expiry_dates = records.get("expiryDates")
    first_expiry = (
        expiry_dates[0] if isinstance(expiry_dates, list) and expiry_dates else None
    )
    parsed_default = pd.to_datetime(
        cast(Any, first_expiry), dayfirst=True, errors="coerce"
    )
    default_expiry = (
        as_of.date() if pd.isna(parsed_default) else pd.Timestamp(parsed_default).date()
    )
    record_rows = records.get("data")
    contracts = _contracts_from_records(
        record_rows if isinstance(record_rows, list) else [],
        underlying=underlying,
        as_of=as_of,
        default_expiry=default_expiry,
        records_meta=records,
        lot_sizes=lot_sizes,
    )
    if not contracts:
        contracts = _filtered_contracts(
            raw,
            underlying,
            as_of,
            records_meta=records,
            lot_sizes=lot_sizes,
        )
    if expiry is not None:
        contracts = [contract for contract in contracts if contract.expiry == expiry]
    if not contracts:
        return None
    spots = [
        value
        for value in (
            _number(
                (row.get("CE") or row.get("PE") or {}).get("underlyingValue"),
                nonnegative=True,
            )
            for row in (record_rows if isinstance(record_rows, list) else [])
            if isinstance(row, dict)
        )
        if value is not None and value > 0
    ]
    spot = float(pd.Series(spots).median()) if spots else None
    return OptionChain(
        underlying=underlying,
        market="india",
        spot=spot,
        as_of=as_of,
        source="nse_live",
        contracts=tuple(contracts),
    )


def _default_fetcher(symbol: str, *, refresh: bool = False) -> dict[str, Any] | None:
    return fetch_option_chain(symbol, refresh=refresh)


class NSELiveOptionsProvider:
    """Adapter over the existing primed NSE option-chain seam."""

    def __init__(
        self,
        *,
        raw_fetcher: RawFetcher = _default_fetcher,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.raw_fetcher = raw_fetcher
        self.now = now or (lambda: datetime.now(timezone.utc))

    def fetch_chain(
        self,
        symbol: str,
        market: OptionsMarket = "india",
        expiry: date | None = None,
        *,
        refresh: bool = False,
    ) -> OptionChain | None:
        if market != "india":
            raise ValueError("NSE live options provider supports only India")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        raw = self.raw_fetcher(normalized, refresh=refresh)
        if not isinstance(raw, dict):
            return None
        return parse_nse_chain(raw, symbol=normalized, expiry=expiry, now=self.now())


# ---------------------------------------------------------------------------
# Raw NSE option-chain HTTP transport
#
# NSE serves the equity/index option chain live only (no historical archive).
# This is the low-level fetch that primes the browser session and returns the
# raw JSON payload; ``NSELiveOptionsProvider`` (above) normalizes it into an
# ``OptionChain``. It lives here so the options package owns its own transport
# and ``unusual_volume`` depends on ``options`` rather than the reverse.
# ---------------------------------------------------------------------------

_OC_EQUITY_URL = "https://www.nseindia.com/api/option-chain-equities?symbol={sym}"
_OC_INDEX_URL = "https://www.nseindia.com/api/option-chain-indices?symbol={sym}"
_OC_INDEX_SYMBOLS = frozenset(
    {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
)
_OC_PAGE = "https://www.nseindia.com/option-chain"


def fetch_option_chain(symbol: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """Fetch the raw live NSE option-chain JSON for ``symbol`` (or ``None``)."""
    import urllib.parse

    normalized = symbol.upper()
    template = _OC_INDEX_URL if normalized in _OC_INDEX_SYMBOLS else _OC_EQUITY_URL
    url = template.format(sym=urllib.parse.quote(normalized))
    raw = nse_cached_json(
        "nse_option_chain",
        ("oc", normalized, str(date.today())),
        url,
        f"option chain {symbol}",
        refresh=refresh,
        extra_prime_page=_OC_PAGE,
    )
    return raw if isinstance(raw, dict) else None


__all__ = [
    "DEFAULT_INDIA_LOT_SIZES",
    "NSELiveOptionsProvider",
    "RawFetcher",
    "fetch_option_chain",
    "parse_nse_chain",
    "resolve_india_lot_size",
]
