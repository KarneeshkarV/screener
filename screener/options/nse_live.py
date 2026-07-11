"""NSE live equity/index option-chain normalization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timezone
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

from screener.options.models import (
    OptionChain,
    OptionContract,
    OptionRight,
    OptionsMarket,
)
from screener.options._parse import number as _number
from screener.options._parse import quote_pair


class RawFetcher(Protocol):
    def __call__(
        self, symbol: str, *, refresh: bool = False
    ) -> dict[str, Any] | None: ...


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
                        as_of=as_of,
                        source="nse_live",
                    )
                )
            except ValueError:
                continue
    return contracts


def _filtered_contracts(
    raw: dict[str, Any], underlying: str, as_of: datetime
) -> list[OptionContract]:
    filtered = raw.get("filtered")
    if not isinstance(filtered, dict):
        return []
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
    )
    if not contracts:
        contracts = _filtered_contracts(raw, underlying, as_of)
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
    from screener.unusual_volume.option_chain import fetch_option_chain

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


__all__ = ["NSELiveOptionsProvider", "RawFetcher", "parse_nse_chain"]
