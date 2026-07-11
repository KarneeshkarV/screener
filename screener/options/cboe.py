"""CBOE delayed US option-chain provider (free, roughly 15-minute delayed)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
import re
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from screener.options.models import (
    OptionChain,
    OptionContract,
    OptionRight,
    OptionsMarket,
)
from screener.options._parse import number as _number
from screener.options._parse import quote_pair as _quote_pair
from screener.providers import CachedProvider, ProviderSpec

CBOE_DELAYED_URL = (
    "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
)
CBOE_INDEX_SYMBOLS = frozenset(
    {"SPX", "VIX", "XSP", "OEX", "RUT", "NDX", "DJX", "MRUT"}
)
_OCC_SYMBOL = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")
_CBOE_CACHE = CachedProvider(
    ProviderSpec(provider="cboe", namespace="options_cboe", ttl_seconds=900)
)


def _venue_timestamp(value: object, fallback: datetime) -> datetime:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback.astimezone(timezone.utc)
    ts = pd.Timestamp(parsed)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo("America/Chicago"))
    return ts.tz_convert(timezone.utc).to_pydatetime()


def _contract_parts(symbol: str) -> tuple[str, date, OptionRight, float] | None:
    match = _OCC_SYMBOL.match(symbol.strip().upper())
    if match is None:
        return None
    root, expiry_raw, right_raw, strike_raw = match.groups()
    try:
        expiry = datetime.strptime(expiry_raw, "%y%m%d").date()
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    return root, expiry, "call" if right_raw == "C" else "put", strike


def parse_cboe_chain(
    raw: dict[str, Any],
    *,
    requested_symbol: str,
    expiry: date | None = None,
    now: datetime | None = None,
) -> OptionChain | None:
    """Normalize one delayed-quotes JSON payload."""
    payload = raw.get("data")
    if not isinstance(payload, dict):
        return None
    option_rows = payload.get("options")
    if not isinstance(option_rows, list):
        return None
    fallback = now or datetime.now(timezone.utc)
    as_of = _venue_timestamp(raw.get("timestamp"), fallback)
    underlying = str(payload.get("symbol") or requested_symbol).lstrip("_").upper()
    spot = _number(payload.get("current_price"), nonnegative=True)
    if spot == 0:
        spot = None
    contracts: list[OptionContract] = []
    for item in option_rows:
        if not isinstance(item, dict):
            continue
        contract_symbol = str(item.get("option") or "").strip().upper()
        parts = _contract_parts(contract_symbol)
        if parts is None:
            continue
        _root, contract_expiry, right, strike = parts
        if expiry is not None and contract_expiry != expiry:
            continue
        bid, ask = _quote_pair(item)
        try:
            contracts.append(
                OptionContract(
                    symbol=contract_symbol,
                    underlying=underlying,
                    expiry=contract_expiry,
                    strike=strike,
                    right=right,
                    oi=_number(item.get("open_interest"), nonnegative=True) or 0.0,
                    volume=_number(item.get("volume"), nonnegative=True) or 0.0,
                    iv=_number(item.get("iv"), nonnegative=True),
                    bid=bid,
                    ask=ask,
                    last=_number(item.get("last_trade_price"), nonnegative=True),
                    previous_close=_number(
                        item.get("prev_day_close"), nonnegative=True
                    ),
                    delta=_number(item.get("delta")),
                    gamma=_number(item.get("gamma")),
                    theta=_number(item.get("theta")),
                    vega=_number(item.get("vega")),
                    rho=_number(item.get("rho")),
                    as_of=as_of,
                    source="cboe_delayed",
                )
            )
        except ValueError:
            continue
    if not contracts:
        return None
    return OptionChain(
        underlying=underlying,
        market="us",
        spot=spot,
        as_of=as_of,
        source="cboe_delayed",
        contracts=tuple(contracts),
    )


def cboe_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if normalized.startswith("_"):
        return normalized
    return f"_{normalized}" if normalized in CBOE_INDEX_SYMBOLS else normalized


class CboeOptionsProvider:
    """Fetch normalized CBOE delayed chains through cache + resilience."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_provider: CachedProvider = _CBOE_CACHE,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.session = session
        self.cache_provider = cache_provider
        self.now = now or (lambda: datetime.now(timezone.utc))

    def _fetch_raw(self, symbol: str) -> dict[str, Any]:
        url = CBOE_DELAYED_URL.format(symbol=cboe_symbol(symbol))
        request = self.session.get if self.session is not None else requests.get
        response = request(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("CBOE delayed quote response is not an object")
        return cast(dict[str, Any], payload)

    def fetch_chain(
        self,
        symbol: str,
        market: OptionsMarket = "us",
        expiry: date | None = None,
        *,
        refresh: bool = False,
    ) -> OptionChain | None:
        if market != "us":
            raise ValueError("CBOE options provider supports only the US market")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        now = self.now()
        raw = self.cache_provider.fetch(
            (cboe_symbol(normalized), now.date().isoformat()),
            lambda: self._fetch_raw(normalized),
            refresh=refresh,
            fallback=None,
            operation=f"CBOE option chain {normalized}",
        )
        if not isinstance(raw, dict):
            return None
        return parse_cboe_chain(
            raw,
            requested_symbol=normalized,
            expiry=expiry,
            now=now,
        )


__all__ = [
    "CBOE_DELAYED_URL",
    "CBOE_INDEX_SYMBOLS",
    "CboeOptionsProvider",
    "cboe_symbol",
    "parse_cboe_chain",
]
