"""yfinance US option-chain fallback normalized behind the provider protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from typing import Any, cast

import pandas as pd
import yfinance as yf

from screener.options._parse import number as _number
from screener.options._parse import quote_pair as _quote_pair
from screener.options.greeks import black_scholes_greeks
from screener.options.models import OptionChain, OptionContract, OptionsMarket
from screener.providers import CachedProvider, ProviderSpec

_YF_CACHE = CachedProvider(
    ProviderSpec(provider="yfinance", namespace="options_yfinance", ttl_seconds=900)
)


def _configure() -> None:
    from screener.backtester.data import _configure_yfinance

    _configure_yfinance()


def _spot_from_ticker(ticker: Any) -> float | None:
    try:
        fast_info = ticker.fast_info
    except Exception:  # noqa: BLE001 - yfinance mapping is lazy and provider-owned
        return None
    if isinstance(fast_info, Mapping):
        value = fast_info.get("last_price") or fast_info.get("lastPrice")
    else:
        value = getattr(fast_info, "last_price", None)
    spot = _number(value, nonnegative=True)
    return spot if spot and spot > 0 else None


def _contracts_from_frame(
    frame: pd.DataFrame,
    *,
    underlying: str,
    expiry: date,
    right: str,
    as_of: datetime,
    spot: float | None,
    risk_free_rate: float,
    missing_volume_as_count: bool,
) -> list[OptionContract]:
    if frame is None or frame.empty:
        return []
    has_volume = "volume" in frame.columns
    contracts: list[OptionContract] = []
    for position, raw_row in enumerate(frame.to_dict("records")):
        row = cast(dict[str, Any], raw_row)
        strike = _number(row.get("strike"), nonnegative=True)
        if strike is None:
            strike = float(position)
        symbol = str(row.get("contractSymbol") or "").strip()
        if not symbol:
            symbol = f"{underlying}-{expiry.isoformat()}-{strike:g}-{right}"
        volume = _number(row.get("volume"), nonnegative=True)
        if volume is None:
            volume = 1.0 if missing_volume_as_count and not has_volume else 0.0
        last = _number(row.get("lastPrice"), nonnegative=True)
        change = _number(row.get("change"))
        previous_close = (
            max(last - change, 0.0) if last is not None and change is not None else None
        )
        iv = _number(row.get("impliedVolatility"), nonnegative=True)
        bid, ask = _quote_pair(row)
        greeks: dict[str, float] = {}
        if spot is not None and strike > 0 and iv is not None and iv > 0:
            time_years = max((expiry - as_of.date()).days, 0) / 365.0
            computed = black_scholes_greeks(
                spot,
                strike,
                time_years,
                risk_free_rate,
                iv,
                "call" if right == "call" else "put",
            )
            if computed is not None:
                greeks = computed
        try:
            contracts.append(
                OptionContract(
                    symbol=symbol,
                    underlying=underlying,
                    expiry=expiry,
                    strike=strike,
                    right="call" if right == "call" else "put",
                    oi=_number(row.get("openInterest"), nonnegative=True) or 0.0,
                    volume=volume,
                    iv=iv,
                    bid=bid,
                    ask=ask,
                    last=last,
                    previous_close=previous_close,
                    as_of=as_of,
                    source="yfinance",
                    **greeks,
                )
            )
        except ValueError:
            continue
    return contracts


def chain_from_yfinance_ticker(
    ticker: Any,
    symbol: str,
    expiries: Iterable[str | date],
    *,
    now: datetime | None = None,
    risk_free_rate: float = 0.04,
    missing_volume_as_count: bool = False,
) -> OptionChain | None:
    """Normalize selected expiries from an already-created yfinance ticker."""
    as_of = now or datetime.now(UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    underlying = symbol.strip().upper()
    spot = _spot_from_ticker(ticker)
    contracts: list[OptionContract] = []
    for raw_expiry in expiries:
        parsed = pd.to_datetime(raw_expiry, errors="coerce")
        if pd.isna(parsed):
            continue
        expiry = pd.Timestamp(parsed).date()
        chain = ticker.option_chain(expiry.isoformat())
        calls = getattr(chain, "calls", pd.DataFrame())
        puts = getattr(chain, "puts", pd.DataFrame())
        contracts.extend(
            _contracts_from_frame(
                calls,
                underlying=underlying,
                expiry=expiry,
                right="call",
                as_of=as_of,
                spot=spot,
                risk_free_rate=risk_free_rate,
                missing_volume_as_count=missing_volume_as_count,
            )
        )
        contracts.extend(
            _contracts_from_frame(
                puts,
                underlying=underlying,
                expiry=expiry,
                right="put",
                as_of=as_of,
                spot=spot,
                risk_free_rate=risk_free_rate,
                missing_volume_as_count=missing_volume_as_count,
            )
        )
    if not contracts:
        return None
    return OptionChain(
        underlying=underlying,
        market="us",
        spot=spot,
        as_of=as_of,
        source="yfinance",
        contracts=tuple(contracts),
    )


class YFinanceOptionsProvider:
    """Fetch up to two near expiries as the no-key US fallback."""

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        configure: Callable[[], None] = _configure,
        cache_provider: Any = _YF_CACHE,
        now: Callable[[], datetime] | None = None,
        max_expiries: int = 2,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.ticker_factory = ticker_factory
        self.configure = configure
        self.cache_provider = cache_provider
        self.now = now or (lambda: datetime.now(UTC))
        self.max_expiries = max(int(max_expiries), 1)
        self.risk_free_rate = float(risk_free_rate)

    def _build_chain(
        self, symbol: str, expiry: date | None, now: datetime
    ) -> OptionChain | None:
        self.configure()
        ticker = self.ticker_factory(symbol)
        if expiry is not None:
            selected: list[str | date] = [expiry]
        else:
            available = list(ticker.options or [])
            dated: list[tuple[date, str]] = []
            for raw in available:
                parsed = pd.to_datetime(raw, errors="coerce")
                if pd.isna(parsed):
                    continue
                day = pd.Timestamp(parsed).date()
                if day >= now.date():
                    dated.append((day, str(raw)))
            selected = [raw for _day, raw in sorted(dated)[: self.max_expiries]]
        if not selected:
            return None
        return chain_from_yfinance_ticker(
            ticker,
            symbol,
            selected,
            now=now,
            risk_free_rate=self.risk_free_rate,
        )

    def fetch_chain(
        self,
        symbol: str,
        market: OptionsMarket = "us",
        expiry: date | None = None,
        *,
        refresh: bool = False,
    ) -> OptionChain | None:
        if market != "us":
            raise ValueError("yfinance options provider supports only the US market")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        now = self.now()

        def fetch_payload() -> dict[str, Any] | None:
            chain = self._build_chain(normalized, expiry, now)
            return chain.model_dump(mode="json") if chain is not None else None

        payload = self.cache_provider.fetch(
            (
                normalized,
                now.date().isoformat(),
                expiry.isoformat() if expiry else None,
            ),
            fetch_payload,
            refresh=refresh,
            fallback=None,
            operation=f"yfinance option chain {normalized}",
        )
        if not isinstance(payload, dict):
            return None
        return OptionChain.model_validate(payload)


__all__ = [
    "YFinanceOptionsProvider",
    "chain_from_yfinance_ticker",
]
