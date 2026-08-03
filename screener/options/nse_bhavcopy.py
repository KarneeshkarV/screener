"""Historical India option chains from NSE's immutable F&O UDiff bhavcopy."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from screener.operator.fetch import CACHE_ROOT, FO_ARCHIVE_URL
from screener.options._parse import nonnegative_or_zero
from screener.options._parse import number as _as_number
from screener.options._parse import positive as _positive
from screener.options.greeks import black_scholes_greeks, implied_volatility
from screener.options.lot_history import historical_lot_sizes
from screener.options.models import OptionChain, OptionContract, OptionRight
from screener.resilience import call_with_resilience
from screener.unusual_volume.nse_client import (
    FO_UDIFF_START,
    fo_bhavcopy_cache_path,
    read_cash_bhavcopy_raw,
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

# Bhavcopy has no IV, so point-in-time IV/greeks are inverted from the daily
# settle (or close) using a flat risk-free rate; India's ~91-day T-bill sits
# near this level. Overridable per run via the environment variable.
_RISK_FREE_RATE_ENV = "SCREENER_INDIA_RISK_FREE_RATE"
_DEFAULT_RISK_FREE_RATE = 0.065


def _india_risk_free_rate() -> float:
    raw = os.environ.get(_RISK_FREE_RATE_ENV)
    if raw is None:
        return _DEFAULT_RISK_FREE_RATE
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_RISK_FREE_RATE


def _derive_iv_greeks(
    *,
    spot: float | None,
    strike: float,
    expiry: date,
    as_of: date,
    right: OptionRight,
    price: float | None,
    rate: float,
) -> tuple[float | None, dict[str, float] | None]:
    """Invert IV (decimal) + greeks from a settle/close price; None on failure.

    Guards invalid geometry (non-positive spot/strike/price, non-future expiry)
    and lets the pricing library return None for prices it cannot identify.
    """
    if spot is None or spot <= 0 or strike <= 0 or price is None or price <= 0:
        return None, None
    if expiry <= as_of:
        return None, None
    time_years = (expiry - as_of).days / 365.25
    if time_years <= 0:
        return None, None
    iv = implied_volatility(price, spot, strike, time_years, rate, right)
    if iv is None:
        return None, None
    greeks = black_scholes_greeks(spot, strike, time_years, rate, iv, right)
    return iv, greeks


def _row_date(row: Mapping[str, Any], fallback: date) -> date:
    parsed = pd.to_datetime(cast(Any, row.get("TradDt")), errors="coerce")
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
    spot_prices: Mapping[str, float] | None = None,
    derive_iv: bool = True,
) -> dict[str, OptionChain]:
    """Normalize UDiff option rows into one multi-expiry chain per underlying.

    ``spot_prices`` (SYMBOL -> price) is a spot fallback for legacy bhavcopies,
    which carry no ``UndrlygPric``; it is used only when the per-contract
    underlying price yields nothing for an underlying.

    Bhavcopy has no IV, so when ``derive_iv`` is set (default) a point-in-time
    IV (decimal) and greeks are inverted from the settle/close price for live
    contracts (positive OI or volume) once spot is known.
    """
    rate = _india_risk_free_rate()
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
        if spot is None and spot_prices:
            fallback = spot_prices.get(underlying)
            if fallback is not None and fallback > 0:
                spot = float(fallback)
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
            right: OptionRight = "call" if right_raw == "CE" else "put"
            oi = nonnegative_or_zero(row.get("OpnIntrst"))
            volume = nonnegative_or_zero(row.get("TtlTradgVol"))
            settle = _positive(row.get("SttlmPric"))
            last = _last_price(row)
            iv: float | None = None
            greeks: dict[str, float] | None = None
            if derive_iv and (oi > 0 or volume > 0):
                iv, greeks = _derive_iv_greeks(
                    spot=spot,
                    strike=strike,
                    expiry=expiry,
                    as_of=contract_as_of,
                    right=right,
                    price=settle or last,
                    rate=rate,
                )
            greek_kwargs = greeks or {}
            contracts.append(
                OptionContract(
                    symbol=_contract_symbol(row, underlying, expiry),
                    underlying=underlying,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    oi=oi,
                    oi_change=_as_number(row.get("ChngInOpnIntrst")),
                    volume=volume,
                    iv=iv,
                    last=last,
                    previous_close=_positive(row.get("PrvsClsgPric")),
                    settle=settle,
                    lot_size=embedded_lot or mapped_lot,
                    as_of=datetime.combine(contract_as_of, time.min, tzinfo=UTC),
                    source="nse_bhavcopy",
                    **greek_kwargs,
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


CashCloseFetcher = Callable[[date], Mapping[str, float]]


def _read_raw(d: date) -> pd.DataFrame:
    return read_fo_bhavcopy_raw(
        d,
        cache_root=CACHE_ROOT,
        archive_url_template=FO_ARCHIVE_URL,
        resilience_call=call_with_resilience,
    )


def _read_cash_closes(d: date) -> Mapping[str, float]:
    """Return {SYMBOL: equity close} for ``d`` (SERIES == 'EQ').

    Degrades to an empty mapping when the cash bhavcopy is unavailable, so
    legacy chains simply keep ``spot=None`` rather than failing.
    """
    try:
        raw = read_cash_bhavcopy_raw(
            d,
            cache_root=CACHE_ROOT,
            resilience_call=call_with_resilience,
        )
    except (OSError, RuntimeError, FileNotFoundError, pd.errors.ParserError):
        return {}
    if raw.empty or "SYMBOL" not in raw.columns or "CLOSE_PRICE" not in raw.columns:
        return {}
    rows = raw[raw["SERIES"] == "EQ"] if "SERIES" in raw.columns else raw
    closes: dict[str, float] = {}
    prices = pd.to_numeric(rows["CLOSE_PRICE"], errors="coerce")
    for symbol, close in zip(rows["SYMBOL"], prices):
        if pd.notna(close):
            closes[str(symbol).strip().upper()] = float(close)
    return closes


def load_bhavcopy_chains(
    d: date,
    *,
    symbols: set[str] | None = None,
    refresh: bool = False,
    fetcher: BhavcopyFetcher | None = None,
    cash_fetcher: CashCloseFetcher | None = None,
    derive_iv: bool = True,
    lot_sizes: Mapping[str, float] | None = None,
) -> dict[str, OptionChain]:
    """Load and normalize all NSE option chains stamped on ``d``.

    Archive bytes are immutable and use the existing NSE cache. ``refresh``
    removes only that date's decoded cache file before re-downloading. For
    pre-UDiff dates (``d < FO_UDIFF_START``), which lack an underlying price,
    equity closes from the cash bhavcopy fill spot; ``cash_fetcher`` is
    injectable for offline tests. Legacy rows also lack an embedded lot size,
    so for those dates a point-in-time ``lot_sizes`` fallback is loaded from the
    user's lot-history file unless one is passed explicitly.
    """
    loader = fetcher or _read_raw
    if refresh and fetcher is None:
        cache_path = fo_bhavcopy_cache_path(d, Path(CACHE_ROOT))
        cache_path.unlink(missing_ok=True)
    frame = loader(d)
    spot_prices: Mapping[str, float] | None = None
    if d < FO_UDIFF_START:
        cash_loader = cash_fetcher or _read_cash_closes
        spot_prices = cash_loader(d)
        if lot_sizes is None:
            lot_sizes = historical_lot_sizes(d)
    return normalize_bhavcopy_options(
        frame,
        as_of=d,
        symbols=symbols,
        spot_prices=spot_prices,
        derive_iv=derive_iv,
        lot_sizes=lot_sizes,
    )


__all__ = [
    "OPTION_INSTRUMENT_TYPES",
    "REQUIRED_COLUMNS",
    "BhavcopyFetcher",
    "load_bhavcopy_chains",
    "normalize_bhavcopy_options",
]
