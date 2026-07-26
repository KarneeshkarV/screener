"""Table-driven tests for the unified ``screener.symbols`` vocabulary.

Covers every rule of each conversion plus the divergences that the old
duplicated variants encoded (``tv_to_nse`` with/without suffix stripping).
"""

from __future__ import annotations

import pytest

from screener import symbols
from screener.backtester.data import tv_to_yf as tv_to_yf_reexport
from screener.rs_breakout import india_symbol as rs_india_symbol
from screener.symbols import normalize_symbol, tv_to_nse, tv_to_yf
from screener.unusual_volume.service import india_symbol as uv_india_symbol


@pytest.mark.parametrize(
    ("symbol", "market", "expected"),
    [
        ("NSE:RELIANCE", "india", "RELIANCE.NS"),
        ("BSE:TCS", "india", "TCS.BO"),
        ("NASDAQ:AAPL", "us", "AAPL"),
        ("AAPL", "us", "AAPL"),
        ("RELIANCE", "india", "RELIANCE.NS"),
        ("RELIANCE.NS", "india", "RELIANCE.NS"),
        ("RELIANCE.BO", "india", "RELIANCE.BO"),
        (" aapl ", "us", "AAPL"),
        ("nse:reliance", "india", "RELIANCE.NS"),
        # Exchange prefix wins over market: prefix decides the suffix.
        ("NASDAQ:TSLA", "india", "TSLA"),
        ("BSE:WIPRO", "us", "WIPRO.BO"),
        ("NSE:INFY", "us", "INFY.NS"),
        # Yahoo uses hyphens where TV liquidity scans often use underscores.
        ("BAJAJ_AUTO", "india", "BAJAJ-AUTO.NS"),
        ("NAM_INDIA", "india", "NAM-INDIA.NS"),
        ("NSE:BAJAJ_AUTO", "india", "BAJAJ-AUTO.NS"),
        ("BAJAJ_AUTO.NS", "india", "BAJAJ-AUTO.NS"),
        # TradingView REIT/InvIT suffix → NSE yfinance form.
        ("EMBASSY.RR", "india", "EMBASSY.NS"),
        ("NSE:EMBASSY.RR", "india", "EMBASSY.NS"),
        ("BAGMANE.RR", "india", "BAGMANE.NS"),
        # Ampersand tickers are already Yahoo-shaped; leave them alone.
        ("M&M", "india", "M&M.NS"),
    ],
)
def test_tv_to_yf(symbol: str, market: str, expected: str) -> None:
    assert tv_to_yf(symbol, market) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        # Exchange-prefixed: take the part after ":" uppercased (both variants).
        ("NSE:RELIANCE", "RELIANCE"),
        ("nse:reliance", "RELIANCE"),
        ("BSE:TCS", "TCS"),
        # Bare symbols: default keeps a yfinance suffix as-is.
        ("RELIANCE", "RELIANCE"),
        ("reliance", "RELIANCE"),
        ("RELIANCE.NS", "RELIANCE.NS"),
        ("RELIANCE.BO", "RELIANCE.BO"),
    ],
)
def test_tv_to_nse_default_keeps_suffix(symbol: str, expected: str) -> None:
    assert tv_to_nse(symbol) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        # Exchange-prefixed branch is identical to the default variant.
        ("NSE:RELIANCE", "RELIANCE"),
        ("nse:reliance", "RELIANCE"),
        # Bare symbols: strip a trailing .NS/.BO before uppercasing.
        ("RELIANCE", "RELIANCE"),
        ("RELIANCE.NS", "RELIANCE"),
        ("RELIANCE.BO", "RELIANCE"),
    ],
)
def test_tv_to_nse_strip_suffix(symbol: str, expected: str) -> None:
    assert tv_to_nse(symbol, strip_suffix=True) == expected


def test_tv_to_nse_variants_diverge_only_on_bare_suffixed_symbol() -> None:
    # The two variants agree everywhere except a bare, suffixed symbol.
    assert tv_to_nse("RELIANCE.NS") == "RELIANCE.NS"
    assert tv_to_nse("RELIANCE.NS", strip_suffix=True) == "RELIANCE"


def test_rs_breakout_india_symbol_uses_strip_variant() -> None:
    # rs_breakout.india_symbol historically stripped suffixes.
    assert rs_india_symbol("nse:reliance") == "RELIANCE"
    assert rs_india_symbol("RELIANCE.NS") == "RELIANCE"
    assert rs_india_symbol("RELIANCE.BO") == "RELIANCE"


def test_unusual_volume_india_symbol_keeps_suffix() -> None:
    # unusual_volume.india_symbol historically kept the suffix on bare symbols.
    assert uv_india_symbol("NSE:RELIANCE") == "RELIANCE"
    assert uv_india_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert uv_india_symbol("reliance") == "RELIANCE"


def test_tv_to_yf_reexport_is_the_same_object() -> None:
    # Back-compat: importing from backtester.data still resolves the canonical fn.
    assert tv_to_yf_reexport is symbols.tv_to_yf


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RELIANCE", "RELIANCE"),
        ("  RELIANCE  ", "RELIANCE"),
        ("\tNSE:INFY\n", "NSE:INFY"),
    ],
)
def test_normalize_symbol_strips(value: str, expected: str) -> None:
    assert normalize_symbol(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_symbol_rejects_empty(value: str) -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        normalize_symbol(value)
