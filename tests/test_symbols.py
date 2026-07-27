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
        # TradingView writes both '&' and '-' as '_'; restore the real one.
        ("M_M", "india", "M&M.NS"),
        ("NSE:M_M", "india", "M&M.NS"),
        ("BSE:M_M", "india", "M&M.BO"),
        ("M_MFIN", "india", "M&MFIN.NS"),
        ("J_KBANK", "india", "J&KBANK.NS"),
        ("GVT_D", "india", "GVT&D.NS"),
        ("ARE_M", "india", "ARE&M.NS"),
        ("BAJAJ_AUTO", "india", "BAJAJ-AUTO.NS"),
        ("NAM_INDIA", "india", "NAM-INDIA.NS"),
        ("BOSCH_HCIL", "india", "BOSCH-HCIL.NS"),
        # Already-suffixed underscore symbols convert the base only, so the
        # '.NS' tail never reaches the ampersand lookup.
        ("M_M.NS", "india", "M&M.NS"),
        ("BAJAJ_AUTO.NS", "india", "BAJAJ-AUTO.NS"),
        # US symbols keep underscores untouched.
        ("BRK_B", "us", "BRK_B"),
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


@pytest.mark.parametrize(
    ("symbol", "expected_default", "expected_stripped"),
    [
        ("M_M", "M&M", "M&M"),
        ("M_M.NS", "M&M.NS", "M&M"),
        ("NSE:M_M", "M&M", "M&M"),
        ("BAJAJ_AUTO", "BAJAJ-AUTO", "BAJAJ-AUTO"),
        ("BAJAJ_AUTO.NS", "BAJAJ-AUTO.NS", "BAJAJ-AUTO"),
    ],
)
def test_tv_to_nse_restores_separator(
    symbol: str, expected_default: str, expected_stripped: str
) -> None:
    # The bhavcopy spells these 'M&M' / 'BAJAJ-AUTO', never with an underscore.
    assert tv_to_nse(symbol) == expected_default
    assert tv_to_nse(symbol, strip_suffix=True) == expected_stripped


def test_restore_india_separator_is_closed_over_nse_ampersand_list() -> None:
    # Every enumerated '&' symbol must round-trip from its underscore form.
    for real in symbols._NSE_AMPERSAND_SYMBOLS:
        assert symbols.restore_india_separator(real.replace("&", "_")) == real


@pytest.mark.parametrize("value", ["RELIANCE", "AAPL", "TCS.NS", ""])
def test_restore_india_separator_passes_through_without_underscore(
    value: str,
) -> None:
    assert symbols.restore_india_separator(value) == value


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
