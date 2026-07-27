"""Symbol vocabulary: TradingView-style symbols → yfinance / NSE forms.

Single home for the symbol-conversion rules that were previously duplicated
(and had drifted) across ``backtester.data``, ``rs_breakout`` and the
``unusual_volume`` package.
"""

from __future__ import annotations


# TradingView writes both ``&`` and ``-`` as ``_`` in Indian symbols, so the
# underscore alone cannot say which separator the real ticker uses. yfinance,
# FMP and the NSE bhavcopy all want the true separator (``M_M`` resolves
# nowhere; ``M&M`` resolves everywhere), and leaving the underscore in place
# silently drops the symbol from every screen.
#
# The ``&`` side is a small closed set, enumerated from NSE's EQUITY_L master
# list; every other underscore is a ``-``. Refresh from
# https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv if NSE lists a
# new ``&`` name.
_NSE_AMPERSAND_SYMBOLS = frozenset(
    {
        "ARE&M",
        "GMRP&UI",
        "GVT&D",
        "IL&FSENGG",
        "IL&FSTRANS",
        "J&KBANK",
        "M&M",
        "M&MFIN",
        "S&SPOWER",
        "SURANAT&P",
    }
)


def restore_india_separator(symbol: str) -> str:
    """Turn a TradingView underscore back into NSE's ``&`` or ``-``.

    ``M_M`` → ``M&M``, ``BAJAJ_AUTO`` → ``BAJAJ-AUTO``. Symbols without an
    underscore are returned unchanged.
    """
    if "_" not in symbol:
        return symbol
    ampersand = symbol.replace("_", "&")
    if ampersand in _NSE_AMPERSAND_SYMBOLS:
        return ampersand
    return symbol.replace("_", "-")


def tv_to_yf(symbol: str, market: str) -> str:
    """Translate a TradingView-style symbol to a yfinance symbol.

    Examples:
      'NSE:RELIANCE' + india → 'RELIANCE.NS'
      'BSE:TCS'     + india → 'TCS.BO'
      'NASDAQ:AAPL' + us    → 'AAPL'
      'AAPL'        + us    → 'AAPL'
      'RELIANCE'    + india → 'RELIANCE.NS'
      'M_M'         + india → 'M&M.NS'
      'BAJAJ_AUTO'  + india → 'BAJAJ-AUTO.NS'
    """
    sym = symbol.strip().upper()
    if ":" in sym:
        exch, rest = sym.split(":", 1)
        if exch == "NSE":
            return f"{restore_india_separator(rest)}.NS"
        if exch == "BSE":
            return f"{restore_india_separator(rest)}.BO"
        return rest
    if market == "india":
        base, dot, suffix = sym.partition(".")
        base = restore_india_separator(base)
        return f"{base}{dot}{suffix}" if dot else f"{base}.NS"
    return sym


def tv_to_nse(symbol: str, *, strip_suffix: bool = False) -> str:
    """Return the NSE bhavcopy ``SYMBOL`` for a TradingView-style symbol.

    Exchange-prefixed symbols always yield the part after ``:`` uppercased
    (``NSE:RELIANCE`` → ``RELIANCE``). The two historical variants differ only
    on bare symbols:

    - ``strip_suffix=False`` (unusual_volume): uppercase as-is, so a yfinance
      suffix is preserved (``RELIANCE.NS`` → ``RELIANCE.NS``).
    - ``strip_suffix=True`` (rs_breakout): also strip a trailing ``.NS``/``.BO``
      suffix (``RELIANCE.NS`` → ``RELIANCE``).

    Both variants restore the true NSE separator, since the bhavcopy spells the
    symbols ``M&M`` and ``BAJAJ-AUTO`` rather than TradingView's ``M_M``.
    """
    if ":" in symbol:
        return restore_india_separator(symbol.split(":", 1)[1].upper())
    if strip_suffix:
        return restore_india_separator(
            symbol.replace(".NS", "").replace(".BO", "").upper()
        )
    # Keep the suffix, but translate only the base so the ``.NS``/``.BO`` tail
    # never lands inside the ampersand lookup.
    base, dot, suffix = symbol.upper().partition(".")
    return f"{restore_india_separator(base)}{dot}{suffix}"


def normalize_symbol(value: str) -> str:
    """Strip surrounding whitespace and reject an empty symbol.

    Shared body of the ``symbol`` pydantic field-validators on the row models.
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized
