"""Symbol vocabulary: TradingView-style symbols → yfinance / NSE forms.

Single home for the symbol-conversion rules that were previously duplicated
(and had drifted) across ``backtester.data``, ``rs_breakout`` and the
``unusual_volume`` package.
"""

from __future__ import annotations

# yfinance exchange suffixes we already understand.
_INDIA_YF_SUFFIXES = (".NS", ".BO")
# TradingView-only suffixes that should not be sent to Yahoo as-is.
# REITs/InvITs often appear as ``EMBASSY.RR`` on TV liquidity scans; Yahoo
# lists the same names under ``EMBASSY.NS``.
_INDIA_TV_ONLY_SUFFIXES = (".RR",)


def _india_yf_root(root: str) -> str:
    """Normalize an India ticker root for yfinance.

    TradingView / some scanners use underscores (``BAJAJ_AUTO``, ``NAM_INDIA``);
    Yahoo Finance uses hyphens (``BAJAJ-AUTO.NS``, ``NAM-INDIA.NS``).
    """
    return root.replace("_", "-")


def _strip_known_suffix(symbol: str, suffixes: tuple[str, ...]) -> tuple[str, str | None]:
    """Return ``(root, suffix)`` when ``symbol`` ends with a known suffix."""
    for suffix in suffixes:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)], suffix
    return symbol, None


def tv_to_yf(symbol: str, market: str) -> str:
    """Translate a TradingView-style symbol to a yfinance symbol.

    Examples:
      'NSE:RELIANCE' + india → 'RELIANCE.NS'
      'BSE:TCS'     + india → 'TCS.BO'
      'NASDAQ:AAPL' + us    → 'AAPL'
      'AAPL'        + us    → 'AAPL'
      'RELIANCE'    + india → 'RELIANCE.NS'
      'BAJAJ_AUTO'  + india → 'BAJAJ-AUTO.NS'
      'EMBASSY.RR'  + india → 'EMBASSY.NS'
    """
    sym = symbol.strip().upper()
    if ":" in sym:
        exch, rest = sym.split(":", 1)
        if exch in ("NSE", "BSE"):
            rest, _ = _strip_known_suffix(rest, _INDIA_TV_ONLY_SUFFIXES)
            rest = _india_yf_root(rest)
            if exch == "NSE":
                return f"{rest}.NS"
            return f"{rest}.BO"
        return rest

    if market == "india":
        root, yf_suffix = _strip_known_suffix(sym, _INDIA_YF_SUFFIXES)
        if yf_suffix is not None:
            return f"{_india_yf_root(root)}{yf_suffix}"
        root, tv_suffix = _strip_known_suffix(sym, _INDIA_TV_ONLY_SUFFIXES)
        if tv_suffix is not None:
            return f"{_india_yf_root(root)}.NS"
        if "." not in sym:
            return f"{_india_yf_root(sym)}.NS"
        return sym

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
    """
    if ":" in symbol:
        return symbol.split(":", 1)[1].upper()
    if strip_suffix:
        return symbol.replace(".NS", "").replace(".BO", "").upper()
    return symbol.upper()


def normalize_symbol(value: str) -> str:
    """Strip surrounding whitespace and reject an empty symbol.

    Shared body of the ``symbol`` pydantic field-validators on the row models.
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized
