"""Symbol vocabulary: TradingView-style symbols → yfinance / NSE forms.

Single home for the symbol-conversion rules that were previously duplicated
(and had drifted) across ``backtester.data``, ``rs_breakout`` and the
``unusual_volume`` package.
"""

from __future__ import annotations


def tv_to_yf(symbol: str, market: str) -> str:
    """Translate a TradingView-style symbol to a yfinance symbol.

    Examples:
      'NSE:RELIANCE' + india → 'RELIANCE.NS'
      'BSE:TCS'     + india → 'TCS.BO'
      'NASDAQ:AAPL' + us    → 'AAPL'
      'AAPL'        + us    → 'AAPL'
      'RELIANCE'    + india → 'RELIANCE.NS'
    """
    sym = symbol.strip().upper()
    if ":" in sym:
        exch, rest = sym.split(":", 1)
        if exch == "NSE":
            return f"{rest}.NS"
        if exch == "BSE":
            return f"{rest}.BO"
        return rest
    if market == "india" and "." not in sym:
        return f"{sym}.NS"
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
