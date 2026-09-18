"""The one definition of "this window rests on prices a market actually set".

Every factor in this package that divides one close by another, or measures
the dispersion of a run of closes, is only meaningful when the closes it reads
are prices rather than placeholders. Two ways that fails, and neither shows up
as missing data - the frame is full-length and every bar carries a close:

* **A stub quote.** A venue carries a dormant OTC or grey-market line at a flat
  sub-cent value. As the denominator of a return ratio it manufactures a
  five-figure percentage; as the input to a volatility estimate it reports a
  perfectly calm stock. One observed name divided a $4.26 close by a $0.0001
  stub and scored +4,259,800%, the top of a 13,326 name momentum field.
* **A series that is mostly not trading.** The vendor repeats the last close
  through months of no trades, so the endpoints are stale marks rather than
  prices. Bar count cannot tell such a series from a liquid one; the volume
  column can.

The rules live here, in a leaf module with no package imports, because the
factors that need them sit in unrelated subpackages - price-score recipes,
relative strength, and the strategy plugins - and a gate that is copied is a
gate that drifts.

Two shapes of factor need two shapes of question:

* :func:`tradeable_window` answers it for a *ratio*, which reads exactly two
  closes. Only those two legs have to clear the price floor.
* :func:`tradeable_span` answers it for a factor that reads *every* close in a
  window, such as a realized-volatility or moving-average spread. Every close
  in the span has to clear the floor, because any one of them can carry the
  defect into the result.

Both take ``volume`` as optional. A close-only frame cannot answer the
liquidity question, so only the price rule applies there. That is stated
rather than assumed away: the gate is as strong as the columns allow.
"""

from __future__ import annotations

import pandas as pd

#: Smallest close that counts as a price. Below one cent a quote is not a
#: market: US OTC and grey-market lines are carried at a flat $0.0001 stub for
#: months at a time, volume zero. A cent is also the minimum tick on both
#: markets this screener covers, so no real quote is excluded by the floor.
MIN_TRADEABLE_PRICE = 0.01

#: Fraction of the sessions a factor's window spans that must carry non-zero
#: volume before the window counts as a market. Half is deliberately
#: permissive - it keeps thinly traded but real names and cuts only series
#: that are mostly not trading.
MIN_TRADED_FRACTION = 0.5


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _traded_fraction_ok(
    volume: pd.Series | None,
    *,
    span: int,
    skip: int,
) -> pd.Series | None:
    """Rolling share of traded sessions over ``span``, offset by ``skip``.

    ``None`` when there is no volume column, which callers read as "this rule
    cannot be applied" rather than as a pass or a fail.
    """
    if volume is None:
        return None
    traded = (_numeric(volume) > 0).astype(float)
    fraction = traded.shift(skip).rolling(max(int(span), 1)).mean()
    return fraction >= MIN_TRADED_FRACTION


def tradeable_window(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    lookback: int,
    skip: int = 0,
) -> pd.Series:
    """Boolean mask: bars whose ``close[t-skip] / close[t-lookback]`` is a return.

    Both legs of the ratio must be at or above :data:`MIN_TRADEABLE_PRICE`, and
    at least :data:`MIN_TRADED_FRACTION` of the sessions the window spans must
    have traded. ``skip`` is the near leg's offset - zero for a plain trailing
    return, 21 for a 12-1 momentum window that drops the reversal month.
    """
    prices = _numeric(close)
    eligible = (prices.shift(skip) >= MIN_TRADEABLE_PRICE) & (
        prices.shift(lookback) >= MIN_TRADEABLE_PRICE
    )
    # Sessions strictly inside the window, i.e. the ones the return is made
    # of. ``shift(skip)`` puts the near leg at bar ``t``; the window then
    # reaches back ``lookback - skip`` sessions to the far leg.
    traded = _traded_fraction_ok(volume, span=int(lookback) - int(skip), skip=int(skip))
    if traded is None:
        return eligible
    return eligible & traded


def tradeable_span(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    window: int,
) -> pd.Series:
    """Boolean mask: bars whose trailing ``window`` closes are all real prices.

    For factors that read the whole run rather than two endpoints - realized
    volatility, moving-average spreads. A single sub-cent stub anywhere in the
    span is enough to corrupt them, so the floor applies to the window's
    minimum rather than to its ends.
    """
    prices = _numeric(close)
    span = max(int(window), 1)
    eligible = prices.rolling(span).min() >= MIN_TRADEABLE_PRICE
    traded = _traded_fraction_ok(volume, span=span, skip=0)
    if traded is None:
        return eligible
    return eligible & traded


__all__ = [
    "MIN_TRADEABLE_PRICE",
    "MIN_TRADED_FRACTION",
    "tradeable_span",
    "tradeable_window",
]
