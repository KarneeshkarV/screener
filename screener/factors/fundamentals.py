"""The fundamental vocabulary shared by the merge and the feature layer.

A fundamental value is only safe to read at bar ``t`` if it was *knowable* at
``t``: the filing date plus the market's reporting lag. The backtester enforces
that once, in
:func:`screener.backtester.fundamentals.merge_fundamentals_into_bars`, which is
the single sanctioned door for a fundamental column onto a bar frame.

This module holds the two things both sides of that door need: the column
names themselves, and a provenance stamp the merge writes and
:class:`screener.factors.BarFeatures` reads. The stamp is what makes "this
column came through the lagged join" checkable instead of assumed - a
fundamental column on a frame with no stamp was assigned directly, which is
lookahead, and the feature layer refuses to read it.

It lives under ``screener.factors`` rather than under the backtester because
both the backtester and the shared score layer import it, and the sanctioned
dependency direction is backtester -> factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Every fundamental column this repo knows how to produce. A frame carrying
#: one of these names is claiming to hold point-in-time fundamental data, so
#: it must also carry the provenance stamp that proves it.
FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "pe_ttm",
    "pb_ttm",
    "roe_ttm",
    "debt_to_equity",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "revenue_up_3q",
    "market_cap",
)

#: ``DataFrame.attrs`` key the stamp is written under.
FUNDAMENTAL_PROVENANCE_KEY = "fundamentals"


@dataclass(frozen=True)
class FundamentalProvenance:
    """How the fundamental columns on a frame got there."""

    #: The columns the merge actually wrote, in request order.
    columns: tuple[str, ...]
    #: The reporting lag, in calendar days, already applied to the filing
    #: dates before the forward-fill. Recorded so a reader can state the
    #: assumption its scores rest on.
    filing_lag_days: int


def stamp_fundamentals(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    filing_lag_days: int,
) -> pd.DataFrame:
    """Record on ``frame`` that ``columns`` arrived through the lagged join.

    Mutates and returns ``frame``. Call it *after* the join: pandas drops
    ``attrs`` across :meth:`~pandas.DataFrame.join`, so a stamp written before
    would not survive.
    """
    frame.attrs[FUNDAMENTAL_PROVENANCE_KEY] = FundamentalProvenance(
        columns=tuple(columns),
        filing_lag_days=int(filing_lag_days),
    )
    return frame


def fundamental_provenance(frame: pd.DataFrame) -> FundamentalProvenance | None:
    """Read the stamp back, or ``None`` when the frame carries none."""
    stamp = frame.attrs.get(FUNDAMENTAL_PROVENANCE_KEY)
    return stamp if isinstance(stamp, FundamentalProvenance) else None


__all__ = [
    "FUNDAMENTAL_COLUMNS",
    "FUNDAMENTAL_PROVENANCE_KEY",
    "FundamentalProvenance",
    "fundamental_provenance",
    "stamp_fundamentals",
]
