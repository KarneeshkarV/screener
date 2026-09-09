"""Cross-sectional helpers shared by the paper-backed momentum families.

A ``prepare_bars`` hook receives every ticker's bars at once, which is what
makes panel-level quantities computable inside a strategy: the volatility of
the momentum portfolio itself (Barroso & Santa-Clara), the breadth of positive
momentum across the universe (Keller & Keuning), and so on. Those quantities
are date-indexed, not ticker-indexed, so they are computed once here and then
broadcast onto every ticker's frame as a column the entry expression can read.

Everything in this module is causal. Panel statistics for date ``t`` use only
bars up to and including ``t``, and any statistic that weights returns uses
weights formed on ``t-1``, so no bar is selected using its own return.
"""

from __future__ import annotations

import pandas as pd

# Trailing-window percentile of a risk statistic above which a state counts as
# "high". Matches ``screener.regime.VOL_HIGH_PERCENTILE`` so the two notions of
# an elevated-volatility state agree.
HIGH_RISK_PERCENTILE = 0.8
RISK_DIST_WINDOW = 252


def close_panel(bars_by_tv: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a ``date x ticker`` close-price frame over the union of bar dates.

    Missing cells stay NaN rather than being filled: a ticker that had not
    listed yet must not contribute a synthetic price to a cross-sectional rank.
    """
    columns = {
        symbol: bars["close"].astype(float)
        for symbol, bars in bars_by_tv.items()
        if bars is not None and not bars.empty and "close" in bars
    }
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()


def trailing_return(closes: pd.DataFrame, window: int, skip: int = 0) -> pd.DataFrame:
    """Return the ``window``-bar return ending ``skip`` bars before each date."""
    return closes.shift(skip) / closes.shift(window) - 1.0


def quantile_portfolio_returns(
    closes: pd.DataFrame, scores: pd.DataFrame, quantile: float = 0.1
) -> pd.Series:
    """Daily return of an equal-weighted top-``quantile`` portfolio of ``scores``.

    Weights for date ``t`` come from ranks observed at ``t-1``, so the return
    earned on ``t`` is never used to decide what was held into ``t``. This is
    the long leg of a cross-sectional momentum sort; its realized volatility is
    what the risk-managed variants scale exposure against.
    """
    if closes.empty or scores.empty:
        return pd.Series(dtype=float)
    returns = closes.pct_change()
    ranks = scores.rank(axis=1, pct=True, ascending=False)
    held = (ranks <= quantile).shift(1).fillna(False)
    masked = returns.where(held)
    return masked.mean(axis=1, skipna=True)


def realized_volatility(returns: pd.Series, window: int = 126) -> pd.Series:
    """Annualized realized volatility of a daily return series."""
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=0)
    return pd.Series(rolling_std * (252**0.5))


def high_risk_state(
    volatility: pd.Series,
    *,
    percentile: float = HIGH_RISK_PERCENTILE,
    window: int = RISK_DIST_WINDOW,
) -> pd.Series:
    """Flag dates whose volatility sits in the top tail of its own history.

    Ranking against a trailing window rather than a fixed threshold keeps the
    state definition free of hindsight about what "high volatility" turned out
    to mean over the sample.
    """
    if volatility.empty:
        return pd.Series(dtype=bool)
    rank = volatility.rolling(window, min_periods=window).rank(pct=True)
    return (rank >= percentile).fillna(False)


def positive_share(scores: pd.DataFrame) -> pd.Series:
    """Share of the universe with a positive score on each date, in 0..1.

    Dates where no ticker has a defined score are 0.0, which reads as "no
    breadth" and so keeps breadth-gated strategies flat during warmup.
    """
    if scores.empty:
        return pd.Series(dtype=float)
    defined = scores.notna().sum(axis=1)
    positive = (scores > 0).sum(axis=1)
    share = positive.divide(defined.where(defined > 0))
    return pd.Series(share.fillna(0.0))


def attach_column(
    prepared: dict[str, pd.DataFrame],
    values: pd.Series,
    name: str,
    default: float | bool,
) -> dict[str, pd.DataFrame]:
    """Broadcast a date-indexed series onto every ticker frame as ``name``.

    Reindexing forward-fills so a ticker trading on a date the panel statistic
    skipped (a holiday in one listing but not another) still sees the most
    recent known state rather than a NaN that would silently fail the gate.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in prepared.items():
        if bars is None or bars.empty:
            out[symbol] = bars
            continue
        frame = bars.copy()
        if values.empty:
            frame[name] = default
        else:
            aligned = values.reindex(frame.index, method="ffill")
            frame[name] = aligned.fillna(default)
        out[symbol] = frame
    return out


__all__ = [
    "HIGH_RISK_PERCENTILE",
    "attach_column",
    "close_panel",
    "high_risk_state",
    "positive_share",
    "quantile_portfolio_returns",
    "realized_volatility",
    "trailing_return",
]
