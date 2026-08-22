"""Shared types and statistics for the trend-filter study.

Split out of the runner scripts so the cached panel artefact can be unpickled
by anything that imports the package, not only by the script that wrote it.

The evaluation contract, in one place because everything else depends on it:

**A filter is a cross-sectional rank cut.** On each date, take the names that
pass the base screen, rank them by one feature at one parameter setting, and
keep the top ``q`` fraction. Every filter is therefore comparable to every
other, and the surviving count is controlled by construction - so a difference
in forward return is attributable to *which* names the feature picked, not to
how many it happened to keep. An absolute threshold would confound the two and
would not survive a regime change.

Features whose ``higher_is_stronger`` is False are ranked ascending, so "top q"
always means "the q fraction this feature considers best".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_TRADING_DAYS = 252.0


def setting_key(name: str, params: dict[str, Any]) -> str:
    """Stable column name for one feature at one parameter setting.

    Shared by the panel builder and the evaluator so a key written by one is
    always readable by the other.
    """
    if not params:
        return name
    parts = "_".join(f"{k}{_fmt(v)}" for k, v in sorted(params.items()))
    return f"{name}__{parts}"


def _fmt(value: Any) -> str:
    if isinstance(value, tuple):
        return "-".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value)


@dataclass(frozen=True)
class PanelSet:
    """Point-in-time panels the evaluation reads. All frames are date x ticker."""

    base: pd.DataFrame
    close: pd.DataFrame
    forward: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    settings: dict[str, dict[str, Any]]
    regime: pd.DataFrame
    benchmark: pd.Series
    meta: dict[str, Any]


def selection_mask(
    base: pd.DataFrame,
    scores: pd.DataFrame | None,
    q: float,
    *,
    ascending: bool = False,
) -> pd.DataFrame:
    """Names surviving the base screen and the top-``q`` cut of ``scores``.

    ``scores is None`` returns the base itself, which is how the base arm is
    evaluated through exactly the same code path as every filter.

    Ranking is per-date and percentile-based, so it needs no threshold tuning
    and adapts to whatever the cross-section looks like that day. A name with a
    missing score is dropped rather than ranked last: "unknown" is not evidence
    of weakness, and treating it as such would quietly filter out every name
    still inside its feature warmup.

    Ties take the *best* rank in their block (``method="min"``), so a block of
    equal scores is either kept whole or dropped whole. This matters for the
    discrete features: with average-ranked ties, a binary flag whose zeros make
    up 95% of the cross-section gives every one of them a mid percentile, and a
    30% cut then discards all of them and keeps a handful of leftovers. That
    produces three-name portfolios with spectacular, meaningless Sharpe ratios.
    Keeping tied blocks intact means a discrete feature simply overshoots ``q``
    - an honest "this feature cannot cut that fine" - instead of silently
    turning into a different experiment.
    """
    if scores is None:
        return base
    masked = scores.where(base)
    ranks = masked.rank(
        axis=1, pct=True, ascending=ascending, na_option="keep", method="min"
    )
    return base & (ranks <= q)


def _annualize(daily: pd.Series) -> tuple[float, float, float]:
    """Return ``(cagr, sharpe, sortino)`` from a daily return series."""
    clean = daily.dropna()
    if clean.empty:
        return (np.nan, np.nan, np.nan)
    growth = float(np.prod(1.0 + clean.to_numpy()))
    years = len(clean) / _TRADING_DAYS
    cagr = growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else np.nan
    std = float(clean.std(ddof=1))
    sharpe = float(clean.mean() / std * np.sqrt(_TRADING_DAYS)) if std > 0 else np.nan
    downside = clean[clean < 0.0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = (
        float(clean.mean() / dstd * np.sqrt(_TRADING_DAYS))
        if dstd and dstd > 0
        else np.nan
    )
    return (cagr, sharpe, sortino)


def _max_drawdown(daily: pd.Series) -> float:
    clean = daily.dropna()
    if clean.empty:
        return np.nan
    curve = (1.0 + clean).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def equal_weight_returns(
    mask: pd.DataFrame, close: pd.DataFrame, *, rebalance: int, cost_bps: float
) -> tuple[pd.Series, float]:
    """Daily returns of an equal-weight book over ``mask``, and its turnover.

    Holdings are refreshed every ``rebalance`` bars from the mask on the
    rebalance date and held in between, which is what a periodic screen
    actually does. Weights are formed on date ``t`` and earn the return from
    ``t`` to ``t+1``, so no position ever earns a return that predates the
    signal that opened it.

    ``cost_bps`` is charged on one-way turnover at each rebalance.
    """
    daily = close.pct_change()
    rebalance_dates = mask.index[::rebalance]
    weights = mask.astype(float)
    counts = weights.sum(axis=1)
    weights = weights.div(counts.where(counts > 0), axis=0).fillna(0.0)
    # Hold the rebalance-date weights until the next rebalance.
    held = weights.reindex(rebalance_dates).reindex(mask.index).ffill().fillna(0.0)
    # Shift so the weights known at t earn t -> t+1.
    gross = (held.shift(1) * daily).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1) / 2.0
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost.fillna(0.0)
    invested = held.sum(axis=1).shift(1)
    net = net.where(invested > 0.0)
    annual_turnover = float(turnover.sum() / (len(mask) / _TRADING_DAYS))
    return net, annual_turnover


def evaluate_mask(
    mask: pd.DataFrame,
    panels: PanelSet,
    *,
    rebalance: int,
    cost_bps: float,
    dates: pd.Index | None = None,
) -> dict[str, Any]:
    """Every reported statistic for one filter, over ``dates``.

    Forward-return statistics are cross-sectional: they pool every
    (date, surviving name) pair, which answers "what does a name that passes
    this filter go on to do". The portfolio statistics answer the different
    question of what a book built on it would have earned after costs.
    """
    if dates is not None:
        mask = mask.loc[mask.index.intersection(dates)]
    counts = mask.sum(axis=1)
    row: dict[str, Any] = {
        "n_dates": int(len(mask)),
        "mean_survivors": float(counts.mean()) if len(counts) else np.nan,
        "median_survivors": float(counts.median()) if len(counts) else np.nan,
    }

    for label, frame in panels.forward.items():
        selected = frame.reindex(index=mask.index, columns=mask.columns).where(mask)
        values = selected.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            row[f"fwd_{label}_mean"] = np.nan
            row[f"fwd_{label}_median"] = np.nan
            row[f"fwd_{label}_win"] = np.nan
            row[f"fwd_{label}_n"] = 0
            continue
        row[f"fwd_{label}_mean"] = float(values.mean())
        row[f"fwd_{label}_median"] = float(np.median(values))
        row[f"fwd_{label}_win"] = float((values > 0.0).mean())
        row[f"fwd_{label}_n"] = int(values.size)

    net, turnover = equal_weight_returns(
        mask, panels.close, rebalance=rebalance, cost_bps=cost_bps
    )
    cagr, sharpe, sortino = _annualize(net)
    max_dd = _max_drawdown(net)
    row.update(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=(cagr / abs(max_dd)) if max_dd and max_dd < 0 else np.nan,
        turnover=turnover,
        cost_bps=cost_bps,
    )

    # Conditional performance. Regime labels are causal, so slicing on them is
    # a report of what happened in each state, not a filter fitted to it.
    regime = panels.regime.reindex(net.index)
    for column in ("trend", "vol"):
        for state, group in net.groupby(regime[column]):
            if state in ("unknown", None) or len(group.dropna()) < 20:
                continue
            _, state_sharpe, _ = _annualize(group)
            row[f"sharpe_{state}"] = state_sharpe
            row[f"ret_{state}"] = float(group.mean() * _TRADING_DAYS)
    return row


def walk_forward_folds(
    dates: pd.Index, *, n_folds: int, min_train: int
) -> list[tuple[pd.Index, pd.Index]]:
    """Expanding-window folds: train on everything before, test on the next slice.

    Expanding rather than rolling because a screen's parameters are not expected
    to be re-fitted from scratch each year, and because it never trains on a
    period that follows its own test slice.
    """
    total = len(dates)
    if total < min_train + n_folds:
        return []
    test_size = (total - min_train) // n_folds
    folds: list[tuple[pd.Index, pd.Index]] = []
    for fold in range(n_folds):
        train_end = min_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else total
        folds.append((dates[:train_end], dates[train_end:test_end]))
    return folds


__all__ = [
    "PanelSet",
    "setting_key",
    "equal_weight_returns",
    "evaluate_mask",
    "selection_mask",
    "walk_forward_folds",
]
