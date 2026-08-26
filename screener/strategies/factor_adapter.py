"""Backtest adapter for the shared price-only score layer.

Turns a :class:`screener.factors.PriceScoreSpec` into the ``prepare_bars``
hook the rolling backtester expects, writing ``rank_score`` (plus the recipe's
``aux_column`` so entry expressions can gate on the raw value).

This is deliberately the *only* place the backtest path turns a shared recipe
into ``rank_score``; the screen's counterpart is
:mod:`screener.scoring.bar_scores`. Both call the same
:func:`screener.factors.score_bars`, so the two paths cannot drift.
"""

from __future__ import annotations

import pandas as pd

from screener.factors import PriceScoreSpec, score_bars
from screener.strategies.spec import LookbackFn, PrepareBarsFn, PrepareCtx


def make_rank_score_prepare(spec: PriceScoreSpec) -> PrepareBarsFn:
    """Build a ``prepare_bars`` hook that writes ``rank_score`` from ``spec``."""

    def _prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            score = score_bars(spec, frame)
            if spec.aux_column:
                frame[spec.aux_column] = score
            # NaN stays NaN: rolling_candidates drops NaN rank_score names,
            # which is exactly the "not enough history -> ineligible" policy.
            frame["rank_score"] = score
            out[tv] = frame
        return out

    return _prepare


def make_rank_score_lookback(spec: PriceScoreSpec) -> LookbackFn:
    """Build the ``required_lookback`` callback for ``spec``."""

    def _lookback() -> int:
        return spec.required_lookback

    return _lookback


__all__ = ["make_rank_score_lookback", "make_rank_score_prepare"]
