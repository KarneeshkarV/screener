"""Published monthly signals adapted to the existing capital-constrained engine.

Calendar month ends use the observed benchmark session calendar.
Portfolio execution and universe changes are declared research adaptations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx

MONTHLY_RULES = (
    "alvarez_monthly_momentum10",
    "alvarez_monthly_sma10",
    "alvarez_monthly_dual10",
    "alvarez_monthly_either10",
    "alvarez_rotation_roc_hv63",
    "alvarez_rotation_roc_hv126",
    "alvarez_rotation_roc_hv252",
    "alvarez_three_factor",
)


def prepare_blog_monthly(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Prepare month-end selection with past-only monthly close and daily ranks."""
    calendar = pd.DatetimeIndex(
        sorted(set().union(*(set(b.index) for b in ctx.bars_by_tv.values())))
    )
    ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).last()
    month_ends = pd.DatetimeIndex(ends.values)
    result = {}
    scores: dict[str, dict[str, pd.Series]] = {}
    returns63 = {}
    returns20 = {}
    vols20 = {}
    for symbol, bars in ctx.bars_by_tv.items():
        frame = bars.copy()
        close = frame["close"]
        monthly = close.reindex(month_ends)
        momentum = monthly / monthly.shift(10) - 1
        trend = monthly - monthly.rolling(10).mean()
        flags = {
            "alvarez_monthly_momentum10": momentum.gt(0),
            "alvarez_monthly_sma10": trend.gt(0),
            "alvarez_monthly_dual10": momentum.gt(0) & trend.gt(0),
            "alvarez_monthly_either10": momentum.gt(0) | trend.gt(0),
        }
        for name, flag in flags.items():
            frame[f"{name}_entry"] = flag.reindex(frame.index, fill_value=False).astype(
                float
            )
            frame[f"{name}_exit"] = (
                ((~flag) & momentum.notna() & trend.notna())
                .reindex(frame.index, fill_value=False)
                .astype(float)
            )
        # Exit before new entries are processed. The source rotates at one
        # next-month open; this engine adaptation liquidates one session early.
        exit_dates = calendar[calendar.get_indexer(month_ends) - 1]
        frame["monthly_rotate_exit"] = frame.index.isin(exit_dates).astype(float)
        for length in (63, 126, 252):
            score = (
                (close / close.shift(length) - 1)
                * pd.Series(np.log(close / close.shift(1)), index=close.index)
                .rolling(length)
                .std(ddof=1)
                * np.sqrt(252)
                * 100
            )
            name = f"alvarez_rotation_roc_hv{length}"
            frame[f"{name}_entry"] = (
                close.gt(close.rolling(200).mean()) & frame.index.isin(month_ends)
            ).astype(float)
            scores.setdefault(name, {})[symbol] = score
        returns63[symbol] = close / close.shift(63) - 1
        returns20[symbol] = close / close.shift(20) - 1
        vols20[symbol] = (
            pd.Series(np.log(close / close.shift(1)), index=close.index)
            .rolling(20)
            .std(ddof=1)
        )
        result[symbol] = frame
    rank63 = ctx.mask_rank_reference(pd.DataFrame(returns63)).rank(
        axis=1, ascending=False, method="average"
    )
    rank20 = ctx.mask_rank_reference(pd.DataFrame(returns20)).rank(
        axis=1, ascending=False, method="average"
    )
    rankvol = ctx.mask_rank_reference(pd.DataFrame(vols20)).rank(
        axis=1, ascending=True, method="average"
    )
    composite = 0.4 * rank63 + 0.4 * rank20 + 0.2 * rankvol
    for symbol, frame in result.items():
        close = frame["close"]
        selected = (
            composite.rank(axis=1, ascending=True, method="first")[symbol]
            .le(1)
            .reindex(frame.index)
        )
        frame["alvarez_three_factor_entry"] = (
            selected
            & close.gt(close.rolling(200).mean())
            & frame.index.isin(month_ends)
        ).astype(float)
        frame["alvarez_three_factor_score"] = -composite[symbol].reindex(frame.index)
        for name in scores:
            frame[f"{name}_score"] = scores[name][symbol]
        if ctx.market == "us":
            for column in list(frame.columns):
                if column.endswith("_exit"):
                    frame[column] = frame[column].shift(1)
    return result
