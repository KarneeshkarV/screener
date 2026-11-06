"""Short-term reversal factor (Jegadeesh 1990).

Paper: Jegadeesh, "Evidence of Predictable Behavior of Security Returns",
Journal of Finance 45(3), 1990. https://doi.org/10.1111/j.1540-6261.1990.tb05110.x

Jegadeesh documented a strong one-month reversal: stocks that fell the most over
the past month tend to bounce the next month. This is the *same* anomaly the
12-1 momentum skip window exists to avoid, so it is the natural complement to
the Jegadeesh-Titman momentum strategies already in the repo.

Signal (causal, as-of bar ``t``):

    ret_21[t] = close[t] / close[t-21] - 1     # trailing one-month return

Selection: cross-sectional factor portfolio. ``rank_score = -ret_21`` so the
rolling backtester fills its ``--top`` slots with the biggest one-month losers.
The entry expression ``ret_21 < 0`` is the eligibility gate (only short-term
losers are long candidates); ranking does the work. A short ``--hold`` (≈21
trading days) matches the paper's monthly rebalance cadence.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 21  # one trading month


def _prepare_reversal(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        ret = close / close.shift(_WINDOW) - 1.0
        frame["ret_21"] = ret
        # Biggest losers rank first -> negate so the descending ranker picks them.
        frame["rank_score"] = -ret
        out[tv] = frame
    return out


def _lookback() -> int:
    return _WINDOW + 1


register_expression_strategy(
    "short_term_reversal",
    entry="ret_21 < 0",
    exit=None,
    prepare_bars=_prepare_reversal,
    required_lookback=_lookback,
)
