"""Long-term reversal factor (De Bondt & Thaler 1985).

Paper: De Bondt & Thaler, "Does the Stock Market Overreact?", Journal of
Finance 40(3), 1985. https://doi.org/10.1111/j.1540-6261.1985.tb05004.x

The founding contrarian paper: portfolios of extreme 3- to 5-year losers beat
extreme winners by 25 percentage points over the subsequent 36 months (and vice
versa in reverse) — the overreaction hypothesis. Where short_term_reversal
(Jegadeesh 1990) captures the 1-month bounce, this captures the multi-year
mean reversion.

Signal (causal, as-of bar ``t``):

    ret_756[t] = close[t] / close[t-756] - 1     # trailing three-year return

Selection: cross-sectional factor portfolio. ``rank_score = -ret_756`` so the
rolling backtester fills its ``--top`` slots with the biggest three-year
losers. The entry expression ``ret_756 < 0`` gates eligibility. Expect deep
value/beaten-down names (PSU banks, old-economy) rather than momentum leaders.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 756  # three trading years


def _prepare_lt_reversal(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        ret = close / close.shift(_WINDOW) - 1.0
        frame["ret_756"] = ret
        # Biggest three-year losers rank first -> negate for the descending ranker.
        frame["rank_score"] = -ret
        out[tv] = frame
    return out


def _lookback() -> int:
    return _WINDOW + 1


register_expression_strategy(
    "long_term_reversal",
    entry="ret_756 < 0",
    exit=None,
    prepare_bars=_prepare_lt_reversal,
    required_lookback=_lookback,
)
