"""Antonacci dual momentum: relative strength with an absolute trend filter.

References: Gary Antonacci, *Dual Momentum Investing* (2014), and "Risk
Premia Harvesting Through Dual Momentum", Journal of Portfolio Management
(2017). Dual momentum combines two distinct signals:

* relative momentum ranks names against one another; and
* absolute momentum permits risk-asset exposure only when both the name and
  the market benchmark have positive momentum.

Signal (causal, as-of bar ``t``):

    mom_12_1[t] = close[t-21] / close[t-252] - 1
    dual_ok[t] = 1 if mom_12_1[t] > 0 and benchmark_mom_12_1[t] > 0 else 0

``rank_score`` is the raw name-level 12-1 momentum, so the rolling backtester
fills ``--top`` slots with the strongest eligible names. ``dual_ok > 0`` is
only the entry gate: when benchmark momentum is non-positive, the strategy has
no eligible entries and moves to cash as existing positions finish their normal
exit path.

The required lookback is 253 bars. The oldest momentum leg needs the close 252
bars before ``t``; requesting 253 bars supplies that prior history together
with the signal bar. Benchmark closes use the shared low-beta acquisition
ladder (prepared price panel, universe bars, then the injected fetcher). If no
benchmark is available, every non-empty frame receives NaN signals and a
warning, yielding no entries rather than silently bypassing the market filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.low_beta import _benchmark_closes
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_REQUIRED_LOOKBACK = 253


def _prepare_dual_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach causal relative- and absolute-momentum signals to each frame."""
    bench_close = _benchmark_closes(ctx)
    out: dict[str, pd.DataFrame] = {}

    if bench_close is None:
        ctx.warnings.append(
            f"dual_momentum: benchmark {ctx.benchmark!r} unavailable; absolute "
            "momentum cannot be computed, so the strategy yields no entries."
        )
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            frame["mom_12_1"] = np.nan
            frame["dual_ok"] = np.nan
            frame["rank_score"] = np.nan
            out[tv] = frame
        return out

    bench_mom = momentum_12_1_score(bench_close)
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        mom = momentum_12_1_score(frame["close"])
        aligned_bench_mom = bench_mom.reindex(frame.index)
        frame["mom_12_1"] = mom
        frame["dual_ok"] = ((mom > 0) & (aligned_bench_mom > 0)).astype(float)
        frame["rank_score"] = mom
        out[tv] = frame
    return out


def _dual_momentum_lookback() -> int:
    """Return the signal bar plus the 252 prior closes used by momentum."""
    return _REQUIRED_LOOKBACK


register_expression_strategy(
    "dual_momentum",
    entry="dual_ok > 0",
    exit=None,
    prepare_bars=_prepare_dual_momentum,
    required_lookback=_dual_momentum_lookback,
)
