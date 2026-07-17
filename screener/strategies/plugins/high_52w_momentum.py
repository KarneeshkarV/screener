"""George-Hwang (2004) 52-week-high momentum.

Paper: George & Hwang, "The 52-Week High and Momentum Investing", Journal of
Finance 59(5), 2004. Nearness to the 52-week high forecasts returns better than
past returns: the winners are the names trading closest to their trailing
52-week high, and that proximity — not raw momentum — is the pricing signal.

Signal (causal, as-of bar ``t``):

    nearness[t] = close[t] / max( high[t-252 .. t-1] )

i.e. today's close divided by the highest HIGH over the prior 252 trading days.
The rolling max is ``shift(1)``-ed so bar ``t`` never sees its own high — the
baseline is the 52-week high *established before today*, which keeps the signal
strictly causal. ``nearness`` lives in ``(0, ~1]``: 1.0 means the close is
sitting on a fresh 52-week high, 0.85 means 15% below it.

If a frame has no ``high`` column (e.g. a close-only source), the rolling max
falls back to ``close`` so the strategy degrades to a 52-week-*close*-high proxy
rather than erroring; this is documented and a warning is emitted once so the
approximation is visible.

Selection: this is a real cross-sectional factor portfolio, so the prepared bars
carry ``rank_score = nearness`` and the rolling backtester fills its ``--top``
slots with the names *closest* to their 52-week high. The entry expression only
gates eligibility (``nearness >= 0.85`` — within 15% of the high); the ranker
then picks the closest eligible names.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_WINDOW = 252  # ~52 weeks of trading days
# Eligibility gate: only consider names within 15% of their 52-week high. This
# is exposed as a module constant so the entry expression and any downstream
# tooling share one source of truth.
_NEARNESS_THRESHOLD = 0.85


def nearness_52w_score(close: pd.Series, high: pd.Series | None = None) -> pd.Series:
    """Return the causal 52-week-high nearness series for one symbol.

    ``high`` defaults to ``close`` when no high column is available, degrading to
    a 52-week-close-high proxy (documented in the module docstring).
    """
    close = close.astype(float)
    baseline_source = close if high is None else high.astype(float)
    # Prior-bar rolling max: rolling(_WINDOW) at t spans [t-251, t]; shift(1)
    # moves it to [t-252, t-1] so today's own high is excluded.
    prior_high = baseline_source.rolling(_WINDOW, min_periods=_WINDOW).max().shift(1)
    return close / prior_high


def _prepare_high_52w(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    warned_no_high = False
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        high = frame["high"] if "high" in frame.columns else None
        if high is None and not warned_no_high:
            ctx.warnings.append(
                "high_52w_momentum: no 'high' column; falling back to close for "
                "the 52-week-high baseline (52-week-close-high proxy)."
            )
            warned_no_high = True
        nearness = nearness_52w_score(frame["close"], high)
        frame["nearness"] = nearness
        frame["rank_score"] = nearness
        out[tv] = frame
    return out


def _high_52w_lookback() -> int:
    # Need 252 prior HIGHs (indices t-252..t-1) before the first defined bar, so
    # the earliest usable bar is index 252 -> 253 bars of history.
    return _WINDOW + 1


register_expression_strategy(
    "high_52w_momentum",
    entry=f"nearness >= {_NEARNESS_THRESHOLD}",
    exit=None,
    prepare_bars=_prepare_high_52w,
    required_lookback=_high_52w_lookback,
)
