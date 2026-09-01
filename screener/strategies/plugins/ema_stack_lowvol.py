"""The default screen, and the one filter that measurably improves it.

``ema_stack`` is the default ``screener screen`` criterion (``ema``) written as
a backtestable expression: EMA5 > EMA20 > EMA100 > EMA200. It is registered on
its own so the screen the CLI actually runs has a strategy to be measured as.

``ema_stack_lowvol`` is that same screen with one addition: among the names
passing the stack, prefer the ones with the lowest **downside** volatility.

    downside_vol_60[t] = sqrt( mean( min(logret, 0)^2 , 60 ) ) * sqrt(252)

Lower is better, so the prepared bars carry ``rank_score = -downside_vol_60``
and the rolling backtester fills its ``--top`` slots with the calmest names in
the stack. Entry is unchanged: the volatility leg only orders the candidates,
it never admits a name the stack rejected.

Why downside deviation rather than full realized volatility (``low_volatility``
already offers the latter): upside dispersion is not a risk a long book should
be penalized for, and the two are close substitutes here anyway - they correlate
0.89 cross-sectionally, and downside deviation scored slightly better on both
samples tested.

Evidence
--------
Measured on ``nifty_midsmall400_pit`` (820 point-in-time members,
2021-08-18 to 2026-08-17), 4 expanding walk-forward folds, first 250 days held
back, equal weight, 21-bar refresh, 20 bps one-way cost, all figures pooled
out-of-sample. Full write-up in ``findings/ema_stack_lowvol.md``.

The volatility leg was the only one of 55 causal candidate features - across
trend, trend-quality, volatility, acceleration, relative-strength, liquidity and
spectral families - that stayed positive under both a change of parameter and a
change of sample. Trend-quality features as a class added nothing, which is what
one should expect: the stack is already a trend filter, so more trend
information is redundant with the screen itself.

Read this before using it
-------------------------
The gain is in Sharpe, not in Calmar. At matched candidate count the filter
moved Sharpe from 1.50 to 1.70 but Calmar from 1.11 to 1.09, and cost about
three points of CAGR. It suppresses day-to-day volatility considerably more than
it suppresses drawdown. On a book judged by drawdown rather than by volatility
this filter earns nothing, and ``ema_stack`` is the better choice.

No second filter stacked on top of it: at matched candidate count, tightening
this one beat every two-factor combination tried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)

# The default `ema` criterion. EMA200 > 0 is the has-enough-history gate, and
# mirrors the live TradingView filter one-for-one.
ENTRY_STACK = (
    "ema(close, 5) > ema(close, 20) "
    "and ema(close, 20) > ema(close, 100) "
    "and ema(close, 100) > ema(close, 200) "
    "and ema(close, 200) > 0"
)

_VOL_WINDOW = 60  # ~3 months
_SLOWEST_EMA = 200
_ANNUAL = 252.0


def downside_volatility(close: pd.Series, window: int = _VOL_WINDOW) -> pd.Series:
    """Annualized trailing semi-deviation of negative log returns.

    Semi-deviation about zero rather than about the mean: the quantity of
    interest is how hard the name falls, not how far its losses sit from its own
    average. Causal - the window at bar ``t`` ends at ``t``.
    """
    prices = close.astype(float)
    log_prices = pd.Series(np.log(prices.where(prices > 0.0)), index=prices.index)
    log_returns = log_prices.diff()
    losses = log_returns.where(log_returns < 0.0, 0.0).where(log_returns.notna())
    mean_square = (losses**2).rolling(window, min_periods=window).mean()
    return pd.Series(np.sqrt(mean_square) * np.sqrt(_ANNUAL), index=prices.index)


def _prepare_lowvol(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        vol = downside_volatility(frame["close"])
        frame["downside_vol_60"] = vol
        # Lower downside volatility ranks higher -> negate so the descending
        # ranker picks the calmest names in the stack first.
        frame["rank_score"] = -vol
        out[tv] = frame
    return out


def _stack_lookback() -> int:
    # The EMA200 leg of the entry needs 200 bars before it is defined; the
    # 60-bar downside window plus its one diff needs 61, so the EMA dominates.
    return _SLOWEST_EMA


# No ``tv_prefilter``, deliberately, even though the `ema` criterion spells
# this very stack in TradingView columns. ``ema()`` seeds at ``out[0] = x[0]``,
# so an EMA200 read off a warmed-up 500-bar panel and one read off a vendor's
# full listing history are different numbers, and near a crossing they order
# the legs differently in *both* directions. A prefilter is only allowed to
# remove names the bar rule would also remove; this one cannot promise that, so
# it does not run. The cost is that a default `-c ema` screen fetches bars for
# the whole universe, which is the price of the two paths agreeing.
register_expression_strategy(
    "ema_stack",
    entry=ENTRY_STACK,
    exit=None,
    required_lookback=_stack_lookback,
    profile=DEFAULT_STRATEGY_PROFILE,
)

register_expression_strategy(
    "ema_stack_lowvol",
    entry=ENTRY_STACK,
    exit=None,
    prepare_bars=_prepare_lowvol,
    required_lookback=_stack_lookback,
    profile=DEFAULT_STRATEGY_PROFILE,
)
