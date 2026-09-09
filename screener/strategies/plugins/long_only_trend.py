"""Long-only trend rules: Faber, Antonacci's absolute momentum, industry breakouts.

The strategies in this file come from the branch of the literature written for
investors who never short. That changes what the evidence is about. A long-short
factor study reports the spread between winners and losers; these papers report
what an investor actually held, so they publish the statistic the spread studies
usually omit - the drawdown of a funded portfolio.

Their shared mechanism is an exposure switch rather than a selection rule: hold
the asset while its own trend is intact, hold Treasury bills when it is not. In
this engine the defensive leg is uninvested cash, which earns nothing, so every
variant here gives up the bill yield the papers credit. That gap is largest in
exactly the stretches the rules are designed for, and it is the main reason
these runs should read below their published counterparts.

Slot ranking is an addition, not a translation. Faber times one index and
Zarattini-Antonacci hold every industry that has broken out; neither ranks. A
fixed number of slots forces a choice about which qualifying names to hold, so
each variant documents the ranker it uses and why.

``faber_sma10``
    Faber's ten-month moving average, applied per name rather than to one index:
    hold while the close is above its 210-day (ten-month) simple average, sell
    when it drops below. Faber reports 0.55 Sharpe with a -50.29% drawdown for
    1901-2005 and 0.80 with -16.73% for the out-of-sample 2006-2016 decade, both
    on monthly observations of a single US index. Slots go to the strongest
    trailing 12-month return per unit of volatility.

``absolute_momentum``
    Antonacci's absolute-momentum overlay: hold while the trailing 12-month
    return beat the Treasury-bill return over the same window. Unlike
    ``dual_momentum_gem`` this uses the full twelve months with no skip, which is
    the paper's own specification - time-series momentum is not contaminated by
    the one-month reversal that the cross-sectional skip exists to avoid. He
    reports 0.55 Sharpe and -22.90% month-end drawdown on MSCI US for 1974-2012,
    against 0.37 and -50.65% for buy-and-hold.

``industry_trend_breakout``
    Zarattini & Antonacci's channel rule, applied to stocks rather than to 48
    industry portfolios: buy when the close breaks the prior 20-day high, sell
    when it breaks the prior 40-day low. Their July 1926-March 2024 test reports
    1.39 Sharpe and roughly -33% drawdown against 0.63 and -84% for the passive
    market - but with volatility-scaled positions and gross exposure up to 200%,
    neither of which a fixed-slot unlevered book can reproduce. Slots are ranked
    by inverse 14-day volatility, the closest available stand-in for their
    risk-budgeted sizing; pair it with ``--sizing inverse_vol`` to size positions
    the same way.
"""

from __future__ import annotations

import pandas as pd

from screener.risk_free import annualized_rate, compounded_hurdle
from screener.strategies.cross_section import attach_column, close_panel
from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.spec import PrepareCtx, register_expression_strategy

# Ten months of trading days, Faber's signal window.
_FABER_WINDOW = 210
_YEAR = 252
# Zarattini-Antonacci channel lengths and volatility window.
_BREAKOUT_WINDOW = 20
_STOP_WINDOW = 40
_VOL_WINDOW = 14

ENTRY_FABER = f"close > sma(close, {_FABER_WINDOW})"
EXIT_FABER = f"close < sma(close, {_FABER_WINDOW})"

ENTRY_ABSOLUTE = "mom_12 > rf_hurdle"
EXIT_ABSOLUTE = "mom_12 <= rf_hurdle"

# The channel bands are computed on prior bars in ``prepare_bars`` because the
# expression language's ``highest``/``lowest`` include the current bar, which
# would make a breakout above the running high unreachable by construction.
ENTRY_BREAKOUT = "close > donchian_upper"
EXIT_BREAKOUT = "close < donchian_lower"


def trailing_year_return(close: pd.Series) -> pd.Series:
    """Trailing 12-month return with no skip month, the time-series signal."""
    close = close.astype(float)
    return close / close.shift(_YEAR) - 1.0


def channel_bands(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the prior-bar 20-day upper and 40-day lower Donchian bands."""
    upper = bars["high"].astype(float).rolling(_BREAKOUT_WINDOW).max().shift(1)
    lower = bars["low"].astype(float).rolling(_STOP_WINDOW).min().shift(1)
    return upper, lower


def _prepare_faber(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[symbol] = bars
            continue
        frame = bars.copy()
        vol = realized_volatility(frame["close"])
        frame["rank_score"] = (trailing_year_return(frame["close"]) / vol).where(
            vol > 0
        )
        out[symbol] = frame
    return out


def _prepare_absolute(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            prepared[symbol] = bars
            continue
        frame = bars.copy()
        mom = trailing_year_return(frame["close"])
        frame["mom_12"] = mom
        frame["rank_score"] = mom
        prepared[symbol] = frame
    closes = close_panel(ctx.bars_by_tv)
    index = pd.DatetimeIndex(closes.index) if not closes.empty else pd.DatetimeIndex([])
    rate = annualized_rate(ctx.market, index, ctx.fetcher, ctx.start, ctx.end)
    return attach_column(prepared, compounded_hurdle(rate), "rf_hurdle", 0.0)


def _prepare_breakout(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[symbol] = bars
            continue
        frame = bars.copy()
        upper, lower = channel_bands(frame)
        frame["donchian_upper"] = upper
        frame["donchian_lower"] = lower
        vol = short_volatility(frame["close"])
        # Inverse volatility, so the calmest qualifying breakouts take the slots.
        frame["rank_score"] = (1.0 / vol).where(vol > 0)
        out[symbol] = frame
    return out


def _faber_lookback() -> int:
    # The 252-bar ranking window is longer than the 210-bar signal.
    return _YEAR + 1


def _absolute_lookback() -> int:
    return _YEAR


def _breakout_lookback() -> int:
    # The 40-bar stop channel is the longest window, plus one bar for its shift.
    return _STOP_WINDOW + 1


def short_volatility(close: pd.Series) -> pd.Series:
    """The paper's 14-day realized volatility, its risk-budgeting input.

    Far shorter than the low-volatility plugin's annual window, so it is
    computed here rather than reused from there.
    """
    returns = close.astype(float).pct_change()
    return pd.Series(returns.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW).std(ddof=0))


register_expression_strategy(
    "faber_sma10",
    entry=ENTRY_FABER,
    exit=EXIT_FABER,
    prepare_bars=_prepare_faber,
    required_lookback=_faber_lookback,
)

register_expression_strategy(
    "absolute_momentum",
    entry=ENTRY_ABSOLUTE,
    exit=EXIT_ABSOLUTE,
    prepare_bars=_prepare_absolute,
    required_lookback=_absolute_lookback,
)

register_expression_strategy(
    "industry_trend_breakout",
    entry=ENTRY_BREAKOUT,
    exit=EXIT_BREAKOUT,
    prepare_bars=_prepare_breakout,
    required_lookback=_breakout_lookback,
)


__all__ = ["channel_bands", "short_volatility", "trailing_year_return"]
