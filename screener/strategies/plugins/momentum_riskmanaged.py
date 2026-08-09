"""Risk-managed cross-sectional momentum (Barroso; Daniel & Moskowitz).

Both papers keep Jegadeesh-Titman's 12-1 ranking and change only the *exposure*
taken to it, because raw winner-minus-loser momentum earns a good average
return alongside catastrophic left-tail episodes: Barroso & Santa-Clara report a
-96.69% maximum drawdown for 1927-2011 WML, cut to -45.20% once exposure is
scaled by forecast volatility. Daniel & Moskowitz condition on the state
instead, and report 0.68 -> 1.20 Sharpe from static to dynamic.

Long-only approximation
-----------------------
Both papers scale a dollar-neutral long-short portfolio continuously, including
levering *above* 1x when forecast volatility is low. This engine holds a fixed
number of long equity slots and cannot short or lever, so continuous scaling
becomes a binary exposure gate: hold the momentum winners in the normal state,
hold nothing in the state the paper scales exposure down in. That keeps the
economic mechanism - cut exposure exactly when momentum is fragile - and drops
the part of the result that comes from leverage and from the short leg. The
Sharpe improvement should therefore be smaller than the papers', because the
long leg alone is what remains.

Scaling exposure down means selling, so the risk state is both an entry gate and
an exit: an open position is closed when the state fires, not merely prevented
from being reopened. This matters more than it sounds. An entry-only version of
the same gate measured *worse* drawdowns than ungated momentum on Indian data -
positions opened before the crash rode it all the way down while the gate
blocked re-entry into the recovery, which is the opposite of what the papers do.

``momentum_12_1_volmanaged``
    Barroso & Santa-Clara. Realized volatility of the momentum portfolio itself
    is the risk forecast: each day the top-decile momentum names are held (from
    the *prior* day's ranks), their equal-weighted return series is formed, and
    its trailing 126-day realized volatility is ranked against its own trailing
    year. The strategy stands aside while that rank sits in the top quintile.
    Unlike a benchmark-volatility gate this reacts to momentum-specific risk,
    which is the paper's point: momentum's own variance is far more forecastable
    than its mean.

``momentum_12_1_dynamic``
    Daniel & Moskowitz. Their crash state is a bear market whose variance is
    elevated - momentum crashes happen when beaten-down losers rebound violently
    off a bottom, not merely when volatility is high. The strategy stands aside
    only when *both* hold on the benchmark: a negative trailing two-year return
    (their bear-market definition) and a high volatility state. In every other
    state it is plain 12-1 momentum.
"""

from __future__ import annotations

import pandas as pd

from screener.regime import vol_regime
from screener.strategies.cross_section import (
    attach_column,
    close_panel,
    high_risk_state,
    quantile_portfolio_returns,
    realized_volatility,
)
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import PrepareCtx, register_expression_strategy

# Barroso's forecast window is six months of daily momentum returns.
_VOL_WINDOW = 126
# The winner decile of the cross-section, matching Jegadeesh-Titman's sort.
_WINNER_QUANTILE = 0.1
# Daniel-Moskowitz define the bear state on the trailing two years of market
# returns, which is long enough that a single sharp correction is not a bear.
_BEAR_WINDOW = 504

ENTRY_VOLMANAGED = "mom_12_1 > 0 and not momentum_high_vol"
ENTRY_DYNAMIC = "mom_12_1 > 0 and not momentum_crash_state"
# The exits mirror the entry gates, so the risk state closes positions instead of
# only blocking new ones.
EXIT_VOLMANAGED = "momentum_high_vol"
EXIT_DYNAMIC = "momentum_crash_state"


def _momentum_frames(ctx: PrepareCtx) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Attach 12-1 momentum to every frame and return the score panel too."""
    prepared: dict[str, pd.DataFrame] = {}
    scores: dict[str, pd.Series] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            prepared[symbol] = bars
            continue
        frame = bars.copy()
        mom = momentum_12_1_score(frame["close"])
        frame["mom_12_1"] = mom
        frame["rank_score"] = mom
        prepared[symbol] = frame
        scores[symbol] = mom
    panel = pd.DataFrame(scores).sort_index() if scores else pd.DataFrame()
    return prepared, panel


def momentum_volatility_state(closes: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """Flag dates whose momentum-portfolio volatility is in its own top quintile."""
    portfolio = quantile_portfolio_returns(closes, scores, _WINNER_QUANTILE)
    if portfolio.empty:
        return pd.Series(dtype=bool)
    return high_risk_state(realized_volatility(portfolio, _VOL_WINDOW))


def crash_state(benchmark_close: pd.Series) -> pd.Series:
    """Flag Daniel-Moskowitz crash states: bear market *and* high volatility."""
    if benchmark_close.empty:
        return pd.Series(dtype=bool)
    two_year = benchmark_close / benchmark_close.shift(_BEAR_WINDOW) - 1.0
    bear = (two_year < 0).fillna(False)
    high_vol = vol_regime(benchmark_close).eq("high_vol")
    return bear & high_vol


def _prepare_volmanaged(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared, scores = _momentum_frames(ctx)
    closes = close_panel(ctx.bars_by_tv)
    state = momentum_volatility_state(closes, scores)
    if state.empty:
        ctx.warnings.append(
            "momentum portfolio volatility unavailable; volmanaged gate is open"
        )
    # Default False: an undated gap must not suppress entries on its own, the
    # warmup that precedes a defined percentile already does that.
    return attach_column(prepared, state, "momentum_high_vol", False)


def _prepare_dynamic(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared, _ = _momentum_frames(ctx)
    benchmark_bars = ctx.price_panel.get(ctx.benchmark)
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for momentum_12_1_dynamic: {ctx.benchmark}"
        )
        state = pd.Series(dtype=bool)
    else:
        state = crash_state(benchmark_bars["close"].astype(float))
    return attach_column(prepared, state, "momentum_crash_state", False)


def _volmanaged_lookback() -> int:
    # 12-1 momentum needs 252 bars before the panel's first score, and the
    # volatility percentile needs a further 252 daily momentum returns on top of
    # its own 126-bar window.
    return 252 + _VOL_WINDOW + 252


def _dynamic_lookback() -> int:
    # The two-year bear window is the binding one; the 252-bar volatility
    # distribution is shorter.
    return _BEAR_WINDOW


register_expression_strategy(
    "momentum_12_1_volmanaged",
    entry=ENTRY_VOLMANAGED,
    exit=EXIT_VOLMANAGED,
    prepare_bars=_prepare_volmanaged,
    required_lookback=_volmanaged_lookback,
)

register_expression_strategy(
    "momentum_12_1_dynamic",
    entry=ENTRY_DYNAMIC,
    exit=EXIT_DYNAMIC,
    prepare_bars=_prepare_dynamic,
    required_lookback=_dynamic_lookback,
)


__all__ = ["crash_state", "momentum_volatility_state"]
