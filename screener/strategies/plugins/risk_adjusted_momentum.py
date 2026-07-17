"""NSE Nifty200 Momentum 30 — risk-adjusted cross-sectional momentum.

Reference: NSE Indices, "Nifty200 Momentum 30 Index" methodology whitepaper
(Sep 2020). India's flagship momentum factor index. Unlike the Jegadeesh-Titman
``momentum_12_1`` port (which skips the most recent month and is *not*
volatility-normalized), the NSE construction (a) uses raw 6-month and 12-month
total returns, (b) divides each by the stock's annualized daily-return
volatility to get *risk-adjusted* returns, and (c) blends the two horizons via an
equal-weighted average of their cross-sectional Z-scores.

Per-symbol causal series (as-of bar ``t``, backward-looking only):

    ret_6m[t]  = close[t] / close[t-126] - 1          # 126 td ~ 6 months
    ret_12m[t] = close[t] / close[t-252] - 1          # 252 td ~ 12 months
    vol_ann[t] = std( log(close).diff() , 252 ) * sqrt(252)

``vol_ann`` uses daily log returns (``diff`` of ``log(close)``) with a rolling
std over 252 returns (``min_periods=252``, pandas-default ``ddof=1`` sample std)
annualized by ``sqrt(252)``. The risk-adjusted legs are:

    radj_6m[t]  = ret_6m[t]  / vol_ann[t]
    radj_12m[t] = ret_12m[t] / vol_ann[t]

Cross-sectional blend (per day, across all names — like ``mom_lowvol_combo``):

    z6[t,i]  = ( radj_6m[t,i]  - mean_t(radj_6m)  ) / std_t(radj_6m)
    z12[t,i] = ( radj_12m[t,i] - mean_t(radj_12m) ) / std_t(radj_12m)
    mom_score[t,i] = 0.5 * z6[t,i] + 0.5 * z12[t,i]

``mean_t`` / ``std_t`` are the NaN-aware cross-sectional moments over names
(``.mean(axis=1)`` / ``.std(axis=1)``, both skip-NaN; ``std`` is the ddof=1
sample std). A date whose cross-sectional ``std`` is exactly 0 (all names equal)
yields NaN Z-scores (ineligible) rather than a divide-by-zero. A name is scored
only on days where both legs are defined, so ``mom_score`` is NaN during warmup.

Selection: ``rank_score = mom_score`` so the descending ranker fills its
``--top`` slots with the highest risk-adjusted-momentum names. The entry gate
``mom_score > 0`` keeps only positive-momentum winners eligible. NSE's final
"normalized momentum score" transform ``1+z if z>=0 else 1/(1-z)`` is strictly
monotonic in ``z``, so applying it would leave the top-N *ranking* unchanged; we
therefore rank on the Z-score blend directly and the selection is identical.

Ranking-universe note (see ``mom_lowvol_combo``): the Z-scores are computed at
prepare time over the full prepared universe (``ctx.bars_by_tv``), not the
per-day post-filter eligible subset. This is intentional and documented there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_RET_6M_WINDOW = 126  # ~6 months of trading days
_RET_12M_WINDOW = 252  # ~12 months of trading days
_VOL_WINDOW = 252  # trailing window for the annualized-vol denominator
_TRADING_DAYS = 252  # annualization factor
_Z6_WEIGHT = 0.5
_Z12_WEIGHT = 0.5


def six_month_return(close: pd.Series) -> pd.Series:
    """Causal trailing 6-month (126-td) total return for one symbol."""
    close = close.astype(float)
    return close / close.shift(_RET_6M_WINDOW) - 1.0


def twelve_month_return(close: pd.Series) -> pd.Series:
    """Causal trailing 12-month (252-td) total return for one symbol."""
    close = close.astype(float)
    return close / close.shift(_RET_12M_WINDOW) - 1.0


def annualized_volatility(close: pd.Series) -> pd.Series:
    """Causal annualized volatility of trailing 252 daily log returns.

    Uses ``log(close).diff()`` (daily log returns), a rolling std with
    ``min_periods=252`` (pandas-default ``ddof=1`` sample std), annualized by
    ``sqrt(252)``. Backward-looking, so causal.
    """
    close_f = close.astype(float)
    log_close = pd.Series(np.log(close_f.to_numpy()), index=close_f.index)
    log_returns = log_close.diff()
    std = log_returns.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW).std()
    vol: pd.Series = std * np.sqrt(_TRADING_DAYS)
    return vol


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row (per-day) Z-score across names (columns).

    NaN-aware: ``mean``/``std`` over ``axis=1`` skip NaN cells. Rows whose
    cross-sectional std is exactly 0 map to NaN (guarded divide-by-zero).
    """
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _prepare_risk_adjusted_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    ret6_by_tv: dict[str, pd.Series] = {}
    ret12_by_tv: dict[str, pd.Series] = {}
    vol_by_tv: dict[str, pd.Series] = {}
    radj6_by_tv: dict[str, pd.Series] = {}
    radj12_by_tv: dict[str, pd.Series] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        close = bars["close"]
        ret6 = six_month_return(close)
        ret12 = twelve_month_return(close)
        vol = annualized_volatility(close)
        ret6_by_tv[tv] = ret6
        ret12_by_tv[tv] = ret12
        vol_by_tv[tv] = vol
        radj6_by_tv[tv] = ret6 / vol
        radj12_by_tv[tv] = ret12 / vol

    out: dict[str, pd.DataFrame] = {tv: bars for tv, bars in ctx.bars_by_tv.items()}
    if not radj6_by_tv:
        return out

    # Cross-sectional Z-scores across names (axis=1) per day, then equal-weight
    # blend. NaN legs stay NaN, so a name is scored only where both are defined.
    z6 = cross_sectional_zscore(pd.DataFrame(radj6_by_tv))
    z12 = cross_sectional_zscore(pd.DataFrame(radj12_by_tv))
    mom_score = _Z6_WEIGHT * z6 + _Z12_WEIGHT * z12

    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        frame["ret_6m"] = ret6_by_tv[tv]
        frame["ret_12m"] = ret12_by_tv[tv]
        frame["vol_ann"] = vol_by_tv[tv]
        frame["mom_score"] = mom_score[tv].reindex(frame.index)
        frame["rank_score"] = frame["mom_score"]
        out[tv] = frame
    return out


def _risk_adjusted_momentum_lookback() -> int:
    # log-return diff consumes one bar, then the 12-month leg / 252-return vol
    # window need 252 prior values: 252 + 1 = 253.
    return _RET_12M_WINDOW + 1


register_expression_strategy(
    "risk_adjusted_momentum",
    entry="mom_score > 0",
    exit=None,
    prepare_bars=_prepare_risk_adjusted_momentum,
    required_lookback=_risk_adjusted_momentum_lookback,
)
