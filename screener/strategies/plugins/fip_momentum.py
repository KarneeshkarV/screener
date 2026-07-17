"""Da-Gurun-Warachka (2014) "Frog in the Pan" momentum.

Paper: Da, Gurun & Warachka, "Frog in the Pan: Continuous Information and
Momentum", Review of Financial Studies 27(7), 2014. Momentum is strongest for
stocks whose formation-period gains arrived *continuously* — a long series of
small moves the market under-reacts to (the frog that doesn't jump out of slowly
heating water) — rather than in a few large, salient jumps that get priced in
quickly.

Two legs, both over the classic 12-1 formation window (skip the last month):

* **12-1 momentum** — reused from ``momentum_12_1``:

      mom_12_1[t] = close[t-21] / close[t-252] - 1

* **Information Discreteness (ID)** — over the *same* formation returns, i.e. the
  231 daily close-to-close returns on index ``t-251 .. t-21`` (rolling 231-day
  window ``shift(21)``-ed to exclude the skipped last month, exactly the span the
  12-1 ratio measures):

      ID[t] = sign(mom_12_1[t]) * (pct_negative_days - pct_positive_days)

  where ``pct_negative_days`` / ``pct_positive_days`` are the counts of down /
  up return-days divided by the *full window length* (231). Zero-return days
  count toward neither numerator but stay in the denominator, so the two
  percentages need not sum to one — this convention keeps ID insensitive to the
  count of flat days. Continuous information => many small moves of mixed sign =>
  ``pct_neg ~ pct_pos`` => ID near zero or negative for a winner; discrete
  information => the sign of the big jumps dominates => ID large and positive.
  **Lower (more negative) ID is better** (more continuous).

Composite selection (cross-sectional percentile blend at prepare time, exactly
like ``mom_lowvol_combo``):

    rank_score[t,i] = 0.5 * pct_rank(mom_12_1[t, :]) + 0.5 * pct_rank(-ID[t, :])

Both legs are percentile-ranked across names per day (scale-free, outlier-robust)
and a name is scored only on days where BOTH legs are defined, so ``rank_score``
is NaN (ineligible) during warmup. The entry gate additionally requires positive
momentum, so the portfolio is "continuous-information *winners*". Each daily
cross-section uses only bar-``t`` (itself causal) factor values, so the blend is
causal. See ``mom_lowvol_combo`` for the ranking-universe note that applies to
every prepare-time percentile blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.plugins.momentum_12_1 import (
    _LOOKBACK,
    _SKIP,
    momentum_12_1_score,
)
from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ID is measured over the formation window minus the skipped last month: the 231
# daily returns the 12-1 ratio actually spans.
_ID_WINDOW = _LOOKBACK - _SKIP  # 231
_MOM_WEIGHT = 0.5
_ID_WEIGHT = 0.5


def information_discreteness(close: pd.Series, mom: pd.Series) -> pd.Series:
    """Return the causal Information-Discreteness series for one symbol.

    Vectorized: rolling sums of down/up indicators over ``_ID_WINDOW`` returns,
    ``shift(_SKIP)``-ed to exclude the most recent month, divided by the window
    length. No per-row Python loop.
    """
    returns = close.astype(float).pct_change()
    neg = (returns < 0.0).astype(float)
    pos = (returns > 0.0).astype(float)
    # rolling(_ID_WINDOW) at t covers returns [t-230, t]; shift(_SKIP) moves the
    # window to [t-251, t-21], the 12-1 formation span (last month excluded).
    neg_count = neg.rolling(_ID_WINDOW, min_periods=_ID_WINDOW).sum().shift(_SKIP)
    pos_count = pos.rolling(_ID_WINDOW, min_periods=_ID_WINDOW).sum().shift(_SKIP)
    pct_neg = neg_count / _ID_WINDOW
    pct_pos = pos_count / _ID_WINDOW
    disc: pd.Series = np.sign(mom) * (pct_neg - pct_pos)
    return disc


def _prepare_fip(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    mom_by_tv: dict[str, pd.Series] = {}
    id_by_tv: dict[str, pd.Series] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        mom = momentum_12_1_score(bars["close"])
        mom_by_tv[tv] = mom
        id_by_tv[tv] = information_discreteness(bars["close"], mom)

    out: dict[str, pd.DataFrame] = {tv: bars for tv, bars in ctx.bars_by_tv.items()}
    if not mom_by_tv:
        return out

    # Cross-sectional percentile ranks across names (axis=1) per day. NaN factor
    # cells are ignored by ``rank`` and stay NaN, so a name is scored only on days
    # where both legs are defined. Lower ID is better, so rank ``-ID``.
    mom_df = pd.DataFrame(mom_by_tv)
    neg_id_df = -pd.DataFrame(id_by_tv)
    mom_pct = mom_df.rank(axis=1, pct=True)
    neg_id_pct = neg_id_df.rank(axis=1, pct=True)
    blended = _MOM_WEIGHT * mom_pct + _ID_WEIGHT * neg_id_pct

    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        frame["mom_12_1"] = mom_by_tv[tv]
        frame["id_disc"] = id_by_tv[tv]
        frame["rank_score"] = blended[tv].reindex(frame.index)
        out[tv] = frame
    return out


def _fip_lookback() -> int:
    # pct_change consumes one bar, then the 12-1 ratio / ID window need 252 prior
    # closes: 252 + 1 = 253.
    return _LOOKBACK + 1


register_expression_strategy(
    "fip_momentum",
    entry="mom_12_1 > 0",
    exit=None,
    prepare_bars=_prepare_fip,
    required_lookback=_fip_lookback,
)
