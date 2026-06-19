"""Combined momentum + low-volatility multi-factor strategy.

Blends two well-documented anomalies into a single cross-sectional score:

* 12-1 momentum  — Jegadeesh & Titman (1993); see ``momentum_12_1``.
* low volatility  — Ang, Hodrick, Xing & Zhang (2006); see ``low_volatility``.

The two factors are complementary: momentum is a return-chasing signal that
suffers sharp crashes, while low-volatility is a defensive signal. Combining them
is the spirit of the "low-vol momentum" / defensive-factor literature
(e.g. Blitz & van Vliet, 2007), targeting momentum's upside with a volatility
brake.

Combo logic & weighting (computed cross-sectionally, per day, over the universe):

    mom_pct[t,i]    = percentile_rank_across_names( mom_12_1[t, :] )      (high = good)
    invvol_pct[t,i] = percentile_rank_across_names( -vol_252[t, :] )      (high = good)
    rank_score[t,i] = 0.5 * mom_pct[t,i] + 0.5 * invvol_pct[t,i]

Equal 50/50 weighting on percentile ranks (not raw values) keeps the blend
scale-free and robust to either factor's outliers. A name is only scored on days
where BOTH factors are defined, so ``rank_score`` is NaN (ineligible) until 252
days of history exist. The entry gate additionally requires positive momentum, so
the portfolio is "low-volatility *winners*". Each cross-sectional rank at bar ``t``
uses only bar-``t`` factor values (themselves causal), so the blend is causal.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.plugins.low_volatility import realized_volatility
from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.spec import PrepareCtx, strategy

_MOM_WEIGHT = 0.5
_VOL_WEIGHT = 0.5


def _prepare_combo(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    mom_by_tv: dict[str, pd.Series] = {}
    vol_by_tv: dict[str, pd.Series] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        mom_by_tv[tv] = momentum_12_1_score(bars["close"])
        vol_by_tv[tv] = realized_volatility(bars["close"])

    out: dict[str, pd.DataFrame] = {tv: bars for tv, bars in ctx.bars_by_tv.items()}
    if not mom_by_tv:
        return out

    # Cross-sectional percentile ranks: rank across names (axis=1) per day. NaN
    # factor cells are ignored by ``rank`` and stay NaN, so a name is only scored
    # on days where both legs are defined.
    mom_df = pd.DataFrame(mom_by_tv)
    invvol_df = -pd.DataFrame(vol_by_tv)
    mom_pct = mom_df.rank(axis=1, pct=True)
    invvol_pct = invvol_df.rank(axis=1, pct=True)
    blended = _MOM_WEIGHT * mom_pct + _VOL_WEIGHT * invvol_pct

    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        frame["mom_12_1"] = mom_by_tv[tv]
        frame["vol_252"] = vol_by_tv[tv]
        frame["rank_score"] = blended[tv].reindex(frame.index)
        out[tv] = frame
    return out


def _combo_lookback() -> int:
    return 253  # max(momentum 252, low-vol pct_change+252)


@strategy(
    "mom_lowvol_combo",
    entry="mom_12_1 > 0 and vol_252 > 0",
    exit=None,
    prepare_bars=_prepare_combo,
    required_lookback=_combo_lookback,
)
def _mom_lowvol_combo() -> None:
    """Expression-only strategy; ranking is driven by the blended rank_score."""
