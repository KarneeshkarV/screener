"""12-month Indian momentum model, one strategy per ablation stage.

The model comes from a retail momentum video. Each stage adds one rule to the
stage before it:

- ``s0_52wh``: rank by closeness to the 52-week high.
- ``s1_roc``: rank by 12-month rate of change.
- ``s2_stage2``: + close above the 30, 50 and 100-day SMA.
- ``s3_near52``: + within 20% of the 52-week high.
- ``s4_volpen``: + rank by 12-month ROC divided by 12-month volatility.
- ``s5_circuit``: + at most 3 price-band-lock days in the last 126 sessions.
- ``s6_buffer50``: + hold while in the top 50 (run setting, see below).
- ``s7_composite``: + 52-week-high distance blended into the rank.
- ``s8_dma200``: + close above the 200-day SMA.

The ``m*`` stages rank on 12-1 momentum, which skips the last month, in place
of the 12-month ROC. Each stage is registered as ``india_mom_<stage>``.

``india_mom_<stage>_lc`` adds a lower-circuit exit. It sells at the next open
after a close locked at the lower band, and blocks a new buy of a name that had
a lower lock in the last 21 sessions.

The rank exit and the hold buffer are run settings, not part of the strategy.
Pass them on the command line, with the buffer from :data:`STAGES`::

    uv run screener backtest-rolling -m india --strategy india_mom_s4_volpen_lc \\
        --top 20 --rank-exit monthly --rank-universe-size 20 \\
        --universe india_turnover750_pit --cost-model india

``scripts/run_india_momentum_ablation.py`` runs the full grid with these
settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.factors.recipes import MOMENTUM_LOOKBACK, MOMENTUM_SKIP
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)
from screener.tradeable import tradeable_span

TOP = 20
LOOKBACK = MOMENTUM_LOOKBACK
#: Sessions 12-1 momentum skips, as in ``screener.factors.momentum_12_1``.
SKIP = MOMENTUM_SKIP
CIRCUIT_WINDOW = 126
CIRCUIT_MAX_DAYS = 3
NEAR_HIGH = 0.80
LOCK_MOVE = 0.049
#: Sessions a lower lock blocks a new buy, one monthly rank cycle.
LOWER_LOCK_COOLDOWN = 21
#: Name prefix and lower-circuit suffix of the registered strategies.
PREFIX = "india_mom_"
LOWER_CIRCUIT_SUFFIX = "_lc"


def band_locks(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sessions that look like an upper and a lower price-band lock.

    A lock closes at least 4.9% away from the prior close and on the day's
    high (upper band) or low (lower band). A 2% band lock is not caught, which
    errs toward keeping names.
    """
    close = bars["close"].astype(float)
    move = close.pct_change()
    tolerance = 0.001 * close
    upper = (move >= LOCK_MOVE) & ((bars["high"] - close).abs() <= tolerance)
    lower = (move <= -LOCK_MOVE) & ((close - bars["low"]).abs() <= tolerance)
    return upper, lower


def circuit_hits(bars: pd.DataFrame, window: int = CIRCUIT_WINDOW) -> pd.Series:
    """Trailing count of sessions that look like a price-band lock."""
    upper, lower = band_locks(bars)
    return (upper | lower).astype(float).rolling(window, min_periods=window).sum()


def rank_score(
    kind: str, momentum: pd.DataFrame, vol: pd.DataFrame, near_high: pd.DataFrame
) -> pd.DataFrame:
    """Cross-sectional score per date (higher ranks first)."""
    if kind == "near_hi":
        return near_high
    if kind == "momentum":
        return momentum
    risk_adjusted = momentum / vol.replace(0, np.nan)
    if kind == "momentum_vol":
        return risk_adjusted
    if kind == "composite":
        return 0.5 * risk_adjusted.rank(axis=1, pct=True) + 0.5 * near_high.rank(
            axis=1, pct=True
        )
    raise ValueError(f"unknown rank kind: {kind}")


def prepare_bars(
    kind: str, momentum: str = "roc_12m"
) -> Callable[[PrepareCtx], dict[str, pd.DataFrame]]:
    """``prepare_bars`` hook: add the factor columns and ``rank_score``.

    ``momentum`` is the column the rank reads: ``roc_12m`` (the last 12
    months) or ``mom_12_1`` (12 months back to 1 month back).
    """

    def prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        factors: dict[str, dict[str, pd.Series]] = {
            "momentum": {},
            "vol": {},
            "near_hi": {},
        }
        out: dict[str, pd.DataFrame] = {}
        for ticker, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[ticker] = bars
                continue
            frame = bars.copy()
            close = frame["close"].astype(float)
            live = tradeable_span(close, frame.get("volume"), window=LOOKBACK)
            high = (
                frame["high"]
                .astype(float)
                .rolling(LOOKBACK, min_periods=LOOKBACK)
                .max()
            )
            frame["roc_12m"] = (close / close.shift(LOOKBACK) - 1).where(live)
            frame["mom_12_1"] = (close.shift(SKIP) / close.shift(LOOKBACK) - 1).where(
                live
            )
            frame["vol_12m"] = (
                close.pct_change()
                .rolling(LOOKBACK, min_periods=LOOKBACK)
                .std()
                .where(live)
            )
            frame["near_hi"] = (close / high).where(live)
            frame["circuit_hits"] = circuit_hits(frame)
            lower = band_locks(frame)[1].astype(float)
            frame["lower_lock"] = lower
            frame["lower_lock_recent"] = lower.rolling(
                LOWER_LOCK_COOLDOWN, min_periods=1
            ).sum()
            for name, column in (
                ("momentum", momentum),
                ("vol", "vol_12m"),
                ("near_hi", "near_hi"),
            ):
                factors[name][ticker] = frame[column]
            out[ticker] = frame
        if not factors["momentum"]:
            return out
        momentum_frame, vol, near_high = (
            ctx.mask_rank_reference(pd.DataFrame(factors[name]))
            for name in ("momentum", "vol", "near_hi")
        )
        score = rank_score(kind, momentum_frame, vol, near_high)
        for ticker, frame in out.items():
            if frame is not None and not frame.empty and ticker in score.columns:
                frame["rank_score"] = score[ticker].reindex(frame.index)
        return out

    return prepare


@dataclass(frozen=True)
class Stage:
    rank: str
    entry: str
    #: ``--rank-universe-size`` the stage runs with: a holding is kept while
    #: it ranks inside this many names.
    hold_buffer: int
    description: str
    momentum: str = "roc_12m"


BASE = "roc_12m > -10 and vol_12m > 0"
TREND = f"{BASE} and close > sma(close, 30) and close > sma(close, 50) and close > sma(close, 100)"
NEAR = f"{TREND} and near_hi >= {NEAR_HIGH}"
NO_LOCKS = f"{NEAR} and circuit_hits <= {CIRCUIT_MAX_DAYS}"
STAGES: dict[str, Stage] = {
    "s0_52wh": Stage("near_hi", BASE, TOP, "Rank by closeness to the 52-week high"),
    "s1_roc": Stage("momentum", BASE, TOP, "Rank by 12-month rate of change"),
    "s1b_roc_buf50": Stage("momentum", BASE, 50, "s1 + hold while in the top 50"),
    "s2_stage2": Stage("momentum", TREND, TOP, "+ close above the 30/50/100-day SMA"),
    "s3_near52": Stage("momentum", NEAR, TOP, "+ within 20% of the 52-week high"),
    "s4_volpen": Stage("momentum_vol", NEAR, TOP, "+ rank by ROC / volatility"),
    "s5_circuit": Stage(
        "momentum_vol", NO_LOCKS, TOP, "+ at most 3 band-lock days in 126"
    ),
    "s6_buffer50": Stage("momentum_vol", NO_LOCKS, 50, "+ hold while in the top 50"),
    "s7_composite": Stage(
        "composite", NO_LOCKS, 50, "+ 52-week-high distance in the rank"
    ),
    "s8_dma200": Stage(
        "composite",
        f"{NO_LOCKS} and close > sma(close, 200)",
        50,
        "+ close above the 200-day SMA",
    ),
    # 12-1 variants: the same stages ranked on 12-1 momentum, which skips the
    # last month's short-term reversal. ``mom_12_1 > 0`` is the floor the
    # repo's momentum_12_1 recipe uses.
    "m1_mom121": Stage(
        "momentum",
        f"{BASE} and mom_12_1 > 0",
        TOP,
        "s1 ranked by 12-1 momentum",
        "mom_12_1",
    ),
    "m4_mom121_volpen": Stage(
        "momentum_vol",
        f"{NEAR} and mom_12_1 > 0",
        TOP,
        "s4 ranked by 12-1 momentum / volatility",
        "mom_12_1",
    ),
    "m6_mom121_buffer50": Stage(
        "momentum_vol",
        f"{NO_LOCKS} and mom_12_1 > 0",
        50,
        "s6 ranked by 12-1 momentum / volatility",
        "mom_12_1",
    ),
    "m7_mom121_composite": Stage(
        "composite",
        f"{NO_LOCKS} and mom_12_1 > 0",
        50,
        "s7 with 12-1 momentum in the composite",
        "mom_12_1",
    ),
}


def strategy_name(stage: str, *, lower_circuit_exit: bool = False) -> str:
    """Registered name of a stage, ``india_mom_s4_volpen`` or ``..._lc``."""
    suffix = LOWER_CIRCUIT_SUFFIX if lower_circuit_exit else ""
    return f"{PREFIX}{stage}{suffix}"


def _lookback() -> int:
    # 252-bar windows plus the pct_change bar, with a few sessions of slack.
    return LOOKBACK + 8


for _stage, _spec in STAGES.items():
    _prepare = prepare_bars(_spec.rank, _spec.momentum)
    register_expression_strategy(
        strategy_name(_stage),
        entry=_spec.entry,
        exit=None,
        prepare_bars=_prepare,
        required_lookback=_lookback,
        profile=DEFAULT_STRATEGY_PROFILE,
    )
    register_expression_strategy(
        strategy_name(_stage, lower_circuit_exit=True),
        entry=f"{_spec.entry} and lower_lock_recent == 0",
        exit="lower_lock > 0",
        prepare_bars=_prepare,
        required_lookback=_lookback,
        profile=DEFAULT_STRATEGY_PROFILE,
    )
