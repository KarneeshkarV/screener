"""Stage2Stocks (Stan Weinstein) daily 4-stage strategies.

Implements causal approximations of the four nightly end-of-day filters
featured on ``https://www.stage2stocks.com/app`` and ``/screener``:

- ``stage2_early_entry``: Stage 2 stocks still early in the move (weeks 1–6)
  and not stretched far above the 50-day SMA (0–18% above), ranked by
  point-in-time 12-month RS rank (0–100), exiting when the stock leaves Stage 2.
- ``stage2_early_entry_breadth``: ``stage2_early_entry`` gated by Stage2Stocks's
  top-down market breadth check (new entries only when at least 25% of the
  point-in-time eligible universe is in Stage 2).
- ``stage2_leaders``: Stage 2 leaders with 12-month RS rank >= 80 (any
  maturity), ranked by RS rank, exiting when the stock leaves Stage 2 (pair
  with ``--trailing-stop 0.20`` to mirror the 20% leader trail).
- ``stage2_setup_score``: Stage 2 stocks ranked by a causal 4-factor setup
  score proxy (12-month RS rank, proximity to the 252-bar closing high,
  short-term acceleration ``SMA20 / SMA50 - 1``, and a late-trend maturity
  penalty), exiting when the stock leaves Stage 2 (pair with
  ``--rank-exit monthly --rank-universe-size 50`` for buffered rotation).

Stage classification (daily EOD) follows Stage2Stocks's published rule:
Stage 2 when ``SMA50 > SMA200`` and ``SMA50`` is rising; Stage 3 when
``SMA50 > SMA200`` and ``SMA50`` flattens or turns down; Stage 4 when
``SMA50 <= SMA200`` and falling; Stage 1 when ``SMA50 <= SMA200`` and turning
up. A small symmetric slope deadband keeps nearly flat trends in their prior
stage rather than flipping daily. Exact proprietary Stage2Stocks weights,
slope cutoffs, and industry-strength terms are unpublished and approximated
causally here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.relative_strength import RS_RANK_WINDOW, relative_strength_rank
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)
from screener.tradeable import tradeable_span

SMA_SHORT = 20
SMA_FAST = 50
SMA_SLOW = 200
HIGH_WINDOW = 252
SLOPE_LOOKBACK = 10
SLOPE_DEADBAND = 0.0025

EARLY_MAX_WEEKS = 6
EARLY_MAX_EXT_50 = 0.18
BREADTH_MIN_STAGE2 = 0.25
LEADERS_MIN_RS = 80.0

EARLY_ENTRY_EXPR = (
    f"s2_stage == 2 and s2_weeks >= 1 and s2_weeks <= {EARLY_MAX_WEEKS} "
    f"and s2_ext50 > 0 and s2_ext50 <= {EARLY_MAX_EXT_50}"
)
EARLY_ENTRY_BREADTH_EXPR = f"{EARLY_ENTRY_EXPR} and breadth_s2 >= {BREADTH_MIN_STAGE2}"
LEADERS_ENTRY_EXPR = f"s2_stage == 2 and rs_rank >= {int(LEADERS_MIN_RS)}"
SETUP_SCORE_ENTRY_EXPR = "s2_stage == 2"
STAGE2_EXIT_EXPR = "s2_stage != 2"


def compute_stage_series(
    close: pd.Series,
    *,
    fast_window: int = SMA_FAST,
    slow_window: int = SMA_SLOW,
    slope_lookback: int = SLOPE_LOOKBACK,
    slope_deadband: float = SLOPE_DEADBAND,
) -> tuple[pd.Series, pd.Series]:
    """Classify each bar into Stan Weinstein's 4 stages with flat-slope hysteresis.

    Returns ``(stage, weeks_in_stage2)`` aligned to ``close.index``. Both series
    are strictly causal: bar ``i`` depends only on closes ``<= i``.
    """
    prices = close.astype(float)
    sma_fast = prices.rolling(fast_window, min_periods=fast_window).mean()
    sma_slow = prices.rolling(slow_window, min_periods=slow_window).mean()
    slope = sma_fast / sma_fast.shift(slope_lookback) - 1.0

    fast_arr = sma_fast.to_numpy(dtype=np.float64)
    slow_arr = sma_slow.to_numpy(dtype=np.float64)
    slope_arr = slope.to_numpy(dtype=np.float64)
    n = len(prices)

    stage_arr = np.full(n, np.nan, dtype=np.float64)
    weeks_s2_arr = np.zeros(n, dtype=np.float64)

    prev_stage = 0
    s2_days = 0
    for i in range(n):
        fast_val = fast_arr[i]
        slow_val = slow_arr[i]
        slope_val = slope_arr[i]
        if not (
            np.isfinite(fast_val) and np.isfinite(slow_val) and np.isfinite(slope_val)
        ):
            prev_stage = 0
            s2_days = 0
            continue
        if fast_val > slow_val:
            if slope_val > slope_deadband:
                stage = 2
            elif slope_val < -slope_deadband:
                stage = 3
            else:
                stage = (
                    prev_stage
                    if prev_stage in (2, 3)
                    else (2 if slope_val >= 0.0 else 3)
                )
        else:
            if slope_val < -slope_deadband:
                stage = 4
            elif slope_val > slope_deadband:
                stage = 1
            else:
                stage = (
                    prev_stage
                    if prev_stage in (1, 4)
                    else (1 if slope_val >= 0.0 else 4)
                )
        stage_arr[i] = float(stage)
        prev_stage = stage
        if stage == 2:
            s2_days += 1
            weeks_s2_arr[i] = float(1 + (s2_days - 1) // 5)
        else:
            s2_days = 0
            weeks_s2_arr[i] = 0.0

    return (
        pd.Series(stage_arr, index=close.index, dtype=float),
        pd.Series(weeks_s2_arr, index=close.index, dtype=float),
    )


def _freshness_score(weeks_s2: pd.Series) -> pd.Series:
    """1.0 during weeks 1–12 of Stage 2, tapering linearly to 0.2 by week 52."""
    weeks = weeks_s2.to_numpy(dtype=np.float64)
    raw = np.where(
        weeks <= 0.0,
        np.nan,
        np.clip(1.0 - np.maximum(weeks - 12.0, 0.0) / 50.0, 0.2, 1.0),
    )
    return pd.Series(raw, index=weeks_s2.index, dtype=float)


def prepare_stage2_bars(
    rank_mode: str = "rs_rank",
) -> Callable[[PrepareCtx], dict[str, pd.DataFrame]]:
    """Build a ``prepare_bars`` hook that attaches causal Stage2Stocks features.

    Cross-sectional columns (``rs_rank``, ``breadth_s2``, and ``rank_score``)
    apply ``ctx.mask_rank_reference`` before computing percentiles or market
    breadth so non-members cannot leak into point-in-time rankings.
    """
    if rank_mode not in {"rs_rank", "setup_score"}:
        raise ValueError(f"unknown stage2stocks rank_mode: {rank_mode!r}")

    def prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        closes: dict[str, pd.Series] = {}
        volumes: dict[str, pd.Series] = {}
        stages: dict[str, pd.Series] = {}
        near_hi_map: dict[str, pd.Series] = {}
        accel_map: dict[str, pd.Series] = {}
        fresh_map: dict[str, pd.Series] = {}
        out: dict[str, pd.DataFrame] = {}

        for ticker, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[ticker] = bars
                continue
            frame = bars.copy()
            close = frame["close"].astype(float)
            volume = frame.get("volume")
            sma_short = close.rolling(SMA_SHORT, min_periods=SMA_SHORT).mean()
            sma_fast = close.rolling(SMA_FAST, min_periods=SMA_FAST).mean()
            stage, weeks_s2 = compute_stage_series(close)
            tradeable_50 = tradeable_span(close, volume, window=SMA_FAST)
            tradeable_252 = tradeable_span(close, volume, window=HIGH_WINDOW)
            ext_50 = (close / sma_fast - 1.0).where(tradeable_50)
            high_252 = close.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW).max()
            near_52wh = (close / high_252).where(tradeable_252)
            accel = (sma_short / sma_fast - 1.0).where(tradeable_50)

            frame["s2_stage"] = stage
            frame["s2_weeks"] = weeks_s2
            frame["s2_ext50"] = ext_50
            frame["s2_near_hi"] = near_52wh
            frame["s2_accel"] = accel

            closes[ticker] = close
            if volume is not None:
                volumes[ticker] = volume
            stages[ticker] = stage
            near_hi_map[ticker] = near_52wh
            accel_map[ticker] = accel
            fresh_map[ticker] = _freshness_score(weeks_s2)
            out[ticker] = frame

        if not closes:
            return out

        rs_df = relative_strength_rank(
            closes,
            volumes_by_symbol=volumes or None,
            window=RS_RANK_WINDOW,
            mask_rank_reference=ctx.mask_rank_reference,
        )

        stage_df = ctx.mask_rank_reference(pd.DataFrame(stages))
        valid_count = stage_df.notna().sum(axis=1)
        s2_count = (stage_df == 2.0).sum(axis=1)
        breadth_s2 = (s2_count / valid_count.replace(0, np.nan)).fillna(0.0)

        if rank_mode == "setup_score":
            s2_mask = stage_df == 2.0
            near_df = ctx.mask_rank_reference(pd.DataFrame(near_hi_map)).where(s2_mask)
            acc_df = ctx.mask_rank_reference(pd.DataFrame(accel_map)).where(s2_mask)
            fresh_df = ctx.mask_rank_reference(pd.DataFrame(fresh_map)).where(s2_mask)
            rs_s2 = rs_df.where(s2_mask) / 100.0
            score_df = 10.0 * (
                0.35 * rs_s2
                + 0.25 * near_df.rank(axis=1, pct=True)
                + 0.20 * acc_df.rank(axis=1, pct=True)
                + 0.20 * fresh_df
            )
        else:
            score_df = rs_df

        for ticker, frame in out.items():
            if frame is None or frame.empty:
                continue
            frame["rs_rank"] = (
                rs_df[ticker].reindex(frame.index)
                if ticker in rs_df.columns
                else np.nan
            )
            frame["breadth_s2"] = breadth_s2.reindex(frame.index)
            frame["rank_score"] = (
                score_df[ticker].reindex(frame.index)
                if ticker in score_df.columns
                else np.nan
            )
        return out

    return prepare


@dataclass(frozen=True)
class Stage2Variant:
    name: str
    label: str
    entry: str
    exit: str
    rank_mode: str
    trailing_stop: float | None = None
    rank_exit: str | None = None
    rank_universe_size: int = 50


VARIANTS: dict[str, Stage2Variant] = {
    "stage2_early_entry": Stage2Variant(
        name="stage2_early_entry",
        label="Early Entry (Stage Exit)",
        entry=EARLY_ENTRY_EXPR,
        exit=STAGE2_EXIT_EXPR,
        rank_mode="rs_rank",
    ),
    "stage2_early_entry_breadth": Stage2Variant(
        name="stage2_early_entry_breadth",
        label="Early Entry + Breadth>=25%",
        entry=EARLY_ENTRY_BREADTH_EXPR,
        exit=STAGE2_EXIT_EXPR,
        rank_mode="rs_rank",
    ),
    "stage2_leaders": Stage2Variant(
        name="stage2_leaders",
        label="Leaders (RS>=80, Stage+20%TS)",
        entry=LEADERS_ENTRY_EXPR,
        exit=STAGE2_EXIT_EXPR,
        rank_mode="rs_rank",
        trailing_stop=0.20,
    ),
    "stage2_setup_score": Stage2Variant(
        name="stage2_setup_score",
        label="Setup Score (Monthly Buf50)",
        entry=SETUP_SCORE_ENTRY_EXPR,
        exit=STAGE2_EXIT_EXPR,
        rank_mode="setup_score",
        rank_exit="monthly",
        rank_universe_size=50,
    ),
}


def required_history_bars() -> int:
    """252-bar RS/52W high window plus slack for slope and rolling alignment."""
    return HIGH_WINDOW + 8


for _variant in VARIANTS.values():
    register_expression_strategy(
        _variant.name,
        entry=_variant.entry,
        exit=_variant.exit,
        prepare_bars=prepare_stage2_bars(_variant.rank_mode),
        required_lookback=required_history_bars,
        profile=DEFAULT_STRATEGY_PROFILE,
    )
