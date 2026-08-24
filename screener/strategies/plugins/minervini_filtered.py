"""Filtered Mark Minervini variants for experimentation.

Each variant reuses the Minervini prepare hook (so rs_rank / sma200_up_1m
exist) and tightens the entry with extra filters. The prepare hook also attaches
the entry-quality features mined from a 5-year trade study (ext_above_sma50,
sma50_150_spread, sma150_200_spread, mom_63d, mom_126d, vol_20d_ann), computed
point-in-time (each bar uses only trailing data) so they can be referenced by
name in entry expressions. Fundamental columns (pe_ttm, revenue_growth_yoy,
eps_growth_yoy, ...) are auto-merged by the rolling backtester when referenced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.minervini import (
    prepare_backtest_frames,
    required_history_bars,
)
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)


def _add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach trend-quality features. Definitions match the trade-study
    enrichment exactly so shipped thresholds reproduce the mined win rates."""
    close = frame["close"].astype(float)
    s50 = close.rolling(50).mean()
    s150 = close.rolling(150).mean()
    s200 = close.rolling(200).mean()
    frame["ext_above_sma50"] = close / s50 - 1.0
    frame["sma50_150_spread"] = s50 / s150 - 1.0
    frame["sma150_200_spread"] = s150 / s200 - 1.0
    frame["mom_63d"] = close / close.shift(63) - 1.0
    frame["mom_126d"] = close / close.shift(126) - 1.0
    frame["vol_20d_ann"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    return frame


def _prep(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    prepared = prepare_backtest_frames(ctx.bars_by_tv)
    return {sym: _add_quality_features(frame) for sym, frame in prepared.items()}


# Full Minervini trend template, parameterised so variants can tighten it.
def _template(rs_min: float, near_high: float, extra: str = "") -> str:
    base = (
        "close > sma(close, 150) and "
        "close > sma(close, 200) and "
        "sma(close, 150) > sma(close, 200) and "
        "sma200_up_1m > 0 and "
        "sma(close, 50) > sma(close, 150) and "
        "sma(close, 50) > sma(close, 200) and "
        "close > sma(close, 50) and "
        "close >= lowest(low, 252) * 1.3 and "
        f"close >= highest(high, 252) * {near_high} and "
        f"rs_rank >= {rs_min}"
    )
    return base + (f" and {extra}" if extra else "")


EXIT = "crossunder(close, sma(close, 50))"


def _register(name: str, entry: str) -> None:
    register_expression_strategy(
        name,
        entry=entry,
        exit=EXIT,
        prepare_bars=_prep,
        required_lookback=required_history_bars,
        profile=DEFAULT_STRATEGY_PROFILE,
    )


# --- Legacy experimentation variants ---------------------------------------
_register("minervini_leaders", _template(85, 0.85))
_register("minervini_growth_us", _template(70, 0.75, "eps_growth_yoy > 0"))
_register("minervini_growth_in", _template(70, 0.75, "revenue_up_3q > 0"))
_register("minervini_pro_us", _template(85, 0.85, "eps_growth_yoy > 0"))
_register("minervini_pro_in", _template(85, 0.85, "revenue_up_3q > 0"))


# --- Entry-quality rules from the 5-year trade study -----------------------
# Base = full Minervini template (rs_rank>=70, near 52w high). Each rule adds
# the verified entry-quality band that lifted the mined win rate. Names encode
# the market the rule was mined from; all are market-agnostic expressions and
# can be backtested on either market.

# US rules: "extended above SMA50 but not overheated" + a non-cheap valuation.
_register(
    "mq_us1",
    _template(
        70,
        0.75,
        "ext_above_sma50 >= 0.13 and sma50_150_spread <= 0.115 and pe_ttm >= 25",
    ),
)
_register(
    "mq_us2",
    _template(70, 0.75, "ext_above_sma50 >= 0.13 and mom_63d <= 0.25 and pe_ttm >= 20"),
)
_register("mq_us3", _template(70, 0.75, "ext_above_sma50 >= 0.13 and mom_63d <= 0.25"))

# India rules: strong extension + growth fundamentals, moderate RS, low vol.
_register(
    "mq_in1",
    _template(
        70,
        0.75,
        "ext_above_sma50 >= 0.20 and sma50_150_spread <= 0.075 and eps_growth_yoy >= 0",
    ),
)
_register("mq_in2", _template(70, 0.75, "rs_rank <= 80 and revenue_growth_yoy >= 50"))
_register(
    "mq_in3",
    _template(
        70,
        0.75,
        "ext_above_sma50 >= 0.06 and eps_growth_yoy >= 139 and mom_126d <= 0.42",
    ),
)
