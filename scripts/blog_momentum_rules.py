"""Causal daily momentum rules for the blog research backtests.

These are research adapters, not exact replicas of long/short portfolios.
Source rules and all adaptations are recorded in findings/blog_momentum.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.blog_monthly_rules import MONTHLY_RULES, prepare_blog_monthly

from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    PrepareCtx,
    register_expression_strategy,
)


def blog_momentum_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-only signals, with 252 sessions representing 12 months."""
    frame = bars.copy()
    close = frame["close"].where(frame["close"] > 0)
    frame["chan_return_12m"] = close / close.shift(252) - 1
    frame["chan_weighted_13612"] = (
        12 * (close / close.shift(21) - 1)
        + 4 * (close / close.shift(63) - 1)
        + 2 * (close / close.shift(126) - 1)
        + (close / close.shift(252) - 1)
    ) / 19
    log_close = np.log(close)
    frame["chan_log_ma_7_10"] = (
        log_close.rolling(7).mean() - log_close.rolling(10).mean()
    )
    frame["chan_log_ma_50_200"] = (
        log_close.rolling(50).mean() - log_close.rolling(200).mean()
    )
    return frame


def prepare_blog_trend_filters(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Prepare published trend filters as long/cash research adaptations."""
    out = {}
    for ticker, bars in ctx.bars_by_tv.items():
        frame = bars.copy()
        close = frame["close"]
        sma200 = close.rolling(200).mean()
        frame["unger_donchian5"] = close - frame["high"].rolling(5).max().shift(1)
        frame["unger_donchian5_exit"] = close - frame["low"].rolling(5).min().shift(1)
        if ctx.market == "us":
            frame["unger_donchian5_exit"] = frame["unger_donchian5_exit"].shift(1)
        frame["unger_close_sma200"] = close - sma200
        frame["unger_sma50_200"] = close.rolling(50).mean() - sma200
        frame["unger_sma200_slope50"] = sma200 - sma200.shift(50)
        frame["robotwealth_close_sma100"] = close - close.rolling(100).mean()
        above = close.gt(sma200).astype(float).where(sma200.notna())
        below = close.lt(sma200).astype(float).where(sma200.notna())
        for count in (3, 5):
            frame[f"alvarez_ma200_confirm{count}"] = (
                above.rolling(count).sum().eq(count).astype(float)
            )
            exit_signal = below.rolling(count).sum().eq(count).astype(float)
            frame[f"alvarez_ma200_confirm{count}_exit"] = (
                exit_signal.shift(1) if ctx.market == "us" else exit_signal
            )
        for column in (
            "unger_close_sma200",
            "unger_sma50_200",
            "unger_sma200_slope50",
            "robotwealth_close_sma100",
        ):
            frame[f"{column}_exit"] = (
                frame[column].shift(1) if ctx.market == "us" else frame[column]
            )
        out[ticker] = frame
    return out


def prepare_chan_return_rank(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Rank candidates by full 12-month return, without a skipped month."""
    return _prepare_chan_rank(ctx, "chan_return_12m")


def prepare_chan_weighted_rank(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Rank candidates by the 13612W signal, not the full canary allocation rule."""
    return _prepare_chan_rank(ctx, "chan_weighted_13612")


def _prepare_chan_rank(ctx: PrepareCtx, column: str) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker, bars in ctx.bars_by_tv.items():
        frame = blog_momentum_columns(bars)
        frame["rank_score"] = frame[column]
        frame[f"{column}_exit"] = (
            frame[column].shift(1) if ctx.market == "us" else frame[column]
        )
        out[ticker] = frame
    return out


def prepare_chan_log_ma(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Prepare log-price crossover signals for next-session execution."""
    out = {}
    for ticker, bars in ctx.bars_by_tv.items():
        frame = blog_momentum_columns(bars)
        for column in ("chan_log_ma_7_10", "chan_log_ma_50_200"):
            frame[f"{column}_exit"] = (
                frame[column].shift(1) if ctx.market == "us" else frame[column]
            )
        out[ticker] = frame
    return out


def chan_daily_lookback() -> int:
    """Return the maximum trailing price window required by these adapters."""
    return 253


def register_blog_momentum_rules() -> None:
    """Register research-only strategies explicitly without changing CLI discovery."""
    for monthly_name in MONTHLY_RULES:

        def prepare_monthly(
            ctx: PrepareCtx, name: str = monthly_name
        ) -> dict[str, pd.DataFrame]:
            frames = prepare_blog_monthly(ctx)
            for frame in frames.values():
                if f"{name}_score" in frame:
                    frame["rank_score"] = frame[f"{name}_score"]
            return frames

        register_expression_strategy(
            monthly_name,
            entry=f"{monthly_name}_entry > 0",
            exit="monthly_rotate_exit > 0"
            if "rotation" in monthly_name or "factor" in monthly_name
            else f"{monthly_name}_exit > 0",
            prepare_bars=prepare_monthly,
            required_lookback=lambda: 350,
            profile=DEFAULT_STRATEGY_PROFILE,
        )
    register_expression_strategy(
        "unger_donchian5",
        entry="unger_donchian5 > 0",
        exit="unger_donchian5_exit < 0",
        prepare_bars=prepare_blog_trend_filters,
        required_lookback=chan_daily_lookback,
        profile=DEFAULT_STRATEGY_PROFILE,
    )
    for count in (3, 5):
        name = f"alvarez_ma200_confirm{count}"
        register_expression_strategy(
            name,
            entry=f"{name} > 0",
            exit=f"{name}_exit > 0",
            prepare_bars=prepare_blog_trend_filters,
            required_lookback=chan_daily_lookback,
            profile=DEFAULT_STRATEGY_PROFILE,
        )
    for name in (
        "unger_close_sma200",
        "unger_sma50_200",
        "unger_sma200_slope50",
        "robotwealth_close_sma100",
    ):
        register_expression_strategy(
            name,
            entry=f"crossover({name}, 0)"
            if name == "robotwealth_close_sma100"
            else f"{name} > 0",
            exit=f"crossunder({name}_exit, 0)"
            if name == "robotwealth_close_sma100"
            else f"{name}_exit < 0",
            prepare_bars=prepare_blog_trend_filters,
            required_lookback=chan_daily_lookback,
            profile=DEFAULT_STRATEGY_PROFILE,
        )
    for name, column, prepare in (
        ("chan_return_12m", "chan_return_12m", prepare_chan_return_rank),
        ("chan_weighted_13612", "chan_weighted_13612", prepare_chan_weighted_rank),
    ):
        register_expression_strategy(
            name,
            entry=f"{column} > 0",
            exit=f"{column}_exit <= 0",
            prepare_bars=prepare,
            required_lookback=chan_daily_lookback,
            profile=DEFAULT_STRATEGY_PROFILE,
        )
    for name in ("chan_log_ma_7_10", "chan_log_ma_50_200"):
        register_expression_strategy(
            name,
            entry=f"crossover({name}, 0)",
            exit=f"crossunder({name}_exit, 0)",
            prepare_bars=prepare_chan_log_ma,
            required_lookback=chan_daily_lookback,
            profile=DEFAULT_STRATEGY_PROFILE,
        )
