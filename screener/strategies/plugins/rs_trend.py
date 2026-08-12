"""Relative strength vs benchmark with a long-term trend gate (India/Nifty style).

Relative-strength stock selection — comparing each name's price to the market
benchmark (SPY for US, ^NSEI for India) — is the most widely used momentum
screen for Indian equities and is standard practice in institutional research
(e.g. the "stock vs Nifty 500" RS screens; Minervini's RS Line). The RS line is

    rs[t] = close[t] / benchmark_close[t]

A rising RS line means the stock is outperforming the market; the signal here
ranks names by the 6-month change of the RS line (outperformance persistence)
and gates entries on positive RS momentum plus price above the 200-day SMA.

``rs_trend`` — rank by rs_126 (6-month RS change), enter when RS is improving
and the price is above its 200-day SMA. Time-based quarterly rotation (--hold)
approximates periodic rebalancing.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_RS_WINDOW = 126  # 6 months of trading days
_TREND_SMA = 200


def _prepare_rs(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(f"benchmark data unavailable for rs_trend: {ctx.benchmark}")
        return ctx.bars_by_tv

    benchmark_close = benchmark_bars["close"]

    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        aligned = benchmark_close.reindex(frame.index).ffill()
        rs = frame["close"].astype(float) / aligned
        frame["rs"] = rs
        frame["rs_126"] = rs / rs.shift(_RS_WINDOW) - 1.0
        # Higher 6-month RS change ranks first (outperformance persistence).
        frame["rank_score"] = frame["rs_126"]
        out[tv] = frame
    return out


def _lookback() -> int:
    return max(_RS_WINDOW, _TREND_SMA)


register_expression_strategy(
    "rs_trend",
    entry=f"rs_126 > 0 and close > sma(close, {_TREND_SMA})",
    exit=None,
    prepare_bars=_prepare_rs,
    required_lookback=_lookback,
)
