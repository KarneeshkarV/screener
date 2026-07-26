"""Relative-strength momentum regime: RS trending up while price is above its 200-EMA."""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy


def _prepare_rs_momentum_regime(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    benchmark_bars = ctx.price_panel.get(ctx.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for rs_momentum_regime: {ctx.benchmark}"
        )
        return ctx.bars_by_tv

    # Bars are indexed by date (no "date" column), so align on the index.
    benchmark_close = benchmark_bars["close"]

    prepared: dict[str, pd.DataFrame] = {}
    for symbol, bars in ctx.bars_by_tv.items():
        merged = bars.copy()
        aligned = benchmark_close.reindex(merged.index).ffill()
        merged["rs"] = merged["close"] / aligned
        prepared[symbol] = merged
    return prepared


def _rs_momentum_regime_lookback() -> int:
    return 200


register_expression_strategy(
    "rs_momentum_regime",
    entry="rs > sma(rs, 50) and close > sma(close, 200)",
    exit="rs < sma(rs, 50)",
    prepare_bars=_prepare_rs_momentum_regime,
    required_lookback=_rs_momentum_regime_lookback,
)
