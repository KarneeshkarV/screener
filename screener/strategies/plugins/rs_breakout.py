"""RS breakout: Pine entry expression + custom bar prep + lookback requirement."""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, strategy


def _prepare_rs_breakout(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    from screener.rs_breakout import india_symbol, prepare_backtest_frames
    from screener.unusual_volume.delivery import load_delivery_panel

    benchmark_bars = ctx.price_panel.get(ctx.cfg.benchmark, pd.DataFrame())
    if benchmark_bars is None or benchmark_bars.empty:
        ctx.warnings.append(
            f"benchmark data unavailable for rs_breakout: {ctx.cfg.benchmark}"
        )
        return ctx.bars_by_tv

    delivery_panel = pd.DataFrame()
    if ctx.cfg.market == "india":
        history_days = max(
            (pd.Timestamp(ctx.end) - pd.Timestamp(ctx.start)).days + 14, 40
        )
        try:
            delivery_panel = load_delivery_panel(
                [india_symbol(symbol) for symbol in ctx.tv_symbols],
                ctx.end,
                history_days=history_days,
            )
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            ctx.warnings.append(f"delivery panel unavailable for rs_breakout: {exc}")

    return prepare_backtest_frames(
        ctx.bars_by_tv,
        benchmark_bars,
        market=ctx.cfg.market,
        delivery_panel=delivery_panel,
    )


def _rs_breakout_lookback() -> int:
    from screener.rs_breakout import required_history_bars

    return required_history_bars()


@strategy(
    "rs_breakout",
    entry="rs_breakout_entry > 0",
    exit=None,
    prepare_bars=_prepare_rs_breakout,
    required_lookback=_rs_breakout_lookback,
)
def _rs_breakout() -> None:
    """Expression-only strategy. Body unused."""
