"""Portfolio execution and result ranking for vectorbt parameter sweeps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click
import numpy as np
import pandas as pd

from screener.backtester.costs import CostModel, build_cost_model
from screener.backtester.vbt.config import (
    COMMISSION_BPS_DEFAULT,
    COST_MODEL_DEFAULT,
    FEE_NOTIONAL_DEFAULT,
    INITIAL_CAPITAL_DEFAULT,
    MetricName,
    SOLO_INDICATORS_NAN_SLOW,
)
from screener.backtester.vbt.indicators import (
    STRATEGY_BUILDERS,
    _build_indicator_signal_panels,
    iter_indicator_combos,
)


def vbt_fee_fraction(
    cost_model: str | CostModel = COST_MODEL_DEFAULT,
    *,
    commission_bps: float = COMMISSION_BPS_DEFAULT,
    notional: float = FEE_NOTIONAL_DEFAULT,
) -> float:
    """Approximate per-side fee fraction for vectorbt's single ``fees`` param.

    vectorbt applies one scalar on every fill. We average buy and sell
    :meth:`~screener.backtester.costs.CostModel.side_cost_fraction` at a
    representative ``notional``. Per-share components (e.g. FINRA TAF) are
    omitted because the fraction API has no share count.
    """
    model = (
        cost_model
        if not isinstance(cost_model, str)
        else build_cost_model(cost_model, commission_bps=commission_bps)
    )
    n = abs(float(notional))
    if n <= 0.0:
        n = FEE_NOTIONAL_DEFAULT
    buy = float(model.side_cost_fraction("buy", n))
    sell = float(model.side_cost_fraction("sell", n))
    if buy < 0.0:
        buy = 0.0
    if sell < 0.0:
        sell = 0.0
    return 0.5 * (buy + sell)


def _require_vectorbt() -> Any:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise click.ClickException(
            "vectorbt is not installed. Install optional deps with: "
            "uv sync --extra vectorbt"
        ) from exc
    return vbt


def _scalar_metric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (pd.Series, pd.DataFrame)):
        flat = value.to_numpy().ravel()
        if flat.size == 0:
            return float("nan")
        return float(flat[0])
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.ravel()[0])


def run_combo_backtest(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    hold: int,
    *,
    vbt: Any,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    fees: float | None = None,
    cost_model: str = COST_MODEL_DEFAULT,
    commission_bps: float = COMMISSION_BPS_DEFAULT,
) -> dict[str, float | int]:
    entries, exits = STRATEGY_BUILDERS["sma_cross"](close, fast, slow, hold, vbt)
    # Match custom engine MOO semantics: signals on bar t fill at bar t+1 open.
    # Shift entries/exits by 1 bar and price the fill at the next bar's open.
    entries_shifted = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_shifted = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    fill_price = open_ if open_ is not None else close
    fee_frac = (
        float(fees)
        if fees is not None
        else vbt_fee_fraction(cost_model, commission_bps=commission_bps)
    )
    pf = vbt.Portfolio.from_signals(
        close,
        entries_shifted,
        exits_shifted,
        price=fill_price,
        init_cash=float(initial_capital),
        fees=fee_frac,
        slippage=0.0,
        group_by=True,
        cash_sharing=True,
        freq="1D",
    )
    win_rate = _scalar_metric(pf.trades.win_rate())
    trade_count = pf.trades.count()
    if isinstance(trade_count, (pd.Series, pd.DataFrame)):
        trade_count = int(trade_count.to_numpy().sum())
    else:
        trade_count = int(trade_count)
    return {
        "fast": fast,
        "slow": slow,
        "hold": hold,
        "sharpe": _scalar_metric(pf.sharpe_ratio()),
        "total_return": _scalar_metric(pf.total_return()),
        "calmar": _scalar_metric(pf.calmar_ratio()),
        "max_drawdown": _scalar_metric(pf.max_drawdown()),
        "win_rate": win_rate,
        "trades": trade_count,
    }


def _combo_metric(
    series: pd.Series | float, ind: str, fast: int, slow: int, hold: int
) -> float:
    if isinstance(series, pd.Series):
        return _scalar_metric(series.loc[(ind, fast, slow, hold)])
    return _scalar_metric(series)


def _portfolio_chunk_metrics(
    close: pd.DataFrame,
    fill_price: pd.DataFrame,
    entries_chunk: pd.DataFrame,
    exits_chunk: pd.DataFrame,
    *,
    vbt: Any,
    initial_capital: float,
    fees: float = 0.0,
) -> tuple[Any, ...]:
    """Run one ``Portfolio.from_signals`` call over a slice of combo columns.

    Materialising the tiled close/price for every combo at once peaks at
    ~``n_days * n_tickers * n_combos * 8`` bytes. Chunking by combo keeps the
    peak bounded for large universes.
    """
    n_combos = entries_chunk.shape[1] // close.shape[1]
    tiled_close = np.tile(close.to_numpy(), (1, n_combos))
    close_broadcast = pd.DataFrame(
        tiled_close, index=close.index, columns=entries_chunk.columns, copy=False
    )
    if fill_price is close:
        price_broadcast = close_broadcast
    else:
        tiled_price = np.tile(fill_price.to_numpy(), (1, n_combos))
        price_broadcast = pd.DataFrame(
            tiled_price,
            index=fill_price.index,
            columns=entries_chunk.columns,
            copy=False,
        )
    pf = vbt.Portfolio.from_signals(
        close_broadcast,
        entries_chunk,
        exits_chunk,
        price=price_broadcast,
        init_cash=float(initial_capital),
        fees=float(fees),
        slippage=0.0,
        group_by=["indicator", "fast", "slow", "hold"],
        cash_sharing=True,
        freq="1D",
    )
    return (
        pf.sharpe_ratio(),
        pf.total_return(),
        pf.calmar_ratio(),
        pf.max_drawdown(),
        pf.trades.win_rate(),
        pf.trades.count(),
    )


def run_parameter_sweep(
    close: pd.DataFrame,
    *,
    fast_values: list[int],
    slow_values: list[int],
    hold_values: list[int],
    indicators: list[str] | None = None,
    breakout_windows: list[int] | None = None,
    bbands_windows: list[int] | None = None,
    supertrend_periods: list[int] | None = None,
    keltner_windows: list[int] | None = None,
    rsi_thresholds: list[int] | None = None,
    obv_ema_windows: list[int] | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    open_: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL_DEFAULT,
    chunk_size: int | None = None,
    cost_model: str = COST_MODEL_DEFAULT,
    commission_bps: float = COMMISSION_BPS_DEFAULT,
    require_vectorbt_fn: Callable[[], Any] | None = None,
    portfolio_chunk_metrics_fn: Callable[..., tuple[Any, ...]] | None = None,
) -> pd.DataFrame:
    require_vectorbt = require_vectorbt_fn or _require_vectorbt
    chunk_metrics = portfolio_chunk_metrics_fn or _portfolio_chunk_metrics
    vbt = require_vectorbt()
    if indicators is None:
        indicators = ["sma"]
    combos = iter_indicator_combos(
        indicators,
        fast_values,
        slow_values,
        hold_values,
        breakout_windows=breakout_windows,
        bbands_windows=bbands_windows,
        supertrend_periods=supertrend_periods,
        keltner_windows=keltner_windows,
        rsi_thresholds=rsi_thresholds,
        obv_ema_windows=obv_ema_windows,
    )
    if not combos:
        raise ValueError("No valid parameter combinations to evaluate.")

    entries, exits = _build_indicator_signal_panels(
        close, combos, vbt=vbt, high=high, low=low, volume=volume
    )
    entries_shifted = entries.astype(bool).shift(1, fill_value=False).astype(bool)
    exits_shifted = exits.astype(bool).shift(1, fill_value=False).astype(bool)
    fill_price = open_ if open_ is not None else close
    fee_frac = vbt_fee_fraction(cost_model, commission_bps=commission_bps)

    n_tickers = close.shape[1]
    n_combos = len(combos)
    # Bound peak memory: tiled close+price for one chunk costs
    # ``2 * n_days * n_tickers * chunk * 8`` bytes. Aim for <1 GiB per chunk.
    if chunk_size is None:
        n_days = close.shape[0]
        bytes_per_combo = n_days * n_tickers * 8 * 2  # close + price tile
        target_bytes = 1 * 1024 * 1024 * 1024  # ~1 GiB
        chunk_size = max(1, min(n_combos, target_bytes // max(1, bytes_per_combo)))

    sharpe_parts: list[pd.Series] = []
    total_return_parts: list[pd.Series] = []
    calmar_parts: list[pd.Series] = []
    max_drawdown_parts: list[pd.Series] = []
    win_rate_parts: list[pd.Series] = []
    trade_count_parts: list[pd.Series] = []

    for start in range(0, n_combos, chunk_size):
        stop = min(start + chunk_size, n_combos)
        col_start = start * n_tickers
        col_stop = stop * n_tickers
        entries_chunk = entries_shifted.iloc[:, col_start:col_stop]
        exits_chunk = exits_shifted.iloc[:, col_start:col_stop]
        sharpe_c, total_c, calmar_c, dd_c, win_c, trades_c = chunk_metrics(
            close,
            fill_price,
            entries_chunk,
            exits_chunk,
            vbt=vbt,
            initial_capital=initial_capital,
            fees=fee_frac,
        )
        sharpe_parts.append(sharpe_c)
        total_return_parts.append(total_c)
        calmar_parts.append(calmar_c)
        max_drawdown_parts.append(dd_c)
        win_rate_parts.append(win_c)
        trade_count_parts.append(trades_c)

    def _concat(parts: list[Any]) -> pd.Series | float:
        cleaned = [p for p in parts if isinstance(p, pd.Series) and not p.empty]
        if not cleaned:
            return parts[0] if parts else float("nan")
        return pd.concat(cleaned)

    sharpe = _concat(sharpe_parts)
    total_return = _concat(total_return_parts)
    calmar = _concat(calmar_parts)
    max_drawdown = _concat(max_drawdown_parts)
    win_rate = _concat(win_rate_parts)
    trade_count = _concat(trade_count_parts)

    rows: list[dict[str, Any]] = []
    for ind, fast_or_w, slow, hold in combos:
        trades_val = _combo_metric(trade_count, ind, fast_or_w, slow, hold)
        row: dict[str, Any] = {
            "indicator": ind,
            "fast": fast_or_w,
            "slow": (float("nan") if ind in SOLO_INDICATORS_NAN_SLOW else slow),
            "hold": hold,
            "sharpe": _combo_metric(sharpe, ind, fast_or_w, slow, hold),
            "total_return": _combo_metric(total_return, ind, fast_or_w, slow, hold),
            "calmar": _combo_metric(calmar, ind, fast_or_w, slow, hold),
            "max_drawdown": _combo_metric(max_drawdown, ind, fast_or_w, slow, hold),
            "win_rate": _combo_metric(win_rate, ind, fast_or_w, slow, hold),
            "trades": int(trades_val) if np.isfinite(trades_val) else 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def rank_results(df: pd.DataFrame, metric: MetricName) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    sort_key = df[metric].replace([np.inf, -np.inf], np.nan)
    return (
        df.assign(_sort_key=sort_key)
        .sort_values("_sort_key", ascending=False, kind="stable", na_position="last")
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )
