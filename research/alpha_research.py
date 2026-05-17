"""Autoresearch: Backtest 4 alpha strategies on US + India markets.

Strategies tested:
1. 52-Week High Momentum
2. RSI-2 Mean Reversion (with trend filter)
3. Golden Cross + Volume
4. EMA Trend + Volume Filter (improved version of existing)

Markets: US (S&P 500) and India (Nifty 50)
Period: 2020-01-01 to 2025-01-01 (5 years)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.universes import load_current_universe


@dataclass
class StrategyResult:
    name: str
    market: str
    n_trades: int
    win_rate: float
    avg_return: float
    total_pnl: float
    sharpe: float
    max_dd: float
    profit_factor: float
    avg_win: float
    avg_loss: float


def run_strategy_backtest(
    name: str,
    entry_expr: str,
    exit_expr: str | None,
    market: str,
    universe_name: str,
    start: date,
    end: date,
    hold: int = 20,
    top: int = 10,
    stop_loss: float = 0.07,
    take_profit: float = 0.15,
    trailing_stop: float = 0.05,
    slippage_bps: float = 5.0,
    commission_bps: float = 10.0,
    initial_capital: float = 1_000_000.0,
    slippage_model: str | None = None,
    half_spread_bps: float = 0.0,
) -> StrategyResult | None:
    """Run a single strategy backtest and return metrics."""

    fetcher = YFinancePriceFetcher()

    # Load universe
    try:
        universe = load_current_universe(universe_name, as_of=end, use_cache=True)
        tickers = list(universe.symbols)
        print(f"  Loaded {len(tickers)} tickers from {universe_name}")
    except Exception as e:
        print(f"  Could not load universe {universe_name}: {e}")
        return None

    # Fetch prices
    start_fetch = start - timedelta(days=90)
    end_fetch = end + timedelta(days=1)
    yf_symbols = [tv_to_yf(t, market) for t in tickers]
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv_sym in tickers:
        yf_sym = tv_to_yf(tv_sym, market)
        df = price_panel.get(yf_sym)
        if df is not None and not df.empty:
            bars_by_tv[tv_sym] = df

    n_fetched = len(bars_by_tv)
    print(f"  Fetched {n_fetched} symbols")
    if n_fetched < 10:
        print("  Not enough data.")
        return None

    bench = "SPY" if market == "us" else "^NSEI"

    cfg = BacktestConfig(
        market=market,
        as_of=end,
        hold=hold,
        top=top,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
        initial_capital=initial_capital,
        benchmark=bench,
        tickers=tuple(tickers),
        universe_file=None,
        max_universe=0,
        min_price=1.0 if market == "us" else 10.0,
        min_avg_dollar_volume=None,
        avg_dollar_volume_window=20,
        reserve_multiple=3,
        reinvest=False,
        slippage_model=slippage_model,
        gap_fills=True,
        entry_order_type="moo",
        entry_limit_bps=None,
        allow_reentry=False,
        max_reentries=0,
        partial_exits=(),
        price_adjustment="full",
    )

    print(f"  Running backtest: {name}...")
    result = run_rolling_backtest(cfg, fetcher, start_date=start, end_date=end)
    trades = result.trades

    if len(trades) < 10:
        print(f"  Only {len(trades)} trades - not enough.")
        return None

    # Calculate metrics
    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]

    win_rate = len(wins) / len(trades)
    avg_return = np.mean([t.return_pct for t in trades])
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.return_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.return_pct for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")

    # Sharpe approximation (daily returns from trades)
    daily_pnls = {}
    for t in trades:
        key = t.exit_date
        daily_pnls[key] = daily_pnls.get(key, 0) + t.pnl
    if daily_pnls:
        returns = np.array(list(daily_pnls.values()))
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cumulative = np.cumsum([t.pnl for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (peak - cumulative) / peak if np.any(peak > 0) else np.zeros_like(cumulative)
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    return StrategyResult(
        name=name,
        market=market,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_return=avg_return,
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_dd=max_dd,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def print_results(results: list[StrategyResult]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(f"{'Strategy':<30} {'Market':<8} {'Trades':<8} {'Win%':<8} {'AvgRet':<10} {'P&L':<15} {'Sharpe':<8} {'MaxDD':<8} {'PF':<8}")
    print("=" * 100)

    for r in results:
        print(f"{r.name:<30} {r.market:<8} {r.n_trades:<8} {r.win_rate:<8.1%} {r.avg_return:<10.2%} ${r.total_pnl:<14,.0f} {r.sharpe:<8.2f} {r.max_dd:<8.1%} {r.profit_factor:<8.2f}")

    print("=" * 100)

    # Best by market
    for market in ["us", "india"]:
        market_results = [r for r in results if r.market == market]
        if not market_results:
            continue
        best = max(market_results, key=lambda r: r.total_pnl)
        print(f"\nBest for {market.upper()}: {best.name} | P&L=${best.total_pnl:,.0f} | Win={best.win_rate:.1%} | Sharpe={best.sharpe:.2f}")


def main() -> None:
    START = date(2020, 1, 1)
    END = date(2025, 1, 1)

    strategies = [
        # (name, entry_expr, exit_expr, hold, top, stop_loss, take_profit, trailing_stop)
        (
            "52w_high_momentum",
            "close > highest(high, 252) * 0.95 and volume > sma(volume, 20)",
            "crossunder(close, ema(close, 20))",
            20, 10, 0.08, 0.20, 0.06,
        ),
        (
            "rsi2_mean_reversion",
            "rsi(close, 2) < 20 and close > ema(close, 200)",
            "rsi(close, 2) > 60",
            5, 10, 0.03, 0.08, 0.02,
        ),
        (
            "golden_cross_volume",
            "crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)",
            "crossunder(sma(close, 50), sma(close, 200))",
            30, 10, 0.10, 0.25, 0.08,
        ),
        (
            "ema_trend_volume",
            "close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)",
            "crossunder(close, ema(close, 20))",
            20, 10, 0.07, 0.15, 0.05,
        ),
        (
            "atr_breakout",
            "close > highest(close, 5) and atr(14) > sma(atr(14), 20) and volume > sma(volume, 20) * 1.5",
            "close < ema(close, 10)",
            10, 10, 0.05, 0.15, 0.04,
        ),
    ]

    markets = [
        ("us", "sp500"),
        ("india", "nifty50"),
    ]

    all_results: list[StrategyResult] = []

    print("=" * 60)
    print("ALPHA STRATEGY RESEARCH: US + INDIA")
    print(f"Period: {START} to {END}")
    print("=" * 60)

    for market, universe in markets:
        print(f"\n--- Market: {market.upper()} | Universe: {universe} ---")
        for name, entry, exit_expr, hold, top, sl, tp, trail in strategies:
            print(f"\nStrategy: {name}")
            result = run_strategy_backtest(
                name=name,
                entry_expr=entry,
                exit_expr=exit_expr,
                market=market,
                universe_name=universe,
                start=START,
                end=END,
                hold=hold,
                top=top,
                stop_loss=sl,
                take_profit=tp,
                trailing_stop=trail,
            )
            if result:
                all_results.append(result)

    print_results(all_results)

    # Save results
    out_dir = Path(__file__).parent / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dict = [
        {
            "name": r.name,
            "market": r.market,
            "n_trades": r.n_trades,
            "win_rate": r.win_rate,
            "avg_return": r.avg_return,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe,
            "max_dd": r.max_dd,
            "profit_factor": r.profit_factor,
            "avg_win": r.avg_win,
            "avg_loss": r.avg_loss,
        }
        for r in all_results
    ]
    with open(out_dir / "alpha_backtest_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {out_dir / 'alpha_backtest_results.json'}")

    # Final recommendation
    if all_results:
        best_overall = max(all_results, key=lambda r: r.sharpe if r.sharpe > 0 else r.total_pnl)
        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATION")
        print("=" * 60)
        print(f"Best Strategy: {best_overall.name}")
        print(f"Market: {best_overall.market.upper()}")
        print(f"Trades: {best_overall.n_trades}")
        print(f"Win Rate: {best_overall.win_rate:.1%}")
        print(f"Avg Return: {best_overall.avg_return:.2%}")
        print(f"Total P&L: ${best_overall.total_pnl:,.2f}")
        print(f"Sharpe: {best_overall.sharpe:.2f}")
        print(f"Max DD: {best_overall.max_dd:.1%}")
        print(f"Profit Factor: {best_overall.profit_factor:.2f}")


if __name__ == "__main__":
    main()
