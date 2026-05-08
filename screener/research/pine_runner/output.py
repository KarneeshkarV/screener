"""Output helpers for the research Pine runner."""
from __future__ import annotations

import json

import numpy as np

from screener.logging_config import get_logger
from screener.research.pine_runner.run import MarketRun
from screener.strategies.registry import STRATEGIES

log = get_logger("pine_runner")


def print_market_table(result: MarketRun) -> None:
    hdr = (
        f"{'Strategy':<18} {'Tkrs':>5} {'Trades':>7} {'Tr/Tk':>6} "
        f"{'Basket':>9} {'Median':>9} {'Bench':>9} {'Alpha':>9} "
        f"{'Win%':>6} {'Exp%':>6}"
    )
    print()
    print("=" * (len(hdr) + 2))
    print(
        f"{result.market.upper()}  |  window {result.window_start.date()} -> "
        f"{result.today}  |  bench={result.benchmark_symbol}="
        f"{'-' if result.benchmark_return is None else f'{result.benchmark_return:+.1%}'}"
    )
    print("=" * (len(hdr) + 2))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name in STRATEGIES:
        results = result.per_strategy[name]
        if not results:
            print(f"{name:<18}  no results  (errors: {result.error_counts[name]})")
            continue
        n_t = len(results)
        returns = [r["total_return"] for r in results]
        total_trades = sum(r["n_trades"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_exp = sum(r["exposure"] for r in results)
        total_bars = sum(r["n_bars"] for r in results) or 1
        basket = float(np.mean(returns))
        med = float(np.median(returns))
        win = (total_wins / total_trades) if total_trades else float("nan")
        alpha = (
            basket - result.benchmark_return
            if result.benchmark_return is not None
            else float("nan")
        )
        rows.append(
            {
                "strategy": name,
                "n": n_t,
                "trades": total_trades,
                "basket": basket,
                "median": med,
                "alpha": alpha,
                "win_rate": win,
                "exposure": total_exp / total_bars,
            }
        )
        print(
            f"{name:<18} {n_t:>5} {total_trades:>7} "
            f"{total_trades / n_t:>6.1f} "
            f"{basket:>+9.1%} {med:>+9.1%} "
            f"{('-' if result.benchmark_return is None else f'{result.benchmark_return:+.1%}'):>9} "
            f"{('-' if np.isnan(alpha) else f'{alpha:+.1%}'):>9} "
            f"{win:>6.1%} {total_exp / total_bars:>6.1%}"
        )
    print()

    if rows:
        best_alpha = max(
            rows, key=lambda r: r["alpha"] if not np.isnan(r["alpha"]) else -9e9
        )
        best_basket = max(rows, key=lambda r: r["basket"])
        best_win = max(rows, key=lambda r: r["win_rate"])
        print("Best in this market:")
        print(
            f"  highest alpha:       {best_alpha['strategy']:<18} "
            f"alpha={best_alpha['alpha']:+.1%}  basket={best_alpha['basket']:+.1%}"
        )
        print(
            f"  highest basket rtn:  {best_basket['strategy']:<18} "
            f"basket={best_basket['basket']:+.1%}"
        )
        print(
            f"  highest win rate:    {best_win['strategy']:<18} "
            f"win={best_win['win_rate']:.1%}  trades={best_win['trades']}"
        )
        print()


def write_trades_json(result: MarketRun, path: str) -> None:
    payload = {
        "market": result.market,
        "window_start": str(result.window_start.date()),
        "window_end": str(result.today),
        "strategies": {},
    }
    for name, results in result.per_strategy.items():
        traded = [r for r in results if r["n_trades"] > 0]
        traded.sort(key=lambda r: r["total_return"], reverse=True)
        payload["strategies"][name] = {
            "n_tickers_traded": len(traded),
            "tickers": [
                {
                    "ticker": r["ticker"],
                    "n_trades": r["n_trades"],
                    "wins": r["wins"],
                    "return": round(r["total_return"], 4),
                }
                for r in traded
            ],
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("backtest.trades_dump_written", path=path)
