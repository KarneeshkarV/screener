"""Efficient alpha strategy backtest. Pre-fetches data once, tests 4 strategies x 2 markets."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig

US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE", "BAC", "CRM", "TMO", "ACN", "MCD",
]
INDIA_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
    "AXISBANK", "MARUTI", "SUNPHARMA", "TATAMOTORS", "TITAN", "BAJAJFINSV",
    "ADANIENT", "COALINDIA", "NESTLEIND", "ULTRACEMCO", "POWERGRID", "ONGC",
    "NTPC", "M&M", "JSWSTEEL", "TATASTEEL",
]

START = date(2022, 1, 1)
END = date(2025, 1, 1)


@dataclass
class Result:
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


def prefetch(market: str, tickers: list[str]):
    print(f"  Prefetching {len(tickers)} {market} stocks...")
    fetcher = YFinancePriceFetcher()
    s = START - timedelta(days=90)
    e = END + timedelta(days=1)
    yf = [tv_to_yf(t, market) for t in tickers]
    panel = fetcher.fetch(yf, s, e)
    bars = {}
    for tv in tickers:
        df = panel.get(tv_to_yf(tv, market))
        if df is not None and not df.empty:
            bars[tv] = df
    print(f"  Got {len(bars)} symbols")
    return bars, fetcher


def run_strategies(bars: dict, fetcher, market: str):
    bench = "SPY" if market == "us" else "^NSEI"
    tickers = tuple(bars.keys())
    strategies = [
        ("52w_high", "close > highest(high, 252) * 0.95 and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.08, 0.20, 0.06),
        ("rsi2_rev", "rsi(close, 2) < 20 and close > ema(close, 200)", "rsi(close, 2) > 60", 5, 5, 0.03, 0.08, 0.02),
        ("golden_cross", "crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
        ("ema_vol", "close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ]
    results = []
    for name, entry, exit_expr, hold, top, sl, tp, trail in strategies:
        print(f"    {name}...", end="", flush=True)
        cfg = BacktestConfig(
            market=market, as_of=END, hold=hold, top=top,
            entry_expr=entry, exit_expr=exit_expr,
            stop_loss=sl, take_profit=tp, trailing_stop=trail,
            slippage_bps=5.0, commission_bps=10.0, initial_capital=500_000.0,
            benchmark=bench, tickers=tickers, max_universe=0,
            min_price=1.0 if market == "us" else 10.0,
            avg_dollar_volume_window=20, reserve_multiple=3, reinvest=False,
            gap_fills=True, entry_order_type="moo",
            allow_reentry=False, max_reentries=0, partial_exits=(),
            price_adjustment="full",
        )
        t0 = time.time()
        result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
        elapsed = time.time() - t0
        trades = result.trades
        if len(trades) < 5:
            print(f" {len(trades)} trades (too few)")
            continue
        wins = [t for t in trades if t.return_pct > 0]
        losses = [t for t in trades if t.return_pct <= 0]
        wr = len(wins) / len(trades)
        avg_r = np.mean([t.return_pct for t in trades])
        pnl = sum(t.pnl for t in trades)
        pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 0
        daily = {}
        for t in trades:
            daily[t.exit_date] = daily.get(t.exit_date, 0) + t.pnl
        rets = np.array(list(daily.values()))
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else 0
        cum = np.cumsum([t.pnl for t in trades])
        peak = np.maximum.accumulate(cum)
        dd = np.max((peak - cum) / peak) if np.any(peak > 0) else 0
        results.append(Result(name, market, len(trades), wr, avg_r, pnl, sharpe, dd, pf,
                              np.mean([t.return_pct for t in wins]) if wins else 0,
                              np.mean([t.return_pct for t in losses]) if losses else 0))
        print(f" {len(trades)} trades, Win={wr:.1%}, P&L=${pnl:,.0f}, Sharpe={sharpe:.2f} ({elapsed:.1f}s)")
    return results


def main():
    print("=" * 60)
    print("ALPHA STRATEGY RESEARCH")
    print(f"Period: {START} to {END}")
    print("=" * 60)

    all_results = []

    for market, tickers in [("us", US_TICKERS), ("india", INDIA_TICKERS)]:
        print(f"\n--- {market.upper()} ---")
        bars, fetcher = prefetch(market, tickers)
        if len(bars) < 10:
            print(f"  Skipping {market} (only {len(bars)} symbols)")
            continue
        results = run_strategies(bars, fetcher, market)
        all_results.extend(results)

    print("\n" + "=" * 110)
    print(f"{'Strategy':<18} {'Market':<8} {'Trades':<8} {'Win%':<8} {'AvgRet':<10} {'P&L':<15} {'Sharpe':<8} {'MaxDD':<8} {'PF':<8}")
    print("=" * 110)
    for r in all_results:
        print(f"{r.name:<18} {r.market:<8} {r.n_trades:<8} {r.win_rate:<8.1%} {r.avg_return:<10.2%} ${r.total_pnl:<14,.0f} {r.sharpe:<8.2f} {r.max_dd:<8.1%} {r.profit_factor:<8.2f}")
    print("=" * 110)

    for m in ["us", "india"]:
        mr = [r for r in all_results if r.market == m]
        if mr:
            best = max(mr, key=lambda r: r.sharpe if r.sharpe > 0 else r.total_pnl)
            print(f"\nBest {m.upper()}: {best.name} | P&L=${best.total_pnl:,.0f} | Win={best.win_rate:.1%} | Sharpe={best.sharpe:.2f} | MaxDD={best.max_dd:.1%}")

    out = Path(__file__).parent / "research"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "alpha_results.json", "w") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)

    if all_results:
        best = max(all_results, key=lambda r: r.sharpe if r.sharpe > 0 else r.total_pnl)
        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATION")
        print("=" * 60)
        print(f"Strategy: {best.name}")
        print(f"Market:   {best.market.upper()}")
        print(f"Trades:   {best.n_trades}")
        print(f"Win Rate: {best.win_rate:.1%}")
        print(f"Avg Ret:  {best.avg_return:.2%}")
        print(f"P&L:      ${best.total_pnl:,.2f}")
        print(f"Sharpe:   {best.sharpe:.2f}")
        print(f"Max DD:   {best.max_dd:.1%}")
        print(f"PF:       {best.profit_factor:.2f}")
        print(f"Avg Win:  {best.avg_win:.2%}")
        print(f"Avg Loss: {best.avg_loss:.2%}")


if __name__ == "__main__":
    main()
