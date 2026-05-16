"""Fast alpha strategy backtest: 4 strategies x 2 markets, 50 stocks each.

Optimized for speed: 50 liquid stocks, 3-year period, top 5 positions.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig

# Top 50 liquid stocks by market
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE", "BAC", "CRM", "TMO", "ACN", "MCD",
    "LIN", "NKE", "ABT", "DIS", "TXN", "VZ", "CMCSA", "PM", "NEE", "RTX",
    "HON", "INTC", "IBM", "QCOM", "AMGN", "LOW", "SPGI", "UNP", "GS", "CAT",
]

INDIA_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
    "AXISBANK", "MARUTI", "SUNPHARMA", "TATAMOTORS", "TITAN", "BAJAJFINSV",
    "ADANIENT", "COALINDIA", "NESTLEIND", "ULTRACEMCO", "POWERGRID", "ONGC",
    "NTPC", "M&M", "JSWSTEEL", "TATASTEEL", "GRASIM", "ADANIPORTS", "CIPLA",
    "WIPRO", "BRITANNIA", "DRREDDY", "TECHM", "APOLLOHOSP", "BAJAJ-AUTO",
    "EICHERMOT", "SHRIRAMFIN", "TATACONSUM", "HEROMOTOCO", "INDUSINDBK",
    "HINDALCO", "SBILIFE", "DIVISLAB", "HDFCLIFE", "BPCL", "IOC",
]


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


def run_backtest(
    name: str,
    entry: str,
    exit_expr: str | None,
    market: str,
    tickers: list[str],
    start: date,
    end: date,
    hold: int = 20,
    top: int = 5,
    sl: float = 0.07,
    tp: float = 0.15,
    trail: float = 0.05,
) -> StrategyResult | None:
    fetcher = YFinancePriceFetcher()
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

    n = len(bars_by_tv)
    print(f"  Fetched {n} symbols")
    if n < 10:
        return None

    bench = "SPY" if market == "us" else "^NSEI"

    cfg = BacktestConfig(
        market=market, as_of=end, hold=hold, top=top,
        entry_expr=entry, exit_expr=exit_expr,
        stop_loss=sl, take_profit=tp, trailing_stop=trail,
        slippage_bps=5.0, commission_bps=10.0, initial_capital=500_000.0,
        benchmark=bench, tickers=tuple(tickers), max_universe=0,
        min_price=1.0 if market == "us" else 10.0,
        avg_dollar_volume_window=20, reserve_multiple=3, reinvest=False,
        gap_fills=True, entry_order_type="moo",
        allow_reentry=False, max_reentries=0, partial_exits=(),
        price_adjustment="full",
    )

    print(f"  Backtesting...")
    result = run_rolling_backtest(cfg, fetcher, start_date=start, end_date=end)
    trades = result.trades
    if len(trades) < 10:
        print(f"  Only {len(trades)} trades")
        return None

    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]

    win_rate = len(wins) / len(trades)
    avg_ret = np.mean([t.return_pct for t in trades])
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.return_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.return_pct for t in losses]) if losses else 0
    pf = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else 0

    daily = {}
    for t in trades:
        daily[t.exit_date] = daily.get(t.exit_date, 0) + t.pnl
    rets = np.array(list(daily.values()))
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else 0

    cum = np.cumsum([t.pnl for t in trades])
    peak = np.maximum.accumulate(cum)
    dd = np.max((peak - cum) / peak) if np.any(peak > 0) else 0

    return StrategyResult(name, market, len(trades), win_rate, avg_ret, total_pnl, sharpe, dd, pf, avg_win, avg_loss)


def main():
    START = date(2022, 1, 1)
    END = date(2025, 1, 1)

    strategies = [
        ("52w_high", "close > highest(high, 252) * 0.95 and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.08, 0.20, 0.06),
        ("rsi2_rev", "rsi(close, 2) < 20 and close > ema(close, 200)", "rsi(close, 2) > 60", 5, 5, 0.03, 0.08, 0.02),
        ("golden_cross", "crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
        ("ema_vol", "close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ]

    all_results = []

    for market, tickers in [("us", US_TICKERS), ("india", INDIA_TICKERS)]:
        print(f"\n{'='*60}\nMARKET: {market.upper()}\n{'='*60}")
        for name, entry, exit_expr, hold, top, sl, tp, trail in strategies:
            print(f"\nStrategy: {name}")
            r = run_backtest(name, entry, exit_expr, market, tickers, START, END, hold, top, sl, tp, trail)
            if r:
                all_results.append(r)
                print(f"  Result: {r.n_trades} trades, Win={r.win_rate:.1%}, P&L=${r.total_pnl:,.0f}, Sharpe={r.sharpe:.2f}")

    # Print table
    print("\n" + "="*110)
    print(f"{'Strategy':<20} {'Market':<8} {'Trades':<8} {'Win%':<8} {'AvgRet':<10} {'P&L':<15} {'Sharpe':<8} {'MaxDD':<8} {'PF':<8}")
    print("="*110)
    for r in all_results:
        print(f"{r.name:<20} {r.market:<8} {r.n_trades:<8} {r.win_rate:<8.1%} {r.avg_return:<10.2%} ${r.total_pnl:<14,.0f} {r.sharpe:<8.2f} {r.max_dd:<8.1%} {r.profit_factor:<8.2f}")
    print("="*110)

    # Best by market
    for m in ["us", "india"]:
        mr = [r for r in all_results if r.market == m]
        if mr:
            best = max(mr, key=lambda r: r.sharpe if r.sharpe > 0 else r.total_pnl)
            print(f"\nBest {m.upper()}: {best.name} | P&L=${best.total_pnl:,.0f} | Win={best.win_rate:.1%} | Sharpe={best.sharpe:.2f}")

    # Save
    out = Path(__file__).parent / "research"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "alpha_results.json", "w") as f:
        json.dump([r.__dict__ for r in all_results], f, indent=2)

    if all_results:
        best = max(all_results, key=lambda r: r.sharpe if r.sharpe > 0 else r.total_pnl)
        print("\n" + "="*60)
        print("FINAL RECOMMENDATION")
        print("="*60)
        print(f"Strategy: {best.name}")
        print(f"Market: {best.market.upper()}")
        print(f"Trades: {best.n_trades} | Win: {best.win_rate:.1%} | AvgRet: {best.avg_return:.2%}")
        print(f"P&L: ${best.total_pnl:,.2f} | Sharpe: {best.sharpe:.2f} | MaxDD: {best.max_dd:.1%}")
        print(f"Profit Factor: {best.profit_factor:.2f}")


if __name__ == "__main__":
    main()
