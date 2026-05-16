"""Generate large-scale training data for ML v4.

Uses multiple strategies + 50 stocks + 2020-2025 to produce 3000+ trades.
Includes US and India markets.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig

US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE", "BAC", "CRM", "TMO", "ACN", "MCD",
    "NKE", "DIS", "ABT", "LIN", "TXN", "VZ", "NEE", "PM", "RTX", "HON",
    "INTC", "T", "IBM", "GS", "LMT", "BA", "CAT", "DE", "LOW", "SBUX",
]

INDIA_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
    "AXISBANK", "MARUTI", "SUNPHARMA", "TATAMOTORS", "TITAN", "BAJAJFINSV",
    "ADANIENT", "COALINDIA", "NESTLEIND", "ULTRACEMCO", "POWERGRID", "ONGC",
    "NTPC", "M&M", "JSWSTEEL", "TATASTEEL", "WIPRO", "TECHM", "GRASIM", "CIPLA",
    "DRREDDY", "TATACONSUM", "EICHERMOT", "DIVISLAB", "BPCL", "HEROMOTOCO",
    "BRITANNIA", "INDUSINDBK", "HINDALCO", "APOLLOHOSP", "UPL", "SHREECEM",
]

STRATEGIES = [
    ("ema_trend", "close > ema(close, 20) and ema(close, 20) > ema(close, 200)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ("ema_vol", "close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ("golden_cross", "crossover(sma(close, 50), sma(close, 200))", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
    ("golden_cross_vol", "crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
    ("rsi2_rev", "rsi(close, 2) < 20 and close > ema(close, 200)", "rsi(close, 2) > 60", 5, 5, 0.03, 0.08, 0.02),
    ("breakout", "close >= highest(close, 252) * 0.95 and volume > sma(volume, 10)", None, 20, 5, 0.08, 0.20, 0.06),
]


def run_strategy(market: str, tickers: list[str], strat: tuple, fetcher, start: date, end: date):
    name, entry, exit_expr, hold, top, sl, tp, trail = strat
    bench = "SPY" if market == "us" else "^NSEI"
    yf_symbols = [tv_to_yf(t, market) for t in tickers]

    start_fetch = start - timedelta(days=90)
    end_fetch = end + timedelta(days=1)
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv in tickers:
        yf = tv_to_yf(tv, market)
        df = price_panel.get(yf)
        if df is not None and not df.empty:
            bars_by_tv[tv] = df

    if len(bars_by_tv) < 10:
        return [], {}

    cfg = BacktestConfig(
        market=market, as_of=end, hold=hold, top=top,
        entry_expr=entry, exit_expr=exit_expr,
        stop_loss=sl, take_profit=tp, trailing_stop=trail,
        slippage_bps=5.0, commission_bps=10.0, initial_capital=500_000.0,
        benchmark=bench, tickers=tuple(bars_by_tv.keys()), max_universe=0,
        min_price=1.0 if market == "us" else 10.0,
        avg_dollar_volume_window=20, reserve_multiple=3, reinvest=False,
        gap_fills=True, entry_order_type="moo",
        allow_reentry=False, max_reentries=0, partial_exits=(),
        price_adjustment="full",
    )

    result = run_rolling_backtest(cfg, fetcher, start_date=start, end_date=end)
    return result.trades, bars_by_tv


def main():
    out_dir = Path(__file__).parent / "training_data_v4"
    out_dir.mkdir(exist_ok=True)

    all_trades = []
    all_bars = {}
    all_benchmarks = {}

    fetcher = YFinancePriceFetcher()
    start = date(2020, 1, 1)
    end = date(2025, 1, 1)

    for market, tickers in [("us", US_TICKERS), ("india", INDIA_TICKERS)]:
        print(f"\n=== {market.upper()} ===")
        bench_sym = "SPY" if market == "us" else "^NSEI"
        bench_df = fetcher.fetch([bench_sym], start - timedelta(days=90), end + timedelta(days=1)).get(bench_sym)
        if bench_df is not None and not bench_df.empty:
            all_benchmarks[market] = bench_df

        for strat in STRATEGIES:
            name = strat[0]
            print(f"  Running {name}...", end="", flush=True)
            trades, bars = run_strategy(market, tickers, strat, fetcher, start, end)
            print(f" {len(trades)} trades")
            for t in trades:
                all_trades.append({
                    "ticker": t.ticker,
                    "market": market,
                    "strategy": name,
                    "rank": t.rank,
                    "signal_date": str(t.signal_date),
                    "entry_date": str(t.entry_date),
                    "entry_price": t.entry_price,
                    "exit_date": str(t.exit_date),
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason),
                    "shares": t.shares,
                    "entry_cost": t.entry_cost,
                    "exit_value": t.exit_value,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "dividend_income": t.dividend_income,
                })
            for sym, df in bars.items():
                key = f"{market}:{sym}"
                if key not in all_bars:
                    all_bars[key] = []
                all_bars[key] = df  # last strategy wins for bars, that's fine

    # Convert bars to JSON
    bars_json = {}
    for key, df in all_bars.items():
        bars_json[key] = [
            {"date": str(idx.date()) if hasattr(idx, "date") else str(idx),
             "open": float(row["open"]), "high": float(row["high"]),
             "low": float(row["low"]), "close": float(row["close"]),
             "volume": float(row["volume"])}
            for idx, row in df.iterrows()
        ]

    bench_json = {}
    for market, df in all_benchmarks.items():
        bench_json[market] = [
            {"date": str(idx.date()) if hasattr(idx, "date") else str(idx),
             "open": float(row["open"]), "high": float(row["high"]),
             "low": float(row["low"]), "close": float(row["close"]),
             "volume": float(row["volume"])}
            for idx, row in df.iterrows()
        ]

    trades_data = {"trades": all_trades}

    with open(out_dir / "trades.json", "w") as f:
        json.dump(trades_data, f, indent=2)
    with open(out_dir / "bars.json", "w") as f:
        json.dump({"bars": bars_json, "benchmarks": bench_json}, f, indent=2)

    wins = sum(1 for t in all_trades if t["return_pct"] > 0)
    print(f"\n{'='*50}")
    print(f"Total trades: {len(all_trades)}")
    print(f"Win rate: {wins}/{len(all_trades)} = {wins/len(all_trades):.1%}")
    print(f"Wrote to {out_dir}")


if __name__ == "__main__":
    main()
