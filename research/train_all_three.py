"""Train all 3 ML enhancements:
1. Regime-aware models (bull/bear/chop)
2. Model hooked into RS Breakout signals
3. Kelly position sizing calibration

Uses 100 liquid US stocks, 2022-2025 for training.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal_v3 import SimpleSignalModel, SimpleFeatureExtractor
from screener.ml_signal_regime import RegimeAwareModel
from screener.ml_kelly import confidence_to_size, kelly_size
from screener.regime import RegimeDetector

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE", "BAC", "CRM", "TMO", "ACN", "MCD",
    "LIN", "NKE", "ABT", "DIS", "TXN", "VZ", "CMCSA", "PM", "NEE", "RTX",
    "HON", "INTC", "IBM", "QCOM", "AMGN", "LOW", "SPGI", "UNP", "GS", "CAT",
    "T", "BMY", "BLK", "DE", "SYK", "MDT", "LMT", "ELV", "GILD", "SCHW",
    "AMAT", "C", "ADI", "CI", "PYPL", "MU", "SBUX", "TJX", "MMC", "DHR",
    "VRTX", "PGR", "ISRG", "LRCX", "BDX", "NOW", "PLD", "MS", "AON", "ZTS",
    "FIS", "APD", "ICE", "TGT", "REGN", "SO", "CL", "ITW", "CSX", "HUM",
    "EOG", "CME", "PNC", "SHW", "SLB", "EQIX", "BSX", "ETN", "FDX", "MCO",
]
MARKET = "us"
START = date(2022, 1, 1)
END = date(2025, 1, 1)


def load_data():
    print(f"\nFetching data for {len(TICKERS)} stocks, {START} to {END}...")
    fetcher = YFinancePriceFetcher()
    start_fetch = START - timedelta(days=90)
    end_fetch = END + timedelta(days=1)

    # Fetch SPY separately to ensure we have it
    print("Fetching SPY benchmark...")
    bench_panel = fetcher.fetch(["SPY"], start_fetch, end_fetch)
    bench_df = bench_panel.get("SPY")
    if bench_df is None or bench_df.empty:
        print("WARNING: Could not fetch SPY benchmark!")
        bench_df = pd.DataFrame()

    yf_symbols = [tv_to_yf(t, MARKET) for t in TICKERS]
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv_sym in TICKERS:
        yf_sym = tv_to_yf(tv_sym, MARKET)
        df = price_panel.get(yf_sym)
        if df is not None and not df.empty:
            bars_by_tv[tv_sym] = df

    print(f"Fetched {len(bars_by_tv)} symbols")
    if bench_df.empty:
        print("WARNING: No benchmark data - regime training will be skipped")
    return bars_by_tv, bench_df, fetcher


def generate_breakout_trades(bars_by_tv, bench_df, fetcher):
    """Generate training trades using RS Breakout-like signals instead of EMA trends."""
    print("\nGenerating breakout-based trades for training...")

    # Use a stronger breakout entry: close > 20d high + volume surge
    cfg = BacktestConfig(
        market=MARKET,
        as_of=END,
        hold=15,
        top=10,
        entry_expr="close > highest(high, 20) * 0.98 and close > ema(close, 50)",
        exit_expr="crossunder(close, ema(close, 20)) or close < lowest(low, 10) * 1.02",
        stop_loss=0.06,
        take_profit=0.20,
        trailing_stop=0.04,
        slippage_bps=5.0,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        benchmark="SPY",
        tickers=tuple(TICKERS),
        max_universe=0,
        min_price=None,
        min_avg_dollar_volume=None,
        avg_dollar_volume_window=20,
        reserve_multiple=3,
        reinvest=False,
        slippage_model=None,
        gap_fills=True,
        entry_order_type="moo",
        entry_limit_bps=None,
        allow_reentry=False,
        max_reentries=0,
        partial_exits=(),
        price_adjustment="full",
    )

    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    print(f"Generated {len(result.trades)} breakout trades")
    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")
    return result.trades


def main():
    bars_by_tv, bench_df, fetcher = load_data()
    trades = generate_breakout_trades(bars_by_tv, bench_df, fetcher)

    if len(trades) < 100:
        print("Not enough trades for training.")
        sys.exit(1)

    out_dir = Path(__file__).parent / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save trades
    trades_dict = [
        {
            "ticker": t.ticker,
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
        }
        for t in trades
    ]
    with open(out_dir / "breakout_trades.json", "w") as f:
        json.dump({"trades": trades_dict}, f)

    print("\n" + "=" * 60)
    print("1. TRAINING BASELINE MODEL (v3, all trades)")
    print("=" * 60)
    baseline_model = SimpleSignalModel()
    baseline_model.train(trades=trades, bars_by_symbol=bars_by_tv, benchmark_bars=bench_df)
    print(f"Baseline AUC: {baseline_model.metrics.get('auc'):.3f}")
    baseline_model.save(out_dir / "model_baseline.pkl")

    print("\n" + "=" * 60)
    print("2. TRAINING REGIME-AWARE MODELS (bull/bear/chop)")
    print("=" * 60)
    if bench_df is not None and not bench_df.empty:
        regime_model = RegimeAwareModel()
        regime_model.train(trades=trades, bars_by_symbol=bars_by_tv, benchmark_bars=bench_df)
        print(f"Regime metrics: {regime_model.metrics}")
        regime_model.save(out_dir / "model_regime.pkl")
    else:
        print("SKIPPED: No benchmark data available for regime detection.")

    print("\n" + "=" * 60)
    print("3. CALIBRATING KELLY SIZING")
    print("=" * 60)
    # Calculate historical avg win/loss for Kelly calibration
    wins = [t.return_pct for t in trades if t.return_pct > 0]
    losses = [abs(t.return_pct) for t in trades if t.return_pct <= 0]
    avg_win = np.mean(wins) if wins else 0.10
    avg_loss = np.mean(losses) if losses else 0.05
    win_rate = len(wins) / len(trades) if trades else 0.5

    # Full Kelly
    b = avg_win / avg_loss if avg_loss > 0 else 2.0
    kelly = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    quarter_kelly = kelly * 0.25

    print(f"Historical: avg_win={avg_win:.2%}, avg_loss={avg_loss:.2%}, win_rate={win_rate:.1%}")
    print(f"Full Kelly: {kelly:.2%}")
    print(f"Quarter Kelly (recommended): {quarter_kelly:.2%}")

    # Kelly calibration data
    kelly_data = {
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_rate": float(win_rate),
        "full_kelly": float(kelly),
        "quarter_kelly": float(quarter_kelly),
    }
    with open(out_dir / "kelly_calibration.json", "w") as f:
        json.dump(kelly_data, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED")
    print("=" * 60)
    print(f"  Baseline: {out_dir / 'model_baseline.pkl'}")
    print(f"  Regime:   {out_dir / 'model_regime.pkl'}")
    print(f"  Kelly:    {out_dir / 'kelly_calibration.json'}")
    print(f"  Trades:   {out_dir / 'breakout_trades.json'}")


if __name__ == "__main__":
    main()
