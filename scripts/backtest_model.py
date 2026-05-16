"""Backtest the trained ML model on out-of-sample data."""
from __future__ import annotations

import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal import BreakoutFeatureExtractor, SignalConfidenceModel

# OOS period (training was 2024-01-01 to 2025-01-01)
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM"]
MARKET = "us"
START = date(2023, 1, 1)
END = date(2024, 1, 1)

def main() -> None:
    model_path = Path(__file__).parent / "training_data" / "model.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("Loading model...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    print(f"Model features: {feature_names}")

    print(f"\nFetching price data for OOS period: {START} to {END}...")
    fetcher = YFinancePriceFetcher()
    start_fetch = START - timedelta(days=60)
    end_fetch = END + timedelta(days=1)

    yf_symbols = [tv_to_yf(t, MARKET) for t in TICKERS]
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv_sym in TICKERS:
        yf_sym = tv_to_yf(tv_sym, MARKET)
        df = price_panel.get(yf_sym)
        if df is not None and not df.empty:
            bars_by_tv[tv_sym] = df

    print(f"Fetched {len(bars_by_tv)} symbols")

    # Get benchmark bars
    bench_df = price_panel.get("SPY")
    all_bars = {sym: df for sym, df in bars_by_tv.items()}

    bench = "SPY"
    cfg = BacktestConfig(
        market=MARKET,
        as_of=END,
        hold=20,
        top=5,
        entry_expr="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
        exit_expr="crossunder(close, ema(close, 20))",
        stop_loss=0.07,
        take_profit=0.15,
        trailing_stop=0.05,
        slippage_bps=5.0,
        commission_bps=10.0,
        initial_capital=100_000.0,
        benchmark=bench,
        tickers=tuple(TICKERS),
        universe_file=None,
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

    print("Running backtest...")
    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    print(f"Backtest complete: {len(result.trades)} trades")

    if len(result.trades) < 10:
        print("WARNING: Not enough trades for meaningful evaluation.")
        sys.exit(1)

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")

    # Build features for each trade
    extractor = BreakoutFeatureExtractor()
    trades_list = result.trades

    print("\nBuilding features for OOS trades...")
    X_rows = []
    valid_trades = []
    for trade in trades_list:
        bars = bars_by_tv.get(trade.ticker)
        if bars is None or bars.empty:
            continue
        features = extractor.extract(bars, benchmark_bars=bench_df, all_bars=all_bars)
        if features.empty:
            continue
        signal_ts = pd.Timestamp(trade.signal_date)
        mask = features.index <= signal_ts
        if not mask.any():
            continue
        row = features.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        X_rows.append(row)
        valid_trades.append(trade)

    if not X_rows:
        print("No valid feature rows after alignment.")
        sys.exit(1)

    X = pd.concat(X_rows, ignore_index=True)
    X = X[feature_names].fillna(0.0)
    print(f"Feature matrix shape: {X.shape}")

    # Predict
    print("\nRunning predictions...")
    y_proba = model.predict_proba(X)[:, 1]

    # Evaluate at different thresholds
    print("\n" + "=" * 60)
    print("MODEL BACKTEST RESULTS (OOS: 2023-01-01 to 2024-01-01)")
    print("=" * 60)

    # Baseline: all trades
    baseline_pnl = sum(t.pnl for t in valid_trades)
    baseline_return = sum(t.return_pct for t in valid_trades) / len(valid_trades)
    baseline_wins = sum(1 for t in valid_trades if t.return_pct > 0)

    print(f"\nBaseline (all {len(valid_trades)} trades):")
    print(f"  Total P&L: ${baseline_pnl:,.2f}")
    print(f"  Avg return: {baseline_return:.2%}")
    print(f"  Win rate: {baseline_wins}/{len(valid_trades)} = {baseline_wins/len(valid_trades):.1%}")

    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = y_proba >= threshold
        n = mask.sum()
        if n == 0:
            continue
        selected = [valid_trades[i] for i in range(len(valid_trades)) if mask[i]]
        pnl = sum(t.pnl for t in selected)
        avg_ret = sum(t.return_pct for t in selected) / n
        wins = sum(1 for t in selected if t.return_pct > 0)
        print(f"\n  Threshold {threshold:.2f} ({n} trades):")
        print(f"    Total P&L: ${pnl:,.2f}")
        print(f"    Avg return: {avg_ret:.2%}")
        print(f"    Win rate: {wins}/{n} = {wins/n:.1%}")
        print(f"    vs Baseline P&L: {pnl - baseline_pnl:+.2f}")

    # Top-N by confidence
    print("\n" + "-" * 40)
    print("Top-N by confidence:")
    sorted_indices = np.argsort(y_proba)[::-1]
    for n in [10, 20, 30, 50]:
        if n > len(valid_trades):
            continue
        top_n = [valid_trades[i] for i in sorted_indices[:n]]
        pnl = sum(t.pnl for t in top_n)
        avg_ret = sum(t.return_pct for t in top_n) / n
        wins = sum(1 for t in top_n if t.return_pct > 0)
        print(f"\n  Top {n}:")
        print(f"    Total P&L: ${pnl:,.2f}")
        print(f"    Avg return: {avg_ret:.2%}")
        print(f"    Win rate: {wins}/{n} = {wins/n:.1%}")
        print(f"    vs Baseline P&L: {pnl - baseline_pnl:+.2f}")

    # Calibration check
    print("\n" + "-" * 40)
    print("Calibration check (predicted vs actual win rate by bucket):")
    actuals = [t.return_pct > 0 for t in valid_trades]
    bins = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    for lo, hi in bins:
        mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        actual_win_rate = sum(actuals[i] for i in range(len(actuals)) if mask[i]) / n_bin
        print(f"  Predicted {lo:.2f}-{hi:.2f}: {n_bin} trades, actual win rate {actual_win_rate:.1%}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
