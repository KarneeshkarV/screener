"""Backtest simplified v3 model OOS."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal_v3 import SimpleSignalModel, SimpleFeatureExtractor

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM"]
MARKET = "us"
START = date(2023, 1, 1)
END = date(2024, 1, 1)


def main() -> None:
    model_path = Path(__file__).parent / "training_data" / "model_v3.pkl"
    model = SimpleSignalModel.load(model_path)
    print(f"Loaded v3 model: AUC={model.metrics.get('auc', 'N/A')}")

    print(f"\nFetching OOS data...")
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

    bench_df = price_panel.get("SPY")

    cfg = BacktestConfig(
        market=MARKET, as_of=END, hold=20, top=5,
        entry_expr="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
        exit_expr="crossunder(close, ema(close, 20))",
        stop_loss=0.07, take_profit=0.15, trailing_stop=0.05,
        slippage_bps=5.0, commission_bps=10.0, initial_capital=100_000.0,
        benchmark="SPY", tickers=tuple(TICKERS),
        reserve_multiple=3, reinvest=False,
        gap_fills=True, entry_order_type="moo",
        allow_reentry=False, max_reentries=0,
        partial_exits=(), price_adjustment="full",
    )

    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    print(f"Backtest: {len(result.trades)} trades")

    extractor = SimpleFeatureExtractor()
    X_rows = []
    valid_trades = []
    for trade in result.trades:
        bars = bars_by_tv.get(trade.ticker)
        if bars is None or bars.empty:
            continue
        features = extractor.extract(bars, benchmark_bars=bench_df)
        if features.empty:
            continue
        sig_ts = pd.Timestamp(trade.signal_date)
        mask = features.index <= sig_ts
        if not mask.any():
            continue
        row = features.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        X_rows.append(row)
        valid_trades.append(trade)

    X = pd.concat(X_rows, ignore_index=True)
    X = X[model.feature_names].fillna(0.0)
    y_proba = model.predict(X)

    baseline_pnl = sum(t.pnl for t in valid_trades)
    baseline_wins = sum(1 for t in valid_trades if t.return_pct > 0)

    print("\n" + "=" * 60)
    print("MODEL V3 OOS BACKTEST (2023-01-01 to 2024-01-01)")
    print("=" * 60)
    print(f"\nBaseline ({len(valid_trades)} trades):")
    print(f"  P&L: ${baseline_pnl:,.2f} | Win rate: {baseline_wins}/{len(valid_trades)} = {baseline_wins/len(valid_trades):.1%}")

    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = y_proba >= threshold
        n = mask.sum()
        if n == 0:
            continue
        selected = [valid_trades[i] for i in range(len(valid_trades)) if mask[i]]
        pnl = sum(t.pnl for t in selected)
        wins = sum(1 for t in selected if t.return_pct > 0)
        print(f"\n  Threshold {threshold:.2f} ({n} trades):")
        print(f"    P&L: ${pnl:,.2f} | Win rate: {wins}/{n} = {wins/n:.1%}")
        print(f"    vs Baseline: {pnl - baseline_pnl:+.2f}")

    print("\n  Top-N by confidence:")
    sorted_idx = np.argsort(y_proba)[::-1]
    for n in [10, 20, 30, 50]:
        if n > len(valid_trades):
            continue
        top = [valid_trades[i] for i in sorted_idx[:n]]
        pnl = sum(t.pnl for t in top)
        wins = sum(1 for t in top if t.return_pct > 0)
        print(f"    Top {n}: P&L=${pnl:,.2f} | Win={wins}/{n}={wins/n:.1%} | vs Base={pnl-baseline_pnl:+.2f}")

    print("\n  Calibration:")
    actuals = [t.return_pct > 0 for t in valid_trades]
    for lo, hi in [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]:
        mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        actual_wr = sum(actuals[i] for i in range(len(actuals)) if mask[i]) / n_bin
        print(f"    {lo:.2f}-{hi:.2f}: {n_bin} trades, actual {actual_wr:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
