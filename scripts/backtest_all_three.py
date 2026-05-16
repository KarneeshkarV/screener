"""Backtest all 3 ML enhancements on OOS data (2019-2022)."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal_v3 import SimpleSignalModel, SimpleFeatureExtractor
from screener.ml_signal_regime import RegimeAwareModel
from screener.ml_kelly import confidence_to_size
from screener.regime import RegimeDetector

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE", "BAC", "CRM", "TMO", "ACN", "MCD",
]
MARKET = "us"
START = date(2019, 1, 1)  # OOS period before training
END = date(2022, 1, 1)


def main():
    data_dir = Path(__file__).parent / "training_data"

    # Load models
    print("Loading models...")
    baseline_model = SimpleSignalModel.load(data_dir / "model_baseline.pkl")
    regime_model = RegimeAwareModel.load(data_dir / "model_regime.pkl")

    with open(data_dir / "kelly_calibration.json") as f:
        kelly_data = json.load(f)

    print(f"Baseline AUC: {baseline_model.metrics.get('auc'):.3f}")
    print(f"Regime: bull={regime_model.metrics.get('bull_auc'):.3f}, "
          f"bear={regime_model.metrics.get('bear_auc'):.3f}, "
          f"chop={regime_model.metrics.get('chop_auc'):.3f}")
    print(f"Kelly: avg_win={kelly_data['avg_win']:.2%}, avg_loss={kelly_data['avg_loss']:.2%}")

    # Fetch OOS data
    print(f"\nFetching OOS data: {START} to {END}...")
    fetcher = YFinancePriceFetcher()
    start_fetch = START - timedelta(days=90)
    end_fetch = END + timedelta(days=1)

    bench_panel = fetcher.fetch(["SPY"], start_fetch, end_fetch)
    bench_df = bench_panel.get("SPY")
    if bench_df is None or bench_df.empty:
        print("ERROR: Could not fetch SPY")
        sys.exit(1)

    yf_symbols = [tv_to_yf(t, MARKET) for t in TICKERS]
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv_sym in TICKERS:
        yf_sym = tv_to_yf(tv_sym, MARKET)
        df = price_panel.get(yf_sym)
        if df is not None and not df.empty:
            bars_by_tv[tv_sym] = df

    print(f"Fetched {len(bars_by_tv)} symbols")

    # Generate OOS trades with same breakout strategy
    cfg = BacktestConfig(
        market=MARKET, as_of=END, hold=15, top=10,
        entry_expr="close > highest(high, 20) * 0.98 and close > ema(close, 50)",
        exit_expr="crossunder(close, ema(close, 20)) or close < lowest(low, 10) * 1.02",
        stop_loss=0.06, take_profit=0.20, trailing_stop=0.04,
        slippage_bps=5.0, commission_bps=10.0, initial_capital=1_000_000.0,
        benchmark="SPY", tickers=tuple(TICKERS), max_universe=0,
        reserve_multiple=3, reinvest=False, gap_fills=True,
        entry_order_type="moo", allow_reentry=False, max_reentries=0,
        partial_exits=(), price_adjustment="full",
    )

    print("Running OOS backtest...")
    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    trades = result.trades
    print(f"OOS trades: {len(trades)}")

    if len(trades) < 50:
        print("Not enough OOS trades.")
        sys.exit(1)

    wins = sum(1 for t in trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(trades)} = {wins/len(trades):.1%}")

    # Build features for all trades
    extractor = SimpleFeatureExtractor()
    X_rows = []
    valid_trades = []
    for trade in trades:
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
    X = X[baseline_model.feature_names].fillna(0.0)

    print(f"Feature matrix: {X.shape}")

    # Predictions
    baseline_proba = baseline_model.predict(X)

    # Regime predictions
    regime_proba = np.zeros(len(X))
    for i, trade in enumerate(valid_trades):
        sig_ts = pd.Timestamp(trade.signal_date)
        bench_sub = bench_df[bench_df.index <= sig_ts]
        if len(bench_sub) >= 30:
            trend = RegimeDetector.trend_regime(
                pd.to_numeric(bench_sub["close"], errors="coerce").dropna(),
                fast=50, slow=200,
            )
            try:
                regime_proba[i] = regime_model.predict(X.iloc[[i]], regime=trend)[0]
            except Exception:
                regime_proba[i] = baseline_proba[i]
        else:
            regime_proba[i] = baseline_proba[i]

    # Evaluate all approaches
    print("\n" + "=" * 70)
    print("OOS BACKTEST: ALL 3 APPROACHES")
    print(f"Period: {START} to {END} | Trades: {len(valid_trades)}")
    print("=" * 70)

    baseline_pnl = sum(t.pnl for t in valid_trades)

    def evaluate(name, scores):
        print(f"\n--- {name} ---")
        # Threshold filtering
        for thresh in [0.5, 0.55, 0.6]:
            mask = scores >= thresh
            n = mask.sum()
            if n == 0:
                continue
            selected = [valid_trades[i] for i in range(len(valid_trades)) if mask[i]]
            pnl = sum(t.pnl for t in selected)
            wins_sel = sum(1 for t in selected if t.return_pct > 0)
            print(f"  Threshold {thresh:.2f}: {n} trades, P&L=${pnl:,.2f}, "
                  f"Win={wins_sel}/{n}={wins_sel/n:.1%}, vs Base={pnl-baseline_pnl:+.2f}")

        # Top-N
        sorted_idx = np.argsort(scores)[::-1]
        for n in [20, 50]:
            if n > len(valid_trades):
                continue
            top = [valid_trades[i] for i in sorted_idx[:n]]
            pnl = sum(t.pnl for t in top)
            wins_top = sum(1 for t in top if t.return_pct > 0)
            print(f"  Top {n}: P&L=${pnl:,.2f}, Win={wins_top}/{n}={wins_top/n:.1%}, vs Base={pnl-baseline_pnl:+.2f}")

        # Kelly sizing (simulate with 1M capital, 10% base position)
        sizes = [confidence_to_size(c, fraction=0.25) for c in scores]
        kelly_pnl = 0
        for i, t in enumerate(valid_trades):
            kelly_pnl += t.pnl * sizes[i]
        print(f"  Kelly sizing: P&L=${kelly_pnl:,.2f} vs Base={baseline_pnl:,.2f} "
              f"(mult={kelly_pnl/baseline_pnl:.2f}x)")

    evaluate("BASELINE (single model)", baseline_proba)
    evaluate("REGIME-AWARE (bull/bear/chop)", regime_proba)

    # Calibration check
    print("\n--- Calibration ---")
    actuals = [t.return_pct > 0 for t in valid_trades]
    for lo, hi in [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]:
        mask = (baseline_proba >= lo) & (baseline_proba < hi)
        n = mask.sum()
        if n == 0:
            continue
        actual_wr = sum(actuals[i] for i in range(len(actuals)) if mask[i]) / n
        print(f"  Baseline {lo:.2f}-{hi:.2f}: {n} trades, actual win {actual_wr:.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
