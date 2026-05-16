"""Backtest the enhanced v2 model on out-of-sample data."""
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
from screener.ml_signal_v2 import EnhancedSignalModel, EnhancedFeatureExtractor

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM"]
MARKET = "us"
START = date(2023, 1, 1)
END = date(2024, 1, 1)


def main() -> None:
    model_path = Path(__file__).parent / "training_data" / "model_v2.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("Loading enhanced model v2...")
    model = EnhancedSignalModel.load(model_path)
    print(f"Model features: {model.feature_names}")
    print(f"Model metrics: {model.metrics}")

    print(f"\nFetching OOS data: {START} to {END}...")
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
    all_bars = {sym: df for sym, df in bars_by_tv.items()}

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
        benchmark="SPY",
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
    print(f"Backtest: {len(result.trades)} trades")

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")

    # Build features
    extractor = EnhancedFeatureExtractor()
    print("\nComputing rank features for OOS...")
    rank_lookup = {}
    signal_dates = sorted(set(t.signal_date for t in result.trades))
    for sig_date in signal_dates:
        ts = pd.Timestamp(sig_date)
        vals = {"rvol": [], "ret": [], "close_ema": [], "syms": []}
        for sym, bars in bars_by_tv.items():
            if bars is None or bars.empty:
                continue
            bars = bars.copy().sort_index()
            if "close" not in bars.columns or "volume" not in bars.columns:
                continue
            bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
            bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
            vol_ma5 = bars["volume"].rolling(5, min_periods=1).mean()
            rvol = bars["volume"] / vol_ma5.replace(0, np.nan)
            ret20 = bars["close"].pct_change(20)
            ema20 = bars["close"].ewm(span=20, adjust=False, min_periods=20).mean()
            close_vs_ema20 = bars["close"] / ema20.replace(0, np.nan) - 1.0
            try:
                vals["rvol"].append(rvol.loc[ts])
                vals["ret"].append(ret20.loc[ts])
                vals["close_ema"].append(close_vs_ema20.loc[ts])
                vals["syms"].append(sym)
            except KeyError:
                continue
        if vals["syms"]:
            rvol_s = pd.Series(vals["rvol"], index=vals["syms"])
            ret_s = pd.Series(vals["ret"], index=vals["syms"])
            ce_s = pd.Series(vals["close_ema"], index=vals["syms"])
            rank_rvol = rvol_s.rank(pct=True)
            rank_ret = ret_s.rank(pct=True)
            rank_ce = ce_s.rank(pct=True)
            for sym in vals["syms"]:
                rank_lookup[(sym, sig_date)] = {
                    "rank_rvol_5d": rank_rvol.get(sym, 0.5),
                    "rank_returns_20d": rank_ret.get(sym, 0.5),
                    "rank_close_vs_ema20": rank_ce.get(sym, 0.5),
                }

    X_rows = []
    valid_trades = []
    for trade in result.trades:
        bars = bars_by_tv.get(trade.ticker)
        if bars is None or bars.empty:
            continue
        sig_ts = pd.Timestamp(trade.signal_date)
        features = extractor.extract(bars, benchmark_bars=bench_df, all_bars=all_bars, signal_date=sig_ts)
        if features.empty:
            continue
        mask = features.index <= sig_ts
        if not mask.any():
            continue
        row = features.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        rank_key = (trade.ticker, trade.signal_date)
        if rank_key in rank_lookup:
            for k, v in rank_lookup[rank_key].items():
                row[k] = v
        else:
            row["rank_rvol_5d"] = 0.5
            row["rank_returns_20d"] = 0.5
            row["rank_close_vs_ema20"] = 0.5
        X_rows.append(row)
        valid_trades.append(trade)

    if not X_rows:
        print("No valid feature rows.")
        sys.exit(1)

    X = pd.concat(X_rows, ignore_index=True)
    X = X[model.feature_names].fillna(0.0)
    print(f"Feature matrix: {X.shape}")

    y_proba = model.predict(X)

    print("\n" + "=" * 60)
    print("ENHANCED MODEL V2 OOS BACKTEST (2023-01-01 to 2024-01-01)")
    print("=" * 60)

    baseline_pnl = sum(t.pnl for t in valid_trades)
    baseline_return = sum(t.return_pct for t in valid_trades) / len(valid_trades)
    baseline_wins = sum(1 for t in valid_trades if t.return_pct > 0)

    print(f"\nBaseline (all {len(valid_trades)} trades):")
    print(f"  Total P&L: ${baseline_pnl:,.2f}")
    print(f"  Avg return: {baseline_return:.2%}")
    print(f"  Win rate: {baseline_wins}/{len(valid_trades)} = {baseline_wins/len(valid_trades):.1%}")

    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
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

    print("\n  Top-N by confidence:")
    sorted_indices = np.argsort(y_proba)[::-1]
    for n in [10, 20, 30, 50]:
        if n > len(valid_trades):
            continue
        top_n = [valid_trades[i] for i in sorted_indices[:n]]
        pnl = sum(t.pnl for t in top_n)
        avg_ret = sum(t.return_pct for t in top_n) / n
        wins = sum(1 for t in top_n if t.return_pct > 0)
        print(f"\n    Top {n}:")
        print(f"      Total P&L: ${pnl:,.2f}")
        print(f"      Avg return: {avg_ret:.2%}")
        print(f"      Win rate: {wins}/{n} = {wins/n:.1%}")
        print(f"      vs Baseline P&L: {pnl - baseline_pnl:+.2f}")

    print("\n  Calibration check:")
    actuals = [t.return_pct > 0 for t in valid_trades]
    bins = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
    for lo, hi in bins:
        mask = (y_proba >= lo) & (y_proba < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        actual_win_rate = sum(actuals[i] for i in range(len(actuals)) if mask[i]) / n_bin
        print(f"    Predicted {lo:.2f}-{hi:.2f}: {n_bin} trades, actual win rate {actual_win_rate:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
