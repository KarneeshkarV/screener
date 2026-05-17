"""Train enhanced ML model v2 with better features and larger dataset."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal_v2 import EnhancedSignalModel

# Expanded universe: top 25 liquid US stocks
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM",
    "AVGO", "LLY", "WMT", "UNH", "MA", "PG", "JNJ", "HD", "CVX", "MRK",
    "COST", "ABBV", "PEP", "KO", "ADBE",
]
MARKET = "us"
START = date(2022, 1, 1)  # 3 years of training data
END = date(2025, 1, 1)

def main() -> None:
    print(f"Training enhanced model v2")
    print(f"Universe: {len(TICKERS)} stocks")
    print(f"Period: {START} to {END}")
    print("=" * 60)

    print("\nFetching price data...")
    fetcher = YFinancePriceFetcher()
    start_fetch = START - timedelta(days=90)
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

    bench_df = price_panel.get("SPY")
    all_bars = {sym: df for sym, df in bars_by_tv.items()}

    bench = "SPY"
    cfg = BacktestConfig(
        market=MARKET,
        as_of=END,
        hold=20,
        top=10,
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

    print("\nRunning backtest for training data...")
    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    print(f"Backtest complete: {len(result.trades)} trades")

    if len(result.trades) < 50:
        print("WARNING: Not enough trades for meaningful training. Need at least 50.")
        sys.exit(1)

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")

    print("\nTraining enhanced model v2...")
    model = EnhancedSignalModel()
    model.train(
        trades=result.trades,
        bars_by_symbol=bars_by_tv,
        benchmark_bars=bench_df,
        all_bars=all_bars,
    )

    print(f"\nModel metrics:")
    for k, v in (model.metrics or {}).items():
        print(f"  {k}: {v}")

    print("\nTop 10 features by importance:")
    fi = model.feature_importance().head(10)
    for _, row in fi.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    output_path = Path(__file__).parent / "training_data" / "model_v2.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

if __name__ == "__main__":
    main()
