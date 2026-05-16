"""Generate backtest trades + bars JSON for ML model training."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig

# Small liquid US universe
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "XOM"]
MARKET = "us"
START = date(2024, 1, 1)
END = date(2025, 1, 1)

def main() -> None:
    print("Fetching price data...")
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
        print("WARNING: Not enough trades for meaningful training. Need at least 10.")
        sys.exit(1)

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")

    # Convert trades to dicts
    trades_data = {
        "trades": [
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
            for t in result.trades
        ],
        "initial_capital": cfg.initial_capital,
        "calendar": [str(d) for d in result.equity_curve.index],
    }

    # Convert bars to JSON-serializable format
    bars_json = {"bars": {}}
    for sym, df in bars_by_tv.items():
        bars_json["bars"][sym] = [
            {
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for idx, row in df.iterrows()
        ]

    # Benchmark bars
    bench_df = fetcher.fetch([bench], start_fetch, end_fetch).get(bench, None)
    if bench_df is not None and not bench_df.empty:
        bars_json["bars"]["benchmark"] = [
            {
                "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for idx, row in bench_df.iterrows()
        ]

    out_dir = Path("scripts/training_data")
    out_dir.mkdir(exist_ok=True)
    trades_path = out_dir / "trades.json"
    bars_path = out_dir / "bars.json"

    trades_path.write_text(json.dumps(trades_data, indent=2))
    bars_path.write_text(json.dumps(bars_json, indent=2))

    print(f"Wrote {trades_path}")
    print(f"Wrote {bars_path}")
    print("Done. Now run:")
    print(f"  python main.py train-model --trades {trades_path} --bars {bars_path} --output scripts/training_data/model.pkl")

if __name__ == "__main__":
    main()
