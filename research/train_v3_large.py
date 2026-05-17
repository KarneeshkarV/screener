"""Train v3 model on 3 years of data with 500 stocks."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig
from screener.ml_signal_v3 import SimpleSignalModel


def load_sp500() -> list[str]:
    """Return S&P 500 tickers from Wikipedia or a cached list."""
    try:
        import requests
        from io import StringIO
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, timeout=30).text
        df = pd.read_html(StringIO(html))[0]
        tickers = df["Symbol"].tolist()
        print(f"Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"Could not fetch S&P 500: {e}")
        # Fallback: top 500 liquid US stocks
        return [
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
            "EMR", "NSC", "OXY", "DG", "GM", "PSA", "NXPI", "KLAC", "DXCM", "MCK",
            "AEP", "CDNS", "MPC", "SRE", "MET", "USB", "MAR", "STZ", "FCX", "F",
            "ECL", "COF", "AZO", "TWTR", "SNPS", "ROP", "PXD", "KMB", "PSX", "AIG",
            "TRV", "D", "O", "DUK", "TFC", "SPG", "MSCI", "AFL", "ADM", "PAYX",
            "EXC", "CCI", "CTSH", "ROST", "VLO", "IQV", "WELL", "KR", "PRU", "ED",
            "LEN", "HCA", "IDXX", "WEC", "WMB", "NEM", "FAST", "AMP", "ALL", "PCAR",
            "KMI", "OKE", "DAL", "HPQ", "FTNT", "STT", "GLW", "COP", "VRSK", "EBAY",
            "NUE", "PPG", "TSCO", "XEL", "DLTR", "ANSS", "KEYS", "ES", "HES", "TROW",
            "EXPE", "WBA", "DRI", "LYB", "MTB", "HAL", "ROK", "BK", "WDC", "RF",
            "VMC", "DFS", "BBY", "AVB", "HPE", "DVN", "RJF", "FTV", "CDW", "CF",
            "WY", "ULTA", "TAP", "BEN", "NVR", "GPN", "CAG", "RCL", "APA", "EQT",
            "BALL", "UAL", "PWR", "PKG", "CTRA", "TPR", "AOS", "ALLE", "EPAM", "MAS",
            "HRL", "SWKS", "DOV", "IP", "TYL", "IRM", "IR", "J", "TXT", "JKHY",
            "AKAM", "LNT", "ROL", "INCY", "LUV", "PFG", "CPT", "AES", "FE", "HST",
            "BRO", "DGX", "PPL", "CINF", "NDSN", "K", "KMX", "GRMN", "JBHT", "CLX",
            "CNP", "TRGP", "ESS", "MRO", "POOL", "SWK", "STLD", "SJM", "IPG", "CBOE",
            "NWSA", "SNA", "UHS", "PKI", "CRL", "CPB", "RHI", "WRB", "LW", "DAY",
            "NDAQ", "LDOS", "TECH", "FDS", "CCL", "EMN", "CSGP", "GNRC", "TPR", "ARES",
            "BR", "CE", "IPGP", "RVTY", "NCLH", "FFIV", "TRMB", "WYNN", "RE", "OGN",
            "CPRT", "ALLE", "MOH", "PENN", "CHRW", "ETSY", "GWW", "PEAK", "GL", "BXP",
            "PARA", "MTD", "BF-B", "MKC", "HAS", "CPAY", "JNPR", "AAL", "BALL", "QRVO",
            "MGM", "WAB", "WYNN", "IVZ", "FOXA", "NCLH", "LVS", "APA", "CZR", "NWSA",
        ]


MARKET = "us"
START = date(2022, 1, 1)
END = date(2025, 1, 1)


def main() -> None:
    tickers = load_sp500()
    print(f"\nTraining v3 model: {len(tickers)} stocks, {START} to {END}")
    print("=" * 60)

    print("\nFetching price data (this may take a few minutes)...")
    fetcher = YFinancePriceFetcher()
    start_fetch = START - timedelta(days=90)
    end_fetch = END + timedelta(days=1)

    yf_symbols = [tv_to_yf(t, MARKET) for t in tickers]
    price_panel = fetcher.fetch(yf_symbols, start_fetch, end_fetch)

    bars_by_tv = {}
    for tv_sym in tickers:
        yf_sym = tv_to_yf(tv_sym, MARKET)
        df = price_panel.get(yf_sym)
        if df is not None and not df.empty:
            bars_by_tv[tv_sym] = df

    print(f"Fetched {len(bars_by_tv)} symbols successfully")

    bench_df = price_panel.get("SPY")

    cfg = BacktestConfig(
        market=MARKET,
        as_of=END,
        hold=20,
        top=20,
        entry_expr="close > ema(close, 20) and ema(close, 20) > ema(close, 200)",
        exit_expr="crossunder(close, ema(close, 20))",
        stop_loss=0.07,
        take_profit=0.15,
        trailing_stop=0.05,
        slippage_bps=5.0,
        commission_bps=10.0,
        initial_capital=1_000_000.0,
        benchmark="SPY",
        tickers=tuple(tickers),
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

    print("\nRunning rolling backtest...")
    result = run_rolling_backtest(cfg, fetcher, start_date=START, end_date=END)
    print(f"Backtest complete: {len(result.trades)} trades")

    if len(result.trades) < 100:
        print("WARNING: Not enough trades for training.")
        sys.exit(1)

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    print(f"Baseline win rate: {wins}/{len(result.trades)} = {wins/len(result.trades):.1%}")

    print("\nTraining v3 model...")
    model = SimpleSignalModel()
    model.train(trades=result.trades, bars_by_symbol=bars_by_tv, benchmark_bars=bench_df)

    print(f"\nModel metrics:")
    for k, v in (model.metrics or {}).items():
        print(f"  {k}: {v}")

    print("\nFeature importance:")
    fi = model.feature_importance()
    for _, row in fi.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    output_dir = Path(__file__).parent / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(output_dir / "model_v3_large.pkl")
    print(f"\nModel saved to {output_dir / 'model_v3_large.pkl'}")

    # Also save trades and bars for later analysis
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
        for t in result.trades
    ]

    with open(output_dir / "trades_large.json", "w") as f:
        json.dump({"trades": trades_dict}, f)

    # Save a sample of bars (too big for all 500)
    bars_sample = {}
    for sym in list(bars_by_tv.keys())[:20]:
        df = bars_by_tv[sym].reset_index()
        if hasattr(df, "to_dict"):
            bars_sample[sym] = df.to_dict(orient="records")

    with open(output_dir / "bars_large.json", "w") as f:
        json.dump({"bars": bars_sample}, f)

    print(f"Saved {len(trades_dict)} trades to trades_large.json")


if __name__ == "__main__":
    main()
