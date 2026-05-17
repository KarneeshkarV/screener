"""Generate LARGE training data for ML v5.

Expands universe to 200 US + 200 India tickers, date range 2018-01-01 to 2025-05-01.
Uses the SAME 6 proven strategies with identical parameters.
Target: 15,000+ trades with same signal quality as original 8,141.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from screener.backtester.data import YFinancePriceFetcher, tv_to_yf
from screener.backtester.rolling import run_rolling_backtest
from screener.backtester.models import BacktestConfig

# 200 S&P 500 liquid names (top 200 by index weight order from Wikipedia)
US_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
    "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA",
    "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP",
    "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY",
    "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW",
    "CHTR", "CVX", "CMG", "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO",
    "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL",
    "CMCSA", "FIX", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW",
    "CPAY", "CTVA", "CSGP", "COST", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS",
    "DHR", "DRI", "DDOG", "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM",
    "FANG", "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI",
    "DTE", "DUK", "DD", "ETN", "EBAY", "SATS", "ECL", "EIX", "EW", "EA",
    "ELV", "EME", "EMR", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR",
    "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD",
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
]

# 200 NIFTY 200 constituents (cleaned - removed dummy VEDL placeholders)
INDIA_TICKERS = [
    "360ONE", "ABB", "APLAPOLLO", "AUBANK", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ATGL", "ABCAPITAL", "ALKEM", "AMBUJACEM", "APOLLOHOSP",
    "ASHOKLEY", "ASIANPAINT", "ASTRAL", "AUROPHARMA", "DMART", "AXISBANK", "BSE",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "BANKINDIA",
    "BDL", "BEL", "BHARATFORG", "BHEL", "BPCL", "BHARTIARTL", "GROWW", "BIOCON",
    "BLUESTARCO", "BOSCHLTD", "BRITANNIA", "CGPOWER", "CANBK", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CUMMINSIND",
    "DLF", "DABUR", "DIVISLAB", "DIXON", "DRREDDY", "EICHERMOT", "ETERNAL", "EXIDEIND",
    "NYKAA", "FEDERALBNK", "FORTIS", "GAIL", "GVT&D", "GMRAIRPORT", "GLENMARK",
    "GODFRYPHLP", "GODREJCP", "GODREJPROP", "GRASIM", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDPETRO", "HINDUNILVR",
    "HINDZINC", "POWERINDIA", "HUDCO", "HYUNDAI", "ICICIBANK", "ICICIGI", "ICICIAMC",
    "IDFCFIRSTB", "ITC", "INDIANB", "INDHOTEL", "IOC", "IRCTC", "IRFC", "IREDA",
    "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "JSWENERGY", "JSWSTEEL",
    "JINDALSTEL", "JIOFIN", "JUBLFOOD", "KEI", "KPITTECH", "KALYANKJIL", "KOTAKBANK",
    "LTF", "LGEINDIA", "LICHSGFIN", "LTM", "LT", "LAURUSLABS", "LENSKART", "LODHA",
    "LUPIN", "MRF", "M&MFIN", "M&M", "MANKIND", "MARICO", "MARUTI", "MFSL", "MAXHEALTH",
    "MAZDOCK", "MOTILALOFS", "MPHASIS", "MCX", "MUTHOOTFIN", "NHPC", "NMDC", "NTPC",
    "NATIONALUM", "NESTLEIND", "OBEROIRLTY", "ONGC", "OIL", "PAYTM", "OFSS", "POLICYBZR",
    "PIIND", "PAGEIND", "PATANJALI", "PERSISTENT", "PHOENIXLTD", "PIDILITIND", "POLYCAB",
    "PFC", "POWERGRID", "PREMIERENE", "PRESTIGE", "PNB", "RECLTD", "RADICO", "RVNL",
    "RELIANCE", "SBICARD", "SBILIFE", "SRF", "MOTHERSON", "SHREECEM", "SHRIRAMFIN",
    "ENRIN", "SIEMENS", "SOLARINDS", "SBIN", "SAIL", "SUNPHARMA", "SUPREMEIND", "SUZLON",
    "SWIGGY", "TVSMOTOR", "TATACAP", "TATACOMM", "TCS", "TATACONSUM", "TATAELXSI",
    "TATAINVEST", "TMCV", "TMPV", "TATAPOWER", "TATASTEEL", "TECHM", "TITAN", "TORNTPHARM",
    "TRENT", "TIINDIA", "UPL", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "VBL", "VEDL",
    "VMM", "IDEA", "VOLTAS", "WIPRO", "TECHNOE", "ZOMATO", "ZYDUSWELL",
]

# Use first 200 valid India tickers
INDIA_TICKERS = INDIA_TICKERS[:200]

# SAME 6 proven strategies with IDENTICAL parameters
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
        print(f"    WARNING: only {len(bars_by_tv)} symbols have data, skipping")
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
    out_dir = Path(__file__).parent / "training_data_v5_large"
    out_dir.mkdir(exist_ok=True)

    all_trades = []
    all_bars = {}
    all_benchmarks = {}

    fetcher = YFinancePriceFetcher()
    start = date(2018, 1, 1)
    end = date(2025, 5, 1)

    for market, tickers in [("us", US_TICKERS), ("india", INDIA_TICKERS)]:
        print(f"\n=== {market.upper()} ({len(tickers)} tickers) ===")
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
                all_bars[key] = df

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

    # Print trades per strategy
    from collections import Counter
    strat_counts = Counter(t["strategy"] for t in all_trades)
    print("\nTrades per strategy:")
    for s, c in strat_counts.most_common():
        print(f"  {s}: {c}")


if __name__ == "__main__":
    main()
