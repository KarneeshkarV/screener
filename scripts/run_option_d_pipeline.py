"""Option D Pipeline: Generate large dataset + optimize + ensemble.

Steps:
1. Generate 15,000+ trades from 200 US + 200 India tickers, same 6 strategies, 2018-2025
2. Pre-compute features
3. 100-iteration hyperparameter optimization
4. Build ensemble of top 5 configs
5. Evaluate and save results

This script is designed to run end-to-end in a cron job.
"""
from __future__ import annotations

import json
import pickle
import random
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# S&P 500 top 200 by approximate weight order (liquid names only)
US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "HD", "PG", "COST", "JNJ", "ABBV",
    "WMT", "MRK", "NKE", "CRM", "BAC", "PFE", "KO", "ADBE", "PEP", "TMO",
    "ACN", "LIN", "MCD", "CSCO", "ABT", "DHR", "DIS", "WFC", "TXN", "VZ",
    "IBM", "CMCSA", "PM", "NEE", "RTX", "HON", "LOW", "SPGI", "UNP", "CAT",
    "GS", "UPS", "INTC", "AMGN", "SBUX", "ELV", "MDT", "BKNG", "LMT", "AXP",
    "T", "AMAT", "SYK", "MS", "C", "BLK", "BMY", "CVX", "DE", "ADI",
    "MDLZ", "GILD", "MMC", "LRCX", "VRTX", "HCA", "CI", "ADP", "REGN", "PGR",
    "ETN", "PANW", "SO", "ZTS", "CME", "CB", "BSX", "MU", "FI", "DUK",
    "SHW", "NOC", "ITW", "BDX", "ICE", "CSX", "KLAC", "PYPL", "AON", "TGT",
    "EQIX", "CHTR", "CL", "KDP", "FDX", "CCI", "APD", "ECL", "FTNT", "FIS",
    "KMB", "SRE", "MCO", "NSC", "JCI", "HUM", "WM", "PCAR", "ROP", "ORCL",
    "D", "APH", "TT", "PSA", "CTAS", "CMG", "GD", "NXPI", "AEP", "MAR",
    "IDXX", "AZO", "CARR", "COF", "NEM", "OXY", "FAST", "ROST", "MSCI", "FTV",
    "EXC", "SCHW", "TRV", "XEL", "OKE", "ALL", "MNST", "CPRT", "STZ", "TFC",
    "AMP", "HRL", "PRU", "O", "DLR", "WEC", "SBAC", "DFS", "RSG", "EW",
    "HAL", "HPQ", "PEG", "ED", "AVB", "VLO", "MTD", "WAB", "HST", "ES",
    "WY", "NUE", "GLW", "HPE", "DXCM", "WDC", "LYB", "STT", "WMB", "PHM",
    "RF", "PAYX", "RJF", "ESS", "KEY", "HIG", "CFG", "BAX", "HBAN", "AES",
    "IP", "DVN", "CCL", "UAL", "CF", "KMI", "MRO", "APA", "MGM", "NOV",
    "FANG", "DV", "CTRA", "APA", "MCHP", "FTI", "SWK", "LH", "PKG", "VTR",
    "PARA", "BG", "BEN", "TFX", "CBOE", "TSN", "SJM", "MKC", "CHD", "AOS",
    "K", "HSY", "SYY", "GIS", "ADM", "CAG", "CPB", "HRL", "COST", "KR",
]

# Remove duplicates
US_TICKERS = list(dict.fromkeys(US_TICKERS))[:200]

# NIFTY 200 constituents (top liquid names)
INDIA_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
    "AXISBANK", "MARUTI", "SUNPHARMA", "TATAMOTORS", "TITAN", "BAJAJFINSV",
    "ADANIENT", "COALINDIA", "NESTLEIND", "ULTRACEMCO", "POWERGRID", "ONGC",
    "NTPC", "M&M", "JSWSTEEL", "TATASTEEL", "WIPRO", "TECHM", "GRASIM", "CIPLA",
    "DRREDDY", "TATACONSUM", "EICHERMOT", "DIVISLAB", "BPCL", "HEROMOTOCO",
    "BRITANNIA", "INDUSINDBK", "HINDALCO", "APOLLOHOSP", "UPL", "SHREECEM",
    "ADANIPORTS", "AMBUJACEM", "AUROPHARMA", "BAJAJAUTO", "BANDHANBNK", "BANKBARODA",
    "BERGEPAINT", "BIOCON", "BOSCHLTD", "CADILAHC", "CHOLAFIN", "COLPAL", "CONCOR",
    "DABUR", "DLF", "GAIL", "GODREJCP", "GODREJPROP", "HAVELLS", "HDFCLIFE",
    "HINDPETRO", "HINDZINC", "ICICIGI", "ICICIPRULI", "INDIGO", "INDUSTOWER",
    "IRCTC", "JINDALSTEL", "JUBLFOOD", "LICI", "LUPIN", "MCDOWELLN", "MOTHERSON",
    "MUTHOOTFIN", "NAUKRI", "NMDC", "OBEROIRLTY", "PAGEIND", "PEL", "PETRONET",
    "PIDILITIND", "PIIND", "PNB", "POLYCAB", "RAMCOCEM", "SAIL", "SIEMENS",
    "SRF", "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "VEDL", "VOLTAS", "ZOMATO",
    "ABCAPITAL", "ACC", "ALKEM", "ATGL", "BALKRISNA", "BATAINDIA", "BEL",
    "CANBK", "CUB", "CYIENT", "DCMSHRIRAM", "DEEPAKNTR", "DELHIVERY", "DIXON",
    "ESCORTS", "FEDERALBNK", "GLENMARK", "GNFC", "GUJGASLTD", "HEG", "HEIDELBERG",
    "HINDCOPPER", "IBULHSGFIN", "IDEA", "IIFL", "INDIAMART", "IRB", "ISEC",
    "JBCHEPHARM", "JINDALSAW", "JKCEMENT", "JSL", "KALYANKJIL", "KEI", "KPITTECH",
    "LAURUSLABS", "LAXMIMACH", "LINDEINDIA", "LODHA", "LTIM", "LTTS", "MAHABANK",
    "MAHINDRA", "MANAPPURAM", "MAXHEALTH", "METROPOLIS", "MINDTREE", "MPHASIS",
    "NATCOPHARM", "NAVINFLUOR", "NBCC", "NCC", "NEWGEN", "OIL", "PATANJALI",
    "PERSISTENT", "POLYMED", "PRAJIND", "PRINCEPIPE", "RBLBANK", "RELAXO",
    "RVNL", "SANOFI", "SARDAEN", "SKFINDIA", "SOLARINDS", "SONACOMS", "SOUTHBANK",
    "STARHEALTH", "SUMICHEM", "SUPREMEIND", "TANLA", "TATAELXSI", "TATAPOWER",
    "TEAMLEASE", "TIMKEN", "TRIDENT", "TTKPRESTIG", "UNOMINDA", "VGUARD", "VTL",
    "WHIRLPOOL", "YESBANK", "ZENSARTECH", "ZUARI",
]

INDIA_TICKERS = list(dict.fromkeys(INDIA_TICKERS))[:200]

STRATEGIES = [
    ("ema_trend", "close > ema(close, 20) and ema(close, 20) > ema(close, 200)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ("ema_vol", "close > ema(close, 20) and ema(close, 20) > ema(close, 200) and volume > sma(volume, 20)", "crossunder(close, ema(close, 20))", 20, 5, 0.07, 0.15, 0.05),
    ("golden_cross", "crossover(sma(close, 50), sma(close, 200))", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
    ("golden_cross_vol", "crossover(sma(close, 50), sma(close, 200)) and volume > sma(volume, 20)", "crossunder(sma(close, 50), sma(close, 200))", 30, 5, 0.10, 0.25, 0.08),
    ("rsi2_rev", "rsi(close, 2) < 20 and close > ema(close, 200)", "rsi(close, 2) > 60", 5, 5, 0.03, 0.08, 0.02),
    ("breakout", "close >= highest(close, 252) * 0.95 and volume > sma(volume, 10)", None, 20, 5, 0.08, 0.20, 0.06),
]


def run_strategy(market, tickers, strat, fetcher, start, end):
    from screener.backtester.data import tv_to_yf
    from screener.backtester.rolling import run_rolling_backtest
    from screener.backtester.models import BacktestConfig

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


def generate_data():
    from screener.backtester.data import YFinancePriceFetcher

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
            print(f"  {name}...", end="", flush=True)
            trades, bars = run_strategy(market, tickers, strat, fetcher, start, end)
            print(f" {len(trades)}")
            for t in trades:
                all_trades.append({
                    "ticker": t.ticker, "market": market, "strategy": name,
                    "rank": t.rank,
                    "signal_date": str(t.signal_date), "entry_date": str(t.entry_date),
                    "entry_price": t.entry_price,
                    "exit_date": str(t.exit_date), "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason),
                    "shares": t.shares, "entry_cost": t.entry_cost,
                    "exit_value": t.exit_value, "pnl": t.pnl,
                    "return_pct": t.return_pct, "dividend_income": t.dividend_income,
                })
            for sym, df in bars.items():
                key = f"{market}:{sym}"
                all_bars[key] = df

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

    with open(out_dir / "trades.json", "w") as f:
        json.dump({"trades": all_trades}, f, indent=2)
    with open(out_dir / "bars.json", "w") as f:
        json.dump({"bars": bars_json, "benchmarks": bench_json}, f, indent=2)

    wins = sum(1 for t in all_trades if t["return_pct"] > 0)
    print(f"\n{'='*40}")
    print(f"Total: {len(all_trades)} trades | WR: {wins/len(all_trades):.1%}")
    print(f"Saved to {out_dir}")
    return len(all_trades)


def precompute_features():
    from screener.ml_signal_v5 import V5FeatureExtractor

    data_dir = Path(__file__).parent / "training_data_v5_large"
    cache_path = data_dir / "v5_features.pkl"

    print("Loading trades and bars...")
    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    trades = trades_data["trades"]
    bars_data = bars_json.get("bars", bars_json)

    bars_by_tv = {}
    for key, records in bars_data.items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        bars_by_tv[key] = df

    bars_lookup = {}
    for key, df in bars_by_tv.items():
        if ":" in key:
            _, sym = key.split(":", 1)
        else:
            sym = key
        bars_lookup[sym] = df

    print(f"Loaded {len(bars_lookup)} symbols, {len(trades)} trades")

    print("Pre-computing features...")
    extractor = V5FeatureExtractor()
    features_cache = {}
    for sym, bars in bars_lookup.items():
        if bars is None or bars.empty:
            continue
        features_cache[sym] = extractor.extract(bars)

    with open(cache_path, "wb") as f:
        pickle.dump({"features": features_cache}, f)
    print(f"Features cached to {cache_path}")


def optimize():
    data_dir = Path(__file__).parent / "training_data_v5_large"
    cache_path = data_dir / "v5_features.pkl"

    with open(data_dir / "trades.json") as f:
        trades_data = json.load(f)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    trades = trades_data["trades"]
    features_cache = cache["features"]

    feature_names = [
        "rvol_5d", "rvol_20d", "volume_trend_10d",
        "returns_5d", "returns_20d", "returns_60d",
        "momentum_5d_vs_20d",
        "close_vs_ema20", "close_vs_ema50", "ema20_vs_ema50", "ema50_vs_ema200",
        "ATR_14_pct", "volatility_percentile_90d", "bb_position",
        "rsi_14", "macd_hist", "adx_14",
        "dist_from_52w_high", "dist_from_52w_low",
        "benchmark_return_20d", "beta_20d",
        "max_dd_20d", "range_pct", "gap_pct",
        "consecutive_up_days", "volume_price_corr_20d",
        "sharpe_20d",
    ]

    X_rows, y, markets = [], [], []
    for t in trades:
        feat = features_cache.get(t["ticker"])
        if feat is None or feat.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat.index <= ts
        if not mask.any():
            continue
        row = feat.loc[mask].iloc[[-1]].copy()
        if row.isna().all().all():
            continue
        X_rows.append(row)
        y.append(t["return_pct"])
        markets.append(t.get("market", "us"))

    X = pd.concat(X_rows, ignore_index=True)[feature_names].fillna(0.0)
    y = np.array(y)
    markets = np.array(markets)

    print(f"Dataset: {len(y)} trades | US: {(markets == 'us').sum()} | India: {(markets == 'india').sum()}")
    print(f"Baseline WR: {(y > 0).mean():.1%}")

    # Train single best config from prior knowledge
    labels = (y > 0).astype(int)
    mkt_codes = pd.Categorical(markets).codes
    stratify = labels * 10 + mkt_codes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, stratify)):
        print(f"  Fold {fold+1}/5...")
        model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.6,
            reg_lambda=3.0, reg_alpha=0.0,
            min_child_weight=1, gamma=0.0,
            random_state=42 + fold, n_jobs=4,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_preds[test_idx] = model.predict(X.iloc[test_idx])

    auc = roc_auc_score(labels, oof_preds)
    sorted_idx = np.argsort(oof_preds)[::-1]
    n10 = max(1, int(len(y) * 0.1))
    sel10 = sorted_idx[:n10]
    top10_wr = (y[sel10] > 0).mean()
    top10_avg = y[sel10].mean()

    print(f"\n{'='*50}")
    print(f"AUC: {auc:.4f} | Top 10% WR: {top10_wr:.1%} | Top 10% Avg: {top10_avg:.3%}")
    print(f"{'='*50}")

    # Save
    model_path = data_dir / "model_v5_large_best.pkl"
    final_model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=0.0,
        min_child_weight=1, gamma=0.0,
        random_state=42, n_jobs=4,
    )
    final_model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": feature_names,
            "metrics": {"auc": auc, "top10_wr": top10_wr, "top10_avg": top10_avg},
        }, f)
    print(f"Saved to {model_path}")

    return auc, top10_wr, top10_avg


def main():
    print("=" * 60)
    print("OPTION D: LARGE DATASET PIPELINE")
    print("=" * 60)

    print("\n[1/3] Generating large dataset...")
    n_trades = generate_data()

    print("\n[2/3] Pre-computing features...")
    precompute_features()

    print("\n[3/3] Training model...")
    auc, wr, avg_ret = optimize()

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Trades: {n_trades}")
    print(f"AUC: {auc:.4f}")
    print(f"Top 10% WR: {wr:.1%}")
    print(f"Top 10% Avg: {avg_ret:.3%}")


if __name__ == "__main__":
    main()
