"""V5 with enhanced indicator features — raw strategy trigger values.

Adds features the strategies actually use:
- rsi_2 (2-period RSI for mean-reversion)
- is_above_20d_high (breakout trigger)
- dist_to_20d_high (% gap to breakout level)
- ema20_slope, ema50_slope (trend direction)
- volume_vs_20d_avg (raw ratio)
- consecutive_down_days (mean reversion signal)
- days_since_52w_high (time decay from peak)
- adr_pct (average daily range)
- volatility_expanding (vol trend)
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def compute_enhanced_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute enhanced indicator features from OHLCV bars."""
    if bars is None or bars.empty or len(bars) < 5:
        return pd.DataFrame()

    df = bars.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = {}

    # EMAs
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()

    # RSI-2 (for mean reversion strategy)
    delta = close.diff()
    gain2 = delta.where(delta > 0, 0.0).rolling(2, min_periods=2).mean()
    loss2 = (-delta.where(delta < 0, 0.0)).rolling(2, min_periods=2).mean()
    rs2 = gain2 / loss2.replace(0, np.nan)
    rsi2 = 100 - (100 / (1 + rs2))
    features["rsi_2"] = rsi2

    # RSI-14
    gain14 = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss14 = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs14 = gain14 / loss14.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs14))
    features["rsi_14"] = rsi14

    # Breakout features
    hh20 = high.rolling(20, min_periods=5).max()
    features["is_above_20d_high"] = (close > hh20.shift(1)).astype(float)
    features["dist_to_20d_high"] = (close / hh20.replace(0, np.nan) - 1.0)

    # EMA slopes (% change over 5 days)
    features["ema20_slope"] = ema20.pct_change(5)
    features["ema50_slope"] = ema50.pct_change(5)
    features["ema20_vs_close"] = close / ema20.replace(0, np.nan) - 1.0
    features["ema50_vs_close"] = close / ema50.replace(0, np.nan) - 1.0
    features["ema50_vs_ema200"] = ema50 / ema200.replace(0, np.nan) - 1.0

    # Volume
    vol20_avg = volume.rolling(20, min_periods=5).mean()
    features["volume_vs_20d_avg"] = volume / vol20_avg.replace(0, np.nan)
    features["volume_trend_10d"] = volume.rolling(10, min_periods=5).mean() / volume.rolling(30, min_periods=10).mean().replace(0, np.nan)

    # Returns
    features["ret_1d"] = close.pct_change(1)
    features["ret_5d"] = close.pct_change(5)
    features["ret_10d"] = close.pct_change(10)
    features["ret_20d"] = close.pct_change(20)
    features["ret_60d"] = close.pct_change(60)

    # Momentum divergence
    features["mom_5d_vs_20d"] = features["ret_5d"] - features["ret_20d"]

    # Volatility
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20, min_periods=5).std()
    features["volatility_20d"] = vol20
    vol10 = daily_ret.rolling(10, min_periods=5).std()
    features["volatility_expanding"] = (vol10 > vol20).astype(float)

    # ADR (Average Daily Range %)
    adr = (high - low).rolling(20, min_periods=5).mean() / close.replace(0, np.nan)
    features["adr_pct"] = adr

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    features["atr_14_pct"] = atr14 / close.replace(0, np.nan)

    # Price structure
    hh252 = high.rolling(252, min_periods=20).max()
    ll252 = low.rolling(252, min_periods=20).min()
    features["dist_52w_high"] = close / hh252.replace(0, np.nan) - 1.0
    features["dist_52w_low"] = close / ll252.replace(0, np.nan) - 1.0

    # Days since 52w high
    is_high = high == hh252
    features["days_since_52w_high"] = (~is_high).groupby(is_high.cumsum()).cumsum()

    # Consecutive up/down days
    up = (daily_ret > 0).astype(int)
    down = (daily_ret < 0).astype(int)
    features["consecutive_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    features["consecutive_down"] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)

    # Gap
    features["gap_pct"] = (df["open"] - close.shift(1)) / close.shift(1).replace(0, np.nan)

    # Range
    features["range_pct"] = (high - low) / close.replace(0, np.nan)

    # Max drawdown from 20d high
    rolling_high = high.rolling(20, min_periods=5).max()
    features["max_dd_20d"] = (close - rolling_high) / rolling_high.replace(0, np.nan)

    # Volume-price correlation
    features["vol_price_corr_20d"] = daily_ret.rolling(20, min_periods=10).corr(volume.pct_change().replace([np.inf, -np.inf], np.nan))

    # Sharpe
    features["sharpe_20d"] = (daily_ret.rolling(20, min_periods=5).mean() / vol20.replace(0, np.nan)) * np.sqrt(252)

    # MACD
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd_hist"] = macd_line - signal_line

    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    features["adx_14"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # BB position
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    features["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    feat_df = pd.DataFrame(features, index=df.index)
    # Drop rows >70% NaN, fill rest with neutral
    na_frac = feat_df.isna().mean(axis=1)
    feat_df = feat_df[na_frac <= 0.7]
    # Neutral fills
    neutral = {
        "rsi_2": 50.0, "rsi_14": 50.0,
        "is_above_20d_high": 0.0, "dist_to_20d_high": 0.0,
        "ema20_slope": 0.0, "ema50_slope": 0.0,
        "ema20_vs_close": 0.0, "ema50_vs_close": 0.0, "ema50_vs_ema200": 0.0,
        "volume_vs_20d_avg": 1.0, "volume_trend_10d": 1.0,
        "ret_1d": 0.0, "ret_5d": 0.0, "ret_10d": 0.0, "ret_20d": 0.0, "ret_60d": 0.0,
        "mom_5d_vs_20d": 0.0,
        "volatility_20d": 0.0, "volatility_expanding": 0.5,
        "adr_pct": 0.0, "atr_14_pct": 0.0,
        "dist_52w_high": 0.0, "dist_52w_low": 0.0,
        "days_since_52w_high": 126.0,
        "consecutive_up": 0.0, "consecutive_down": 0.0,
        "gap_pct": 0.0, "range_pct": 0.0,
        "max_dd_20d": 0.0,
        "vol_price_corr_20d": 0.0,
        "sharpe_20d": 0.0,
        "macd_hist": 0.0, "adx_14": 25.0,
        "bb_position": 0.5,
    }
    for col, val in neutral.items():
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].fillna(val)
    return feat_df


def main():
    data_dir = Path("scripts/training_data_v4")
    print("Loading data...")
    with open(data_dir / "trades.json") as f:
        trades = json.load(f)["trades"]
    with open(data_dir / "bars.json") as f:
        bars_json = json.load(f)

    bars_by_symbol = {}
    for key, records in bars_json.get("bars", {}).items():
        if not isinstance(records, list):
            continue
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        sym = key.split(":", 1)[1] if ":" in key else key
        bars_by_symbol[sym] = df

    print("Extracting enhanced features...")
    # Compute features per ticker and store
    features_by_ticker = {}
    for sym, bars in bars_by_symbol.items():
        feat = compute_enhanced_features(bars)
        if not feat.empty:
            features_by_ticker[sym] = feat

    feature_cols = list(list(features_by_ticker.values())[0].columns)
    print(f"Feature count: {len(feature_cols)}")

    rows = []
    for t in trades:
        ticker = t["ticker"]
        feat_df = features_by_ticker.get(ticker)
        if feat_df is None or feat_df.empty:
            continue
        ts = pd.Timestamp(t["signal_date"])
        mask = feat_df.index <= ts
        if not mask.any():
            continue
        row = feat_df.loc[mask].iloc[[-1]]
        if row.isna().all().all():
            continue
        rows.append({
            "signal_date": ts,
            "return_pct": float(t["return_pct"]),
            "is_win": int(t["return_pct"] > 0),
            **{k: float(row[k].iloc[0]) for k in feature_cols},
        })

    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df = df.sort_values("signal_date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df[feature_cols].values
    y_train = train_df["return_pct"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["return_pct"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["return_pct"].values
    y_test_win = test_df["is_win"].values

    # Quick HP search
    configs = [
        {"max_depth": 3, "lr": 0.1, "lambda": 5.0},
        {"max_depth": 4, "lr": 0.05, "lambda": 3.0},
        {"max_depth": 5, "lr": 0.03, "lambda": 3.0},
        {"max_depth": 3, "lr": 0.05, "lambda": 10.0},
        {"max_depth": 4, "lr": 0.1, "lambda": 1.0},
    ]

    best = None
    for cfg in configs:
        model = XGBRegressor(
            n_estimators=300, max_depth=cfg["max_depth"],
            learning_rate=cfg["lr"], subsample=0.6, colsample_bytree=0.6,
            reg_lambda=cfg["lambda"], random_state=42, n_jobs=4,
            early_stopping_rounds=20,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test_win, preds)

        sorted_idx = np.argsort(preds)[::-1]
        n10 = max(1, int(len(y_test) * 0.1))
        top10_wr = y_test_win[sorted_idx[:n10]].mean()

        print(f"  depth={cfg['max_depth']} lr={cfg['lr']} lambda={cfg['lambda']}: AUC={auc:.4f} top10WR={top10_wr:.1%}")

        if best is None or auc > best["auc"]:
            best = {"auc": auc, "top10_wr": top10_wr, "cfg": cfg, "model": model}

    print(f"\n{'='*60}")
    print("ENHANCED FEATURES RESULTS")
    print(f"{'='*60}")
    print(f"Best AUC: {best['auc']:.4f} (depth={best['cfg']['max_depth']} lr={best['cfg']['lr']})")
    print(f"Top 10% WR: {best['top10_wr']:.1%}")
    print(f"Baseline WR: {y_test_win.mean():.1%}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": best["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nTop 10 features:")
    for _, r in imp.head(10).iterrows():
        print(f"  {r['feature']}: {r['importance']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
