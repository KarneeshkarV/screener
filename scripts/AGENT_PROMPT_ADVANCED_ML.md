# Agent Prompt: Advanced ML Model for Screener Signal Confidence

## Context

You are working on a stock screener project. We already have a working v5 model (XGBoost regression) that predicts expected return per trade signal. We ran a 100-iteration hyperparameter optimization and achieved:

- **AUC = 0.6346** (direction prediction)
- **Top 10% win rate = 66.0%** (baseline = 33.6%)
- **Top 10% avg return = +4.22%**

The repo is at `/home/hermes/screener-wt/ml-signal` (branch `feat/alpha-strategy-research`).

## Your Mission

Push the model performance further by building and evaluating **at least ONE** of the following approaches. The goal is to beat the current best: **AUC > 0.650** and **top 10% WR > 65%**.

## Option 1: Ensemble of Top XGBoost Configs (RECOMMENDED — lowest risk)

Train the top 5 configurations from the optimization results, then ensemble their predictions.

### Steps:
1. Load `scripts/training_data_v4/optimization_results_v5.json` to find top 5 configs
2. For each of the 5 configs, train a separate XGBRegressor on the full dataset
3. Ensemble predictions using:
   - Simple mean of predicted returns
   - Weighted mean by cross-validation AUC
   - Rank-averaging (average percentile ranks instead of raw values)
4. Evaluate: AUC, top 10%/20% win rate, top 10%/20% avg return
5. Compare against single best model

### Expected outcome:
- Low risk, should improve AUC by 0.01–0.02
- Top 10% WR should stay ~65–68%

---

## Option 2: Universal Neural Net with Stock Embeddings

Build a single neural network trained on ALL stocks together, using a stock/ticker embedding to learn stock-specific behavior.

### Steps:
1. Load cached features from `scripts/training_data_v4/v5_features.pkl`
2. Load trades from `scripts/training_data_v4/trades.json`
3. Build feature matrix (same 27 features as v5)
4. Create a **stock embedding layer**:
   - Map each ticker to an embedding vector (e.g., 8-dimensional)
   - The embedding learns "AAPL behaves like MSFT" vs "TSLA behaves differently"
5. Architecture suggestion:
```
Input: (batch_size, 27 features + 1 stock_id)
  → Stock Embedding (n_stocks × 8) → concat with features
  → Dense(128, ReLU, Dropout=0.4)
  → Dense(64, ReLU, Dropout=0.3)
  → Dense(32, ReLU, Dropout=0.2)
  → Dense(1)  # predict return_pct
```
6. Use **PyTorch** or **TensorFlow/Keras**
7. Split data with **purged group k-fold** (by date, not randomly) to prevent leakage
8. Evaluate same metrics as v5

### Critical constraints:
- **DO NOT** shuffle randomly — sort by date and use time-based splits
- **DO NOT** use signal date as a feature
- Use **early stopping** on validation loss
- Use **weight decay (L2)** = 0.001–0.01
- Use **batch size** ≥ 256 (we have 8,141 samples)
- Train for max 200 epochs, early stop at patience=15

### Expected outcome:
- Risk: may overfit due to small dataset
- Target: AUC ≥ 0.620 to be viable
- If AUC < 0.600, abandon and report why

---

## Option 3: TabNet or FT-Transformer for Tabular Data

Use a deep learning architecture specifically designed for tabular data.

### Steps:
1. Install TabNet: `pip install pytorch-tabnet`
2. Use same feature matrix as v5 (27 features)
3. Train TabNetRegressor with:
```python
TabNetRegressor(
    n_d=16, n_a=16, n_steps=3,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.01),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
)
```
4. Or try FT-Transformer via `pytorch-widedeep` or `rtdl`
5. Use the same cross-validation setup as v5 (stratified 5-fold)
6. Evaluate and compare

### Expected outcome:
- TabNet sometimes beats XGBoost on tabular data with enough regularization
- Target: AUC ≥ 0.630 to justify complexity

---

## Data Loading Reference

```python
import pickle, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# Load features cache
with open('scripts/training_data_v4/v5_features.pkl', 'rb') as f:
    cache = pickle.load(f)
features_cache = cache['features']

# Load trades
with open('scripts/training_data_v4/trades.json') as f:
    trades_data = json.load(f)
trades = trades_data['trades']

# Feature names (27 total)
FEATURE_NAMES = [
    'rvol_5d', 'rvol_20d', 'volume_trend_10d',
    'returns_5d', 'returns_20d', 'returns_60d',
    'momentum_5d_vs_20d',
    'close_vs_ema20', 'close_vs_ema50', 'ema20_vs_ema50', 'ema50_vs_ema200',
    'ATR_14_pct', 'volatility_percentile_90d', 'bb_position',
    'rsi_14', 'macd_hist', 'adx_14',
    'dist_from_52w_high', 'dist_from_52w_low',
    'benchmark_return_20d', 'beta_20d',
    'max_dd_20d', 'range_pct', 'gap_pct',
    'consecutive_up_days', 'volume_price_corr_20d',
    'sharpe_20d',
]

# Build matrix
X_rows, y, markets, tickers, dates = [], [], [], [], []
for t in trades:
    feat = features_cache.get(t['ticker'])
    if feat is None or feat.empty:
        continue
    ts = pd.Timestamp(t['signal_date'])
    mask = feat.index <= ts
    if not mask.any():
        continue
    row = feat.loc[mask].iloc[[-1]].copy()
    if row.isna().all().all():
        continue
    X_rows.append(row)
    y.append(t['return_pct'])
    markets.append(t.get('market', 'us'))
    tickers.append(t['ticker'])
    dates.append(t['signal_date'])

X = pd.concat(X_rows, ignore_index=True)[FEATURE_NAMES].fillna(0.0)
y = np.array(y)
markets = np.array(markets)
tickers = np.array(tickers)
dates = pd.to_datetime(dates)

print(f"Samples: {len(y)}")
print(f"US: {(markets=='us').sum()}, India: {(markets=='india').sum()}")
print(f"Unique tickers: {len(set(tickers))}")
print(f"Baseline WR: {(y > 0).mean():.1%}")
```

## Evaluation Protocol

For EVERY approach, report these exact metrics:

1. **Random-split 5-fold CV**:
   - `auc_mean`, `auc_std`
   - `top10_wr_mean`, `top10_avg_mean`
   - `top20_wr_mean`, `top20_avg_mean`
   - `us_auc_mean`, `india_auc_mean`

2. **Time-based split** (train on first 70% of dates, test on last 30%):
   - Same metrics as above
   - This tests temporal generalization

3. **Walk-forward** (optional but preferred):
   - Train on 6-month rolling window, predict next 1 month
   - Report average AUC across windows

## Deliverables

1. **Source code file** in `scripts/` (e.g., `scripts/ensemble_top5.py`, `scripts/nn_stock_embedding.py`, `scripts/tabnet_v5.py`)
2. **Results summary** appended to `OPTIMIZATION_V5_100_ITERATIONS.md` or new file `ADVANCED_ML_RESULTS.md`
3. **Trained model artifact** saved to `scripts/training_data_v4/` as `.pkl` or `.pt`
4. **Git commit** with clear message

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Random-split AUC | > 0.620 | > 0.650 |
| Top 10% WR | > 60% | > 65% |
| Top 10% avg return | > +3.0% | > +4.5% |
| Time-split AUC | > 0.550 | > 0.580 |

If your approach does NOT beat the v5 baseline (AUC=0.635, top10 WR=66%), clearly document why and what you learned.

## Anti-Patterns to Avoid

1. **Random shuffling** — always respect time ordering
2. **Using return_pct as an input feature** — that's the target
3. **Training on test data** — use proper CV
4. **Overfitting to CV** — if train AUC is 0.90+ but test AUC is 0.55, you've overfit
5. **Ignoring class imbalance** — only 33.6% of trades are winners; use this as a sanity check
6. **Adding complexity without validation** — every new feature/architecture must be justified by metrics

## Existing Code to Reference

- `screener/ml_signal_v5.py` — feature extractor (reuse `V5FeatureExtractor`)
- `scripts/optimize_v5_100.py` — optimization loop pattern
- `scripts/backtest_v5_fast.py` — walk-forward validation pattern
- `OPTIMIZATION_V5_100_ITERATIONS.md` — current best results

## Final Note

The dataset is small (8,141 samples). **Simplicity + strong regularization wins.** A 2-layer NN with heavy dropout might beat a 10-layer ResNet. An ensemble of 5 XGBoosts might beat a single TabNet. Optimize for **out-of-sample generalization**, not training accuracy.
