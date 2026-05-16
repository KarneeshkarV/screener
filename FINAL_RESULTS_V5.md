# Final Results: V5 ML Signal Confidence -- Full Experiment Log

## Mission Status

| Target | Best Achieved | Status | Gap |
|--------|--------------|--------|-----|
| AUC > 0.650 | **0.6438** | ❌ MISS | -0.0062 |
| Top 10% WR > 65% | **67.4%** | ✅ HIT | +2.4pp |
| Top 10% Avg > 4.5% | **4.57%** | ✅ HIT | +0.07pp |

**Result: 2 out of 3 targets hit. AUC ceiling appears to be ~0.644 for this dataset.**

---

## Complete Experiment History

### Experiment 1: 100-Iteration Hyperparameter Optimization
- 100 configs × 5-fold CV = 500 model trainings
- Best single model: AUC=0.6346, top10% WR=66.0%
- Best config: XGBRegressor(depth=5, lr=0.05, subsample=0.6, lambda=3.0)

### Experiment 2: Ensemble of Top 5 XGBoost Configs ✅ BEST
- Stacked Ridge meta-learner on 5 XGBoost OOF predictions
- **AUC=0.6438**, top10% WR=67.4%, top10% avg=+4.57%
- Improvement: +0.0103 AUC over single best

### Experiment 3: Expanded Ensemble (5 XGB + LightGBM + CatBoost) ❌ WORSE
- Added LightGBM (AUC=0.6229) and CatBoost (AUC=0.6041)
- 7-model weighted ensemble: AUC=0.6362
- **Verdict**: weaker models diluted the ensemble

### Experiment 4: Sklearn MLP Neural Net ❌ WORSE
- 3-layer MLP (64→32→16): AUC=0.5724
- **Verdict**: 8,141 samples too small for NN to beat trees

### Experiment 5: TabNet ⚠️ TIMEOUT
- Too slow for CPU training (>300s)

### Experiment 6: Feature Interactions ❌ WORSE
- Added 10 domain-specific interactions (rvol×returns, sharpe×drawdown, etc.)
- 5-model ensemble with interactions: AUC=0.6415
- **Verdict**: interactions added noise rather than signal

---

## Why AUC > 0.650 Could Not Be Reached

### 1. Dataset Size Ceiling
- 8,141 trades is the bottleneck
- With 27 features, the signal-to-noise ratio is fixed
- More data (15K+ trades) is the only reliable path to higher AUC

### 2. Base Strategy Alpha Decay
- Walk-forward AUC on the strategy is ~0.50 (near random)
- The base strategy's win rate dropped from ~44% to ~28% over time
- No ML model can fix a decayed underlying edge

### 3. Regime Non-Stationarity
- Training spans 2020–2024 (bull → bear → chop)
- Cross-sectional ranking works (AUC ~0.64) but temporal generalization is poor
- The model captures "today's best setups" but not "tomorrow's winners"

### 4. Feature Set Limitations
- All features are derived from OHLCV + basic technicals
- Missing: options flow, sentiment, inter-market correlations, fundamentals
- These are the features that typically push AUC from 0.64 → 0.70+

---

## What Actually Works

| Approach | AUC | Top10% WR | Recommendation |
|----------|-----|-----------|----------------|
| **5-XGB Stacked Ridge** | **0.6438** | **67.4%** | **← Use this in production** |
| Single optimized XGB | 0.6346 | 66.0% | Good fallback |
| Baseline v5 | 0.586 | 59.0% | Deprecated |

**Production model: `scripts/training_data_v4/model_v5_ensemble.pkl`**

---

## Recommendations to Hit AUC > 0.650 in the Future

1. **More data**: Need 15,000+ trades. The current 8,141 is the hard ceiling.
2. **New features**: Add VIX, sector breadth, options put/call ratio, earnings surprise, institutional flow
3. **Temporal sequences**: Use LSTM/Transformer on last 3–5 signal feature sequences (requires more data)
4. **Multi-strategy ensemble**: Train separate models for EMA-trend, RS-breakout, mean-reversion, etc. then ensemble
5. **Meta-labeling**: Use the ML model to predict win probability of a *secondary* strategy, not the primary one
6. **Switch base strategy**: The current strategy's alpha has decayed. A strategy with 40%+ baseline WR would give the ML layer more to work with

---

## Files Summary

| File | Purpose |
|------|---------|
| `scripts/optimize_v5_100.py` | 100-iteration HP search |
| `scripts/ensemble_top5_v5.py` | 5-XGB ensemble (best result) |
| `scripts/ensemble_expanded_v5.py` | 7-model ensemble (XGB+LGB+CB) |
| `scripts/nn_sklearn_v5.py` | MLP baseline |
| `scripts/tabnet_v5.py` | TabNet (timed out) |
| `scripts/interaction_features_v5.py` | Feature interactions experiment |
| `scripts/training_data_v4/model_v5_ensemble.pkl` | **Production model** |
| `OPTIMIZATION_V5_100_ITERATIONS.md` | HP optimization report |
| `ADVANCED_ML_RESULTS.md` | Ensemble + NN results |
| `FINAL_RESULTS_V5.md` | This file |
