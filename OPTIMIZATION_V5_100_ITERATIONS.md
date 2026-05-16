# V5 Model — 100-Iteration Hyperparameter Optimization Report

## Summary

Ran **100 configurations** × **5-fold stratified CV** = **500 model trainings** to find the optimal hyperparameters for the V5 regression-based signal confidence model.

## Dataset

- **Total trades**: 8,141
- **US**: 4,093 trades | **India**: 4,048 trades
- **Baseline win rate**: 33.6%
- **Baseline avg return**: -0.015%

## Best Configuration (Rank #1)

| Metric | Value |
|--------|-------|
| **AUC** | **0.6346** (± 0.0083) |
| **Top 10% Win Rate** | **66.0%** (+32.4% vs baseline) |
| **Top 10% Avg Return** | **+4.22%** |
| **Top 20% Win Rate** | **54.4%** (+20.8% vs baseline) |
| **Top 20% Avg Return** | **+2.33%** |
| US AUC | 0.6322 |
| India AUC | 0.6362 |

### Hyperparameters

```python
XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.6,
    colsample_bylevel=0.9,
    reg_lambda=3.0,
    reg_alpha=0.0,
    min_child_weight=1,
    gamma=0.0,
)
```

### Key Insights

- **All 27 features** were used — the extended feature set adds real signal
- **Aggressive subsampling** (0.6) prevents overfitting better than higher values
- **Moderate regularization** (lambda=3.0) — heavier regularization killed signal
- **Depth 5** is the sweet spot — depth 2-3 underfits, depth 6+ overfits

## Top 10 Configurations

| Rank | AUC | Top10 WR | Top10 Avg | Top20 WR | Score | Feats | Depth | LR | Lambda |
|------|-----|----------|-----------|----------|-------|-------|-------|----|--------|
| 1 | 0.6346 | 66.0% | 4.222% | 54.4% | 1.2837 | 27 | 5 | 0.05 | 3.0 |
| 2 | 0.6195 | 63.6% | 4.218% | 53.2% | 1.2192 | 12 | 6 | 0.10 | 2.0 |
| 3 | 0.6208 | 62.7% | 3.889% | 52.0% | 1.2033 | 15 | 5 | 0.10 | 0.5 |
| 4 | 0.6107 | 61.4% | 3.719% | 52.6% | 1.1659 | 25 | 6 | 0.07 | 2.0 |
| 5 | 0.5975 | 58.1% | 3.203% | 50.6% | 1.0885 | 27 | 6 | 0.03 | 10.0 |
| 6 | 0.5933 | 56.5% | 3.118% | 48.7% | 1.0523 | 17 | 4 | 0.10 | 10.0 |
| 7 | 0.5845 | 56.8% | 3.125% | 47.4% | 1.0483 | 19 | 6 | 0.15 | 1.0 |
| 8 | 0.5907 | 56.0% | 3.137% | 48.9% | 1.0398 | 15 | 4 | 0.07 | 3.0 |
| 9 | 0.5896 | 55.4% | 2.844% | 48.2% | 1.0263 | 23 | 3 | 0.05 | 2.0 |
| 10 | 0.5772 | 55.1% | 2.838% | 46.8% | 1.0065 | 19 | 3 | 0.15 | 0.5 |

## What Didn't Work

- **~50% of configs** produced AUC = 0.5000 exactly — these were degenerate models where regularization was either too heavy (lambda=10 + low depth) or the learning rate/feature combo failed to converge
- **Very low subsample** (< 0.6) often caused instability
- **Depth 2** models consistently underfit (AUC ~0.55-0.56 max)
- **High learning rate + high depth** (e.g., lr=0.15, depth=6) overfit on some folds

## Improvements Over V5 Baseline

| Metric | V5 Baseline | Optimized | Improvement |
|--------|-------------|-----------|-------------|
| AUC | 0.586 | **0.635** | +8.4% |
| Top 10% WR | 59% | **66%** | +7.0 pp |
| Top 10% Avg Return | +2.4% | **+4.22%** | +75% |

## Production Models

Three optimized models saved:

1. `model_v5_optimized.pkl` — trained on all 8,141 trades (global)
2. `model_v5_us_optimized.pkl` — trained on 4,093 US trades
3. `model_v5_india_optimized.pkl` — trained on 4,048 India trades

## Files

- `scripts/optimize_v5_100.py` — optimization script (can re-run with different seeds)
- `scripts/training_data_v4/optimization_results_v5.json` — full results of all 100 configs
- `OPTIMIZATION_V5_100_ITERATIONS.md` — this report
