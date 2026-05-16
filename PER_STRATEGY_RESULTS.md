# Per-Strategy Model Results

## Category Models

- trend: n=3337 AUC=0.5781 top10WR=54.4%
- mean_rev: n=3877 AUC=0.4871 top10WR=28.4%
- breakout: n=927 AUC=0.5134 top10WR=40.2%

## Ensemble Results

| Method | AUC | Top 10% WR | Top 10% Avg |
|--------|-----|-----------|-------------|
| Weighted Average | 0.5550 | 49.0% | 1.739% |
| Stacked Ridge | 0.5490 | 49.8% | 2.102% |

## Comparison to Baseline

| Model | AUC | Top 10% WR |
|-------|-----|-----------|
| v5 Single (best) | 0.6346 | 66.0% |
| v5 Ensemble | 0.6438 | 67.4% |
| Per-Strategy Weighted | 0.5550 | 49.0% |
| Per-Strategy Stacked | 0.5490 | 49.8% |
