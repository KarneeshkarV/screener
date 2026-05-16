# ML Signal Confidence Model

Trained on a US backtest of 10 large-cap stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, V, XOM) over 2024-01-01 to 2025-01-01 using an EMA-trend strategy.

## Model Specs
- **Algorithm**: XGBoost Classifier (n_estimators=100, max_depth=4)
- **Features**: 15 quantitative features (volume profile, momentum, trend alignment, volatility, market context)
- **Training data**: 146 trades (43.8% win rate)
  - Train: 116 trades
  - Test: 30 trades
- **Performance**:
  - AUC: 0.6018
  - Accuracy: 60.0%

## How to regenerate

```bash
python scripts/generate_training_data.py
python main.py train-model \
  --trades scripts/training_data/trades.json \
  --bars scripts/training_data/bars.json \
  --output scripts/training_data/model.pkl
```

## Usage in screening

```bash
python main.py rs-breakout -m us --tickers AAPL,MSFT,NVDA \
  --confidence-model scripts/training_data/model.pkl \
  --confidence-threshold 0.55
```
