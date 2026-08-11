# Research Strategy Backtest Report

**Methodology.** 15 research-backed strategies (see the linked per-strategy markdown files) backtested with `screener backtest-rolling` on **India Nifty 500** and **US S&P 500** over trailing 5/3/2/1-year windows ending 2026-08-11. **Price data: yfinance** (auto-adjusted OHLCV) — the FMP-based lever sweep and cross-source comparison live in `RESEARCH_LEVER_SWEEP_REPORT.md` / `RESEARCH_SOURCE_COMPARISON.md`. Realistic costs: India — NSE statutory fee model (`--cost-model india`) + 10 bps slippage; US — flat 1 bp commission + 5 bps slippage. Portfolio size is strategy-dependent: trend followers run 10 slots with long holds (100-500 days), momentum oscillators 15 slots, mean reversion and reversal factors 20 slots with ~20-day holds. Initial capital 100,000. Benchmark: ^NSEI (India) / SPY (US). Survivorship caveat: universes are today's index members applied to history (no point-in-time membership). Research, not financial advice.


## India (Nifty 500)

### 5-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +16.94 | +0.94 | +1.34 | +0.66 | -25.5% | +37.66 | — | 77 | +49.6% |
| `fifty_two_week_high` | +11.60 | +0.67 | +0.92 | +0.36 | -31.8% | +37.62 | — | 319 | +49.6% |
| `bll_trading_range_break` | +17.78 | +1.10 | +1.49 | +0.84 | -21.2% | +47.06 | — | 51 | +49.6% |
| `keltner_breakout` | +12.82 | +0.82 | +1.13 | +0.47 | -27.4% | +51.08 | — | 186 | +49.6% |
| `adx_trend` | +15.75 | +0.93 | +1.28 | +0.73 | -21.4% | +36.97 | — | 568 | +49.6% |
| `long_term_reversal` | +16.41 | +1.21 | +1.82 | +1.21 | -13.5% | +60.00 | — | 30 | +49.4% |
| `macd_signal_cross` | -9.52 | -0.45 | -0.57 | -0.19 | -50.5% | +33.82 | — | 1425 | +49.4% |
| `stochastic_cross` | -0.45 | +0.06 | +0.08 | -0.01 | -33.7% | +55.13 | — | 1092 | +49.4% |
| `connors_rsi2` | -8.45 | -0.29 | -0.48 | -0.14 | -59.0% | +48.78 | — | 4381 | +49.4% |
| `connors_rsi2_bull` | +0.99 | +0.15 | +0.32 | +0.03 | -39.2% | +54.79 | — | 3656 | +49.4% |
| `bollinger_mean_reversion` | -1.98 | -0.03 | -0.04 | -0.05 | -36.5% | +58.26 | — | 1701 | +49.4% |
| `williams_percent_r` | +1.87 | +0.19 | +0.29 | +0.06 | -33.6% | +57.59 | — | 1667 | +49.4% |
| `cci_reversion` | +5.47 | +0.43 | +0.59 | +0.24 | -22.5% | +53.04 | — | 1333 | +49.4% |
| `short_term_reversal` | +11.73 | +0.70 | +0.97 | +0.41 | -28.6% | +51.69 | — | 1180 | +49.4% |
| `turn_of_month` | +2.41 | +0.32 | +0.41 | +0.15 | -16.4% | +51.50 | — | 1200 | +49.4% |

### 3-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +12.26 | +0.68 | +0.95 | +0.41 | -29.9% | +51.02 | — | 49 | +25.9% |
| `fifty_two_week_high` | +13.93 | +0.79 | +1.07 | +0.67 | -20.7% | +41.27 | — | 189 | +25.9% |
| `bll_trading_range_break` | +23.47 | +1.08 | +1.48 | +0.68 | -34.6% | +55.56 | — | 36 | +25.9% |
| `keltner_breakout` | +21.28 | +1.25 | +1.73 | +1.34 | -15.9% | +57.01 | — | 107 | +25.9% |
| `adx_trend` | +17.21 | +1.00 | +1.43 | +0.74 | -23.3% | +39.38 | — | 325 | +25.8% |
| `long_term_reversal` | +2.46 | +0.24 | +0.36 | +0.09 | -26.6% | +50.00 | — | 10 | +25.8% |
| `macd_signal_cross` | -3.72 | -0.13 | -0.17 | -0.09 | -41.3% | +34.18 | — | 863 | +25.8% |
| `stochastic_cross` | +3.33 | +0.29 | +0.41 | +0.11 | -29.3% | +55.85 | — | 650 | +25.8% |
| `connors_rsi2` | -3.59 | -0.05 | -0.09 | -0.08 | -46.0% | +52.84 | — | 2619 | +25.8% |
| `connors_rsi2_bull` | +7.11 | +0.35 | +0.99 | +0.22 | -32.2% | +56.01 | — | 2246 | +25.8% |
| `bollinger_mean_reversion` | +4.00 | +0.32 | +0.48 | +0.13 | -30.9% | +60.62 | — | 1026 | +25.8% |
| `williams_percent_r` | +7.43 | +0.46 | +0.77 | +0.24 | -31.3% | +58.43 | — | 991 | +25.8% |
| `cci_reversion` | +9.03 | +0.68 | +0.97 | +0.43 | -20.9% | +54.25 | — | 800 | +25.8% |
| `short_term_reversal` | +17.88 | +1.12 | +1.66 | +0.89 | -20.0% | +52.09 | — | 718 | +25.8% |
| `turn_of_month` | -0.85 | -0.03 | -0.04 | -0.05 | -18.5% | +46.39 | — | 720 | +25.8% |

### 2-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | -0.49 | +0.07 | +0.10 | -0.02 | -25.1% | +42.22 | — | 45 | +0.5% |
| `fifty_two_week_high` | -5.98 | -0.18 | -0.24 | -0.23 | -26.2% | +34.27 | — | 143 | +0.5% |
| `bll_trading_range_break` | -10.46 | -0.45 | -0.59 | -0.24 | -42.6% | +39.39 | — | 33 | +0.5% |
| `keltner_breakout` | +7.88 | +0.47 | +0.65 | +0.34 | -23.2% | +45.45 | — | 77 | +0.5% |
| `adx_trend` | -11.38 | -0.45 | -0.61 | -0.33 | -34.9% | +33.61 | — | 241 | +0.4% |
| `long_term_reversal` | +0.00 | +0.00 | +0.00 | +0.00 | +0.0% | +0.00 | — | 0 | +0.4% |
| `macd_signal_cross` | -25.97 | -1.48 | -1.91 | -0.51 | -51.3% | +29.15 | — | 590 | +0.4% |
| `stochastic_cross` | -12.86 | -0.66 | -0.91 | -0.37 | -34.6% | +50.11 | — | 437 | +0.4% |
| `connors_rsi2` | -12.09 | -0.39 | -0.72 | -0.26 | -47.1% | +47.40 | — | 1692 | +0.4% |
| `connors_rsi2_bull` | +3.10 | +0.23 | +0.71 | +0.09 | -33.0% | +52.70 | — | 1391 | +0.4% |
| `bollinger_mean_reversion` | -5.08 | -0.16 | -0.24 | -0.15 | -34.7% | +55.46 | — | 678 | +0.4% |
| `williams_percent_r` | -8.17 | -0.32 | -0.48 | -0.22 | -37.2% | +54.30 | — | 663 | +0.4% |
| `cci_reversion` | -0.68 | +0.06 | +0.08 | -0.03 | -25.5% | +49.16 | — | 535 | +0.4% |
| `short_term_reversal` | +1.69 | +0.19 | +0.27 | +0.07 | -25.8% | +47.71 | — | 480 | +0.4% |
| `turn_of_month` | -0.93 | -0.09 | -0.12 | -0.13 | -7.1% | +47.50 | — | 480 | +0.4% |

### 1-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | -11.72 | -0.69 | -0.90 | -0.52 | -22.5% | +25.81 | — | 31 | -0.5% |
| `fifty_two_week_high` | +6.22 | +0.40 | +0.54 | +0.36 | -17.2% | +36.11 | — | 72 | -0.5% |
| `bll_trading_range_break` | +7.60 | +0.49 | +0.69 | +0.39 | -19.3% | +35.29 | — | 17 | -0.5% |
| `keltner_breakout` | +14.60 | +0.71 | +1.08 | +0.56 | -26.2% | +46.81 | — | 47 | -0.5% |
| `adx_trend` | -12.05 | -0.51 | -0.68 | -0.43 | -27.7% | +38.60 | — | 114 | -0.5% |
| `long_term_reversal` | +0.00 | +0.00 | +0.00 | +0.00 | +0.0% | +0.00 | — | 0 | -0.5% |
| `macd_signal_cross` | -23.57 | -1.46 | -1.87 | -0.83 | -28.5% | +32.63 | — | 285 | -0.5% |
| `stochastic_cross` | -11.11 | -0.63 | -0.88 | -0.42 | -26.3% | +49.09 | — | 220 | -0.5% |
| `connors_rsi2` | +4.75 | +0.29 | +0.71 | +0.16 | -29.9% | +51.72 | — | 903 | -0.5% |
| `connors_rsi2_bull` | +36.17 | +0.89 | +4.11 | +2.76 | -13.1% | +58.11 | — | 740 | -0.5% |
| `bollinger_mean_reversion` | -1.03 | +0.05 | +0.07 | -0.04 | -28.6% | +55.49 | — | 346 | -0.5% |
| `williams_percent_r` | +6.79 | +0.39 | +0.74 | +0.24 | -27.9% | +55.16 | — | 339 | -0.5% |
| `cci_reversion` | +8.25 | +0.56 | +0.82 | +0.39 | -21.4% | +52.54 | — | 276 | -0.5% |
| `short_term_reversal` | +7.04 | +0.43 | +0.64 | +0.28 | -25.3% | +47.92 | — | 240 | -0.5% |
| `turn_of_month` | -3.57 | -0.45 | -0.57 | -0.50 | -7.1% | +48.33 | — | 240 | -0.5% |


## US (S&P 500)

### 5-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +6.24 | +0.42 | +0.61 | +0.20 | -32.0% | +33.33 | — | 93 | +85.8% |
| `fifty_two_week_high` | +11.84 | +0.63 | +0.89 | +0.50 | -23.5% | +38.55 | — | 332 | +85.8% |
| `bll_trading_range_break` | +10.28 | +0.64 | +0.91 | +0.39 | -26.4% | +38.18 | — | 55 | +85.8% |
| `keltner_breakout` | +4.20 | +0.31 | +0.43 | +0.13 | -31.8% | +33.48 | — | 230 | +85.8% |
| `adx_trend` | +7.22 | +0.42 | +0.58 | +0.23 | -31.4% | +36.03 | — | 705 | +85.8% |
| `long_term_reversal` | +18.29 | +0.97 | +1.52 | +0.69 | -26.4% | +76.67 | — | 30 | +85.8% |
| `macd_signal_cross` | +7.51 | +0.51 | +0.72 | +0.24 | -31.6% | +39.00 | — | 1300 | +85.8% |
| `stochastic_cross` | +12.76 | +0.75 | +1.10 | +0.48 | -26.6% | +61.85 | — | 1135 | +85.8% |
| `connors_rsi2` | +10.88 | +0.71 | +1.10 | +0.50 | -21.9% | +63.02 | — | 4367 | +85.8% |
| `connors_rsi2_bull` | +9.06 | +0.75 | +1.16 | +0.67 | -13.6% | +62.99 | — | 3561 | +85.8% |
| `bollinger_mean_reversion` | +8.81 | +0.57 | +0.84 | +0.44 | -20.1% | +64.10 | — | 1794 | +85.8% |
| `williams_percent_r` | +12.48 | +0.71 | +1.04 | +0.51 | -24.6% | +63.49 | — | 1742 | +85.8% |
| `cci_reversion` | +13.84 | +0.79 | +1.17 | +0.56 | -24.6% | +56.70 | — | 1365 | +85.8% |
| `short_term_reversal` | +9.96 | +0.49 | +0.73 | +0.27 | -37.2% | +49.00 | — | 1200 | +85.8% |
| `turn_of_month` | +3.21 | +0.32 | +0.45 | +0.15 | -20.7% | +51.25 | — | 1200 | +85.8% |

### 3-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +13.11 | +0.73 | +1.07 | +0.53 | -24.7% | +44.90 | — | 49 | +79.1% |
| `fifty_two_week_high` | +20.50 | +0.92 | +1.32 | +1.03 | -19.9% | +37.11 | — | 194 | +79.1% |
| `bll_trading_range_break` | +35.33 | +1.27 | +1.86 | +0.87 | -40.6% | +54.84 | — | 31 | +79.1% |
| `keltner_breakout` | +11.36 | +0.69 | +0.97 | +0.47 | -24.4% | +36.43 | — | 140 | +79.1% |
| `adx_trend` | +12.03 | +0.61 | +0.86 | +0.43 | -27.7% | +38.53 | — | 423 | +79.1% |
| `long_term_reversal` | +12.87 | +0.96 | +1.51 | +1.06 | -12.2% | +90.00 | — | 10 | +79.1% |
| `macd_signal_cross` | +12.36 | +0.90 | +1.30 | +0.70 | -17.6% | +39.48 | — | 808 | +79.1% |
| `stochastic_cross` | +19.29 | +1.22 | +1.79 | +1.05 | -18.4% | +62.66 | — | 699 | +79.1% |
| `connors_rsi2` | +11.78 | +0.74 | +1.21 | +0.73 | -16.1% | +62.96 | — | 2716 | +79.1% |
| `connors_rsi2_bull` | +14.75 | +1.16 | +1.92 | +1.59 | -9.3% | +63.85 | — | 2335 | +79.1% |
| `bollinger_mean_reversion` | +9.43 | +0.64 | +0.92 | +0.51 | -18.4% | +63.33 | — | 1080 | +79.1% |
| `williams_percent_r` | +21.09 | +1.34 | +2.02 | +1.44 | -14.7% | +65.97 | — | 1055 | +79.1% |
| `cci_reversion` | +19.88 | +1.24 | +1.91 | +1.21 | -16.4% | +58.95 | — | 816 | +79.1% |
| `short_term_reversal` | +20.82 | +0.98 | +1.53 | +1.01 | -20.6% | +54.17 | — | 720 | +79.1% |
| `turn_of_month` | +3.56 | +0.33 | +0.45 | +0.17 | -20.9% | +48.75 | — | 720 | +79.1% |

### 2-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +22.59 | +1.21 | +1.79 | +1.23 | -18.3% | +58.06 | — | 31 | +48.4% |
| `fifty_two_week_high` | +31.94 | +1.19 | +1.71 | +1.41 | -22.7% | +36.92 | — | 130 | +48.4% |
| `bll_trading_range_break` | +16.67 | +1.00 | +1.43 | +1.11 | -15.0% | +59.09 | — | 22 | +48.4% |
| `keltner_breakout` | +19.63 | +0.90 | +1.25 | +0.88 | -22.4% | +36.17 | — | 94 | +48.4% |
| `adx_trend` | +17.49 | +0.77 | +1.08 | +0.62 | -28.2% | +39.65 | — | 285 | +48.4% |
| `long_term_reversal` | +0.00 | +0.00 | +0.00 | +0.00 | +0.0% | +0.00 | — | 0 | +48.4% |
| `macd_signal_cross` | +10.94 | +0.68 | +0.98 | +0.54 | -20.3% | +38.14 | — | 548 | +48.4% |
| `stochastic_cross` | +23.49 | +1.27 | +1.88 | +1.16 | -20.3% | +61.03 | — | 467 | +48.4% |
| `connors_rsi2` | +12.22 | +0.66 | +1.09 | +0.69 | -17.8% | +61.78 | — | 1813 | +48.4% |
| `connors_rsi2_bull` | +15.44 | +1.01 | +1.71 | +1.45 | -10.7% | +62.70 | — | 1539 | +48.4% |
| `bollinger_mean_reversion` | +11.22 | +0.69 | +0.98 | +0.57 | -19.6% | +63.31 | — | 725 | +48.4% |
| `williams_percent_r` | +26.37 | +1.37 | +2.09 | +1.49 | -17.7% | +66.48 | — | 707 | +48.4% |
| `cci_reversion` | +23.26 | +1.16 | +1.82 | +1.24 | -18.8% | +58.65 | — | 549 | +48.4% |
| `short_term_reversal` | +23.68 | +1.00 | +1.59 | +1.14 | -20.9% | +55.62 | — | 480 | +48.4% |
| `turn_of_month` | +5.12 | +0.41 | +0.56 | +0.31 | -16.6% | +47.50 | — | 480 | +48.4% |

### 1-year window

| Strategy | CAGR | Sharpe | Sortino | Calmar | Max Drawdown | Hit Rate | Avg Exposure | Trades | Benchmark Return |
|---|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +40.61 | +1.62 | +2.56 | +2.62 | -15.5% | +52.63 | — | 19 | +22.9% |
| `fifty_two_week_high` | +68.38 | +1.78 | +2.65 | +3.80 | -18.0% | +42.19 | — | 64 | +22.9% |
| `bll_trading_range_break` | +75.32 | +2.06 | +3.16 | +4.11 | -18.3% | +71.43 | — | 14 | +22.9% |
| `keltner_breakout` | +72.63 | +2.18 | +3.35 | +4.46 | -16.3% | +36.54 | — | 52 | +22.9% |
| `adx_trend` | +38.41 | +1.28 | +1.86 | +2.57 | -14.9% | +39.86 | — | 138 | +22.9% |
| `long_term_reversal` | +0.00 | +0.00 | +0.00 | +0.00 | +0.0% | +0.00 | — | 0 | +22.9% |
| `macd_signal_cross` | +7.12 | +0.45 | +0.66 | +0.41 | -17.3% | +35.99 | — | 289 | +22.9% |
| `stochastic_cross` | +43.53 | +2.35 | +3.75 | +4.39 | -9.9% | +65.56 | — | 241 | +22.9% |
| `connors_rsi2` | +25.08 | +1.12 | +2.02 | +1.58 | -15.9% | +62.03 | — | 985 | +22.9% |
| `connors_rsi2_bull` | +22.44 | +1.27 | +2.32 | +1.92 | -11.7% | +61.24 | — | 872 | +22.9% |
| `bollinger_mean_reversion` | +25.65 | +1.62 | +2.55 | +3.02 | -8.5% | +65.22 | — | 391 | +22.9% |
| `williams_percent_r` | +43.53 | +2.01 | +3.37 | +4.34 | -10.0% | +69.25 | — | 361 | +22.9% |
| `cci_reversion` | +35.07 | +1.66 | +2.83 | +3.08 | -11.4% | +60.29 | — | 277 | +22.9% |
| `short_term_reversal` | +34.91 | +1.56 | +2.54 | +2.35 | -14.9% | +55.00 | — | 240 | +22.9% |
| `turn_of_month` | +21.20 | +1.31 | +2.12 | +2.23 | -9.5% | +56.67 | — | 240 | +22.9% |


## Headline takeaways

### India (Nifty 500)

- Best CAGR: **`connors_rsi2_bull`** 1y → +36.17 (Sharpe +0.89, MaxDD -13.1%)
- Worst CAGR: **`macd_signal_cross`** 2y → -25.97 (Sharpe -1.48)
- `golden_cross_50_200` average CAGR +4.2% — 5/3/2/1y: +16.94 / +12.26 / -0.49 / -11.72
- `fifty_two_week_high` average CAGR +6.4% — 5/3/2/1y: +11.60 / +13.93 / -5.98 / +6.22
- `bll_trading_range_break` average CAGR +9.6% — 5/3/2/1y: +17.78 / +23.47 / -10.46 / +7.60
- `keltner_breakout` average CAGR +14.1% — 5/3/2/1y: +12.82 / +21.28 / +7.88 / +14.60
- `adx_trend` average CAGR +2.4% — 5/3/2/1y: +15.75 / +17.21 / -11.38 / -12.05
- `long_term_reversal` average CAGR +4.7% — 5/3/2/1y: +16.41 / +2.46 / +0.00 / +0.00
- `macd_signal_cross` average CAGR -15.7% — 5/3/2/1y: -9.52 / -3.72 / -25.97 / -23.57
- `stochastic_cross` average CAGR -5.3% — 5/3/2/1y: -0.45 / +3.33 / -12.86 / -11.11
- `connors_rsi2` average CAGR -4.8% — 5/3/2/1y: -8.45 / -3.59 / -12.09 / +4.75
- `connors_rsi2_bull` average CAGR +11.8% — 5/3/2/1y: +0.99 / +7.11 / +3.10 / +36.17
- `bollinger_mean_reversion` average CAGR -1.0% — 5/3/2/1y: -1.98 / +4.00 / -5.08 / -1.03
- `williams_percent_r` average CAGR +2.0% — 5/3/2/1y: +1.87 / +7.43 / -8.17 / +6.79
- `cci_reversion` average CAGR +5.5% — 5/3/2/1y: +5.47 / +9.03 / -0.68 / +8.25
- `short_term_reversal` average CAGR +9.6% — 5/3/2/1y: +11.73 / +17.88 / +1.69 / +7.04
- `turn_of_month` average CAGR -0.7% — 5/3/2/1y: +2.41 / -0.85 / -0.93 / -3.57
### US (S&P 500)

- Best CAGR: **`bll_trading_range_break`** 1y → +75.32 (Sharpe +2.06, MaxDD -18.3%)
- Worst CAGR: **`turn_of_month`** 5y → +3.21 (Sharpe +0.32)
- `golden_cross_50_200` average CAGR +20.6% — 5/3/2/1y: +6.24 / +13.11 / +22.59 / +40.61
- `fifty_two_week_high` average CAGR +33.2% — 5/3/2/1y: +11.84 / +20.50 / +31.94 / +68.38
- `bll_trading_range_break` average CAGR +34.4% — 5/3/2/1y: +10.28 / +35.33 / +16.67 / +75.32
- `keltner_breakout` average CAGR +27.0% — 5/3/2/1y: +4.20 / +11.36 / +19.63 / +72.63
- `adx_trend` average CAGR +18.8% — 5/3/2/1y: +7.22 / +12.03 / +17.49 / +38.41
- `long_term_reversal` average CAGR +7.8% — 5/3/2/1y: +18.29 / +12.87 / +0.00 / +0.00
- `macd_signal_cross` average CAGR +9.5% — 5/3/2/1y: +7.51 / +12.36 / +10.94 / +7.12
- `stochastic_cross` average CAGR +24.8% — 5/3/2/1y: +12.76 / +19.29 / +23.49 / +43.53
- `connors_rsi2` average CAGR +15.0% — 5/3/2/1y: +10.88 / +11.78 / +12.22 / +25.08
- `connors_rsi2_bull` average CAGR +15.4% — 5/3/2/1y: +9.06 / +14.75 / +15.44 / +22.44
- `bollinger_mean_reversion` average CAGR +13.8% — 5/3/2/1y: +8.81 / +9.43 / +11.22 / +25.65
- `williams_percent_r` average CAGR +25.9% — 5/3/2/1y: +12.48 / +21.09 / +26.37 / +43.53
- `cci_reversion` average CAGR +23.0% — 5/3/2/1y: +13.84 / +19.88 / +23.26 / +35.07
- `short_term_reversal` average CAGR +22.3% — 5/3/2/1y: +9.96 / +20.82 / +23.68 / +34.91
- `turn_of_month` average CAGR +8.3% — 5/3/2/1y: +3.21 / +3.56 / +5.12 / +21.20