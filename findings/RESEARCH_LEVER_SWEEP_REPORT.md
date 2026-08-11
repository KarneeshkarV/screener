# Lever Sweep Report — execution levers per strategy (FMP price data)

**Method.** For every strategy x market, a 3-year window (ending 2026-08-11) was swept over regime filter (none / bull / bull+pullback), stop loss (none/8%/15%/25%), take profit (none/25%), trailing stop (none/15%/25%), and sizing (equal_slot + atr_risk/fixed_fraction/inverse_vol on the grid-best). Best config chosen by Sharpe with >= 8 trades. Price data: **FMP** (historical-price-full, dividend/split adjusted). Costs: India NSE statutory + 10bps slippage; US flat 1bp + 5bps slippage. Tuned-vs-baseline validated on 5/2/1y windows. Research, not financial advice.


## India (Nifty 500)

| Strategy | Baseline 3y CAGR / Sharpe | Best 3y CAGR / Sharpe / MDD | Best config | Tuned 5y/2y/1y CAGR (baseline) |
|---|---|---|---|---|
| `golden_cross_50_200` | +15.5% / +0.83 | +18.8% / +1.64 / -7.4% | `regime=bull,pullback SL none TP 0.25 trail 0.15` | +10.4% (+13.0%) / +4.0% (+3.4%) / -8.7% (-16.9%) |
| `fifty_two_week_high` | +20.5% / +1.16 | +20.0% / +1.56 / -13.0% | `regime=bull SL 0.25 TP 0.25 trail none sizing=atr_risk` | +12.0% (+14.9%) / +3.5% (+7.6%) / +0.0% (+17.3%) |
| `bll_trading_range_break` | +18.6% / +0.94 | +17.0% / +1.30 / -12.4% | `regime=bull,pullback SL none TP none trail 0.15 sizing=atr_risk` | +7.6% (+17.0%) / -6.2% (-9.4%) / -4.4% (+7.2%) |
| `keltner_breakout` | +14.4% / +0.90 | +29.9% / +1.48 / -10.9% | `regime=bull,pullback SL none TP 0.25 trail none` | +8.6% (+9.4%) / +4.1% (-1.5%) / +14.9% (-2.3%) |
| `adx_trend` | +21.9% / +1.26 | +22.1% / +1.26 / -16.8% | `regime=none SL 0.15 TP none trail 0.25` | +17.9% (+18.4%) / -3.7% (-4.2%) / -6.9% (-4.9%) |
| `long_term_reversal` | -1.1% / -0.02 | +3.3% / +0.32 / -24.5% | `regime=none SL none TP 0.25 trail 0.25` | +7.6% (+16.2%) / +0.0% (+0.0%) / +0.0% (+0.0%) |
| `macd_signal_cross` | -4.5% / -0.18 | +3.8% / +0.36 / -32.0% | `regime=bull,pullback SL 0.15 TP 0.25 trail none` | -1.1% (-7.0%) / -19.8% (-23.8%) / -18.2% (-23.8%) |
| `stochastic_cross` | +5.8% / +0.44 | +5.9% / +0.45 / -25.7% | `regime=none SL none TP 0.25 trail none` | +3.8% (-0.4%) / -6.6% (-6.8%) / -4.8% (-5.3%) |
| `connors_rsi2` | -7.3% / -0.32 | -7.2% / -0.32 / -46.0% | `regime=none SL 0.25 TP 0.25 trail none` | -11.8% (-11.0%) / -15.1% (-15.4%) / -5.7% (-5.3%) |
| `connors_rsi2_bull` | +6.6% / +0.35 | +6.6% / +0.35 / -30.9% | `regime=none SL none TP none trail none` | +1.0% (+1.0%) / +5.3% (+5.3%) / +36.0% (+36.0%) |
| `bollinger_mean_reversion` | +1.6% / +0.18 | +2.6% / +0.25 / -29.7% | `regime=none SL 0.25 TP 0.25 trail none` | -2.2% (-3.0%) / -8.3% (-7.3%) / -3.4% (-2.3%) |
| `williams_percent_r` | +2.8% / +0.25 | +5.1% / +0.39 / -30.1% | `regime=none SL 0.15 TP none trail none` | +0.3% (+0.1%) / -7.5% (-4.9%) / +3.1% (+3.7%) |
| `cci_reversion` | +5.7% / +0.46 | +8.6% / +0.65 / -21.1% | `regime=none SL none TP none trail 0.25` | +6.1% (+3.8%) / -8.2% (-2.5%) / +2.7% (+4.2%) |
| `short_term_reversal` | +16.7% / +1.02 | +16.7% / +1.02 / -19.3% | `regime=none SL none TP none trail none` | +12.6% (+12.6%) / -2.8% (-2.8%) / +9.7% (+9.7%) |
| `turn_of_month` | -2.5% / -0.17 | -0.8% / -0.04 / -18.9% | `regime=bull,pullback SL none TP 0.25 trail 0.15` | +2.2% (+1.5%) / -1.2% (-1.5%) / -2.5% (-4.6%) |

## US (S&P 500)

| Strategy | Baseline 3y CAGR / Sharpe | Best 3y CAGR / Sharpe / MDD | Best config | Tuned 5y/2y/1y CAGR (baseline) |
|---|---|---|---|---|
| `golden_cross_50_200` | +9.1% / +0.56 | +19.6% / +1.20 / -18.4% | `regime=bull SL 0.08 TP none trail none` | +14.4% (+7.3%) / +11.9% (+16.0%) / +67.4% (+47.7%) |
| `fifty_two_week_high` | +20.3% / +0.90 | +25.9% / +1.13 / -17.5% | `regime=bull,pullback SL 0.15 TP none trail none sizing=inverse_vol` | +14.6% (+11.7%) / +35.8% (+31.5%) / +62.0% (+67.3%) |
| `bll_trading_range_break` | +35.3% / +1.27 | +21.4% / +1.66 / -9.9% | `regime=bull SL 0.15 TP 0.25 trail 0.25` | +10.6% (+10.3%) / +17.8% (+16.7%) / +18.1% (+75.3%) |
| `keltner_breakout` | +4.1% / +0.33 | +22.8% / +1.20 / -16.7% | `regime=bull SL 0.25 TP none trail none sizing=inverse_vol` | +15.7% (+2.1%) / +21.1% (+19.7%) / +77.3% (+72.0%) |
| `adx_trend` | +11.5% / +0.59 | +14.0% / +0.87 / -17.2% | `regime=bull,pullback SL none TP 0.25 trail 0.15 sizing=atr_risk` | +12.0% (+7.3%) / +15.0% (+16.6%) / +24.2% (+36.2%) |
| `long_term_reversal` | +12.9% / +0.96 | +16.6% / +1.38 / -6.5% | `regime=bull SL 0.25 TP 0.25 trail none` | +11.1% (+18.3%) / +0.0% (+0.0%) / +0.0% (+0.0%) |
| `macd_signal_cross` | +11.9% / +0.88 | +13.8% / +1.04 / -17.2% | `regime=none SL none TP none trail 0.15` | +8.5% (+7.3%) / +13.0% (+10.3%) / +5.4% (+11.1%) |
| `stochastic_cross` | +17.9% / +1.15 | +17.7% / +1.27 / -17.4% | `regime=none SL 0.08 TP none trail none sizing=atr_risk` | +11.5% (+12.5%) / +12.1% (+21.6%) / +23.5% (+40.4%) |
| `connors_rsi2` | +11.5% / +0.72 | +13.8% / +0.85 / -16.3% | `regime=none SL 0.08 TP 0.25 trail 0.15` | +10.4% (+10.6%) / +14.2% (+12.0%) / +33.6% (+26.2%) |
| `connors_rsi2_bull` | +14.4% / +1.13 | +14.5% / +1.15 / -9.2% | `regime=none SL none TP 0.25 trail 0.25` | +8.7% (+8.6%) / +14.9% (+14.8%) / +22.6% (+21.6%) |
| `bollinger_mean_reversion` | +8.7% / +0.61 | +11.8% / +0.79 / -15.7% | `regime=none SL none TP 0.25 trail 0.15` | +9.3% (+8.3%) / +15.0% (+9.7%) / +27.2% (+23.2%) |
| `williams_percent_r` | +20.5% / +1.35 | +21.2% / +1.36 / -14.4% | `regime=none SL none TP 0.25 trail none` | +12.2% (+12.1%) / +26.5% (+25.2%) / +42.4% (+39.1%) |
| `cci_reversion` | +19.1% / +1.19 | +19.2% / +1.34 / -13.9% | `regime=none SL 0.08 TP 0.25 trail 0.25` | +12.8% (+13.7%) / +21.9% (+23.2%) / +26.1% (+33.5%) |
| `short_term_reversal` | +21.2% / +1.01 | +21.2% / +1.01 / -20.4% | `regime=none SL none TP none trail none` | +10.2% (+10.2%) / +23.8% (+23.8%) / +28.0% (+28.0%) |
| `turn_of_month` | +3.4% / +0.32 | +3.7% / +0.35 / -19.0% | `regime=none SL 0.08 TP 0.25 trail none` | +3.9% (+3.3%) / +4.6% (+5.1%) / +22.4% (+22.2%) |