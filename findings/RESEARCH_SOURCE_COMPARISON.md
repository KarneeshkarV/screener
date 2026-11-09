# Data-Source Comparison — yfinance vs FMP

Baseline (no levers) numbers for the 15 strategies, same universes, same costs, same windows — only the price provider changes (yfinance auto-adjusted OHLCV vs FMP historical-price-full adjClose). The last column shows the tuned lever config on FMP. Research, not financial advice.


## India (Nifty 500)

| Strategy | 3y yf CAGR | 3y FMP CAGR | 5y yf | 5y FMP base | 5y FMP tuned | 1y yf | 1y FMP base | 1y FMP tuned |
|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +12.3% | +15.5% | +16.9% | +13.0% | +10.4% | -11.7% | -16.9% | -8.7% |
| `fifty_two_week_high` | +13.9% | +20.5% | +11.6% | +14.9% | +12.0% | +6.2% | +17.3% | +0.0% |
| `bll_trading_range_break` | +23.5% | +18.6% | +17.8% | +17.0% | +7.6% | +7.6% | +7.2% | -4.4% |
| `keltner_breakout` | +21.3% | +14.4% | +12.8% | +9.4% | +8.6% | +14.6% | -2.3% | +14.9% |
| `adx_trend` | +17.2% | +21.9% | +15.8% | +18.4% | +17.9% | -12.1% | -4.9% | -6.9% |
| `long_term_reversal` | +2.5% | -1.1% | +16.4% | +16.2% | +7.6% | +0.0% | +0.0% | +0.0% |
| `macd_signal_cross` | -3.7% | -4.5% | -9.5% | -7.0% | -1.1% | -23.6% | -23.8% | -18.2% |
| `stochastic_cross` | +3.3% | +5.8% | -0.5% | -0.4% | +3.8% | -11.1% | -5.3% | -4.8% |
| `connors_rsi2` | -3.6% | -7.3% | -8.4% | -11.0% | -11.8% | +4.8% | -5.3% | -5.7% |
| `connors_rsi2_bull` | +7.1% | +6.6% | +1.0% | +1.0% | +1.0% | +36.2% | +36.0% | +36.0% |
| `bollinger_mean_reversion` | +4.0% | +1.6% | -2.0% | -3.0% | -2.2% | -1.0% | -2.3% | -3.4% |
| `williams_percent_r` | +7.4% | +2.8% | +1.9% | +0.1% | +0.3% | +6.8% | +3.7% | +3.1% |
| `cci_reversion` | +9.0% | +5.7% | +5.5% | +3.8% | +6.1% | +8.2% | +4.2% | +2.7% |
| `short_term_reversal` | +17.9% | +16.7% | +11.7% | +12.6% | +12.6% | +7.0% | +9.7% | +9.7% |
| `turn_of_month` | -0.8% | -2.5% | +2.4% | +1.5% | +2.2% | -3.6% | -4.6% | -2.5% |

## US (S&P 500)

| Strategy | 3y yf CAGR | 3y FMP CAGR | 5y yf | 5y FMP base | 5y FMP tuned | 1y yf | 1y FMP base | 1y FMP tuned |
|---|---|---|---|---|---|---|---|---|
| `golden_cross_50_200` | +13.1% | +9.1% | +6.2% | +7.3% | +14.4% | +40.6% | +47.7% | +67.4% |
| `fifty_two_week_high` | +20.5% | +20.3% | +11.8% | +11.7% | +14.6% | +68.4% | +67.3% | +62.0% |
| `bll_trading_range_break` | +35.3% | +35.3% | +10.3% | +10.3% | +10.6% | +75.3% | +75.3% | +18.1% |
| `keltner_breakout` | +11.4% | +4.1% | +4.2% | +2.1% | +15.7% | +72.6% | +72.0% | +77.3% |
| `adx_trend` | +12.0% | +11.5% | +7.2% | +7.3% | +12.0% | +38.4% | +36.2% | +24.2% |
| `long_term_reversal` | +12.9% | +12.9% | +18.3% | +18.3% | +11.1% | +0.0% | +0.0% | +0.0% |
| `macd_signal_cross` | +12.4% | +11.9% | +7.5% | +7.3% | +8.5% | +7.1% | +11.1% | +5.4% |
| `stochastic_cross` | +19.3% | +17.9% | +12.8% | +12.5% | +11.5% | +43.5% | +40.4% | +23.5% |
| `connors_rsi2` | +11.8% | +11.5% | +10.9% | +10.6% | +10.4% | +25.1% | +26.2% | +33.6% |
| `connors_rsi2_bull` | +14.8% | +14.4% | +9.1% | +8.6% | +8.7% | +22.4% | +21.6% | +22.6% |
| `bollinger_mean_reversion` | +9.4% | +8.7% | +8.8% | +8.3% | +9.3% | +25.6% | +23.2% | +27.2% |
| `williams_percent_r` | +21.1% | +20.5% | +12.5% | +12.1% | +12.2% | +43.5% | +39.1% | +42.4% |
| `cci_reversion` | +19.9% | +19.1% | +13.8% | +13.7% | +12.8% | +35.1% | +33.5% | +26.1% |
| `short_term_reversal` | +20.8% | +21.2% | +10.0% | +10.2% | +10.2% | +34.9% | +28.0% | +28.0% |
| `turn_of_month` | +3.6% | +3.4% | +3.2% | +3.3% | +3.9% | +21.2% | +22.2% | +22.4% |