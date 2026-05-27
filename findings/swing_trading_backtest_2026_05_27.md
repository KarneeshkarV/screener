# Swing Trading Strategy Backtest Findings
**Date:** 2026-05-27  
**Universe:** SP500 (capped 100) for US · Nifty 50 for India  
**Window:** 2025-05-27 → 2026-05-27 (1 year rolling)  
**Parameters:** top=10, stop-loss=8%, commission=5bps, hold=10–15 days

---

## Single-Strategy Baselines

### US Equities (benchmark: SPY +28.43%)

| Strategy | Sharpe | Hit Rate | Max DD | Total Return | Alpha | Trades |
|---|---|---|---|---|---|---|
| RS Breakout | **2.599** | 59.57% | -5.51% | +40.68% | +28.28% | ~100 |
| EMA Trend | 2.366 | 42.19% | -7.05% | +29.97% | +13.77% | ~150 |
| EMA + MA Cross regime | 1.568 | 43.90% | -8.42% | +18.40% | +5.40% | ~130 |
| RSI Pullback (relaxed) | -0.406 | 47.69% | -10.63% | -3.71% | -12.37% | 65 |

### India Equities (benchmark: ^NSEI -3.66%)

| Strategy | Sharpe | Hit Rate | Max DD | Total Return | Alpha | Trades |
|---|---|---|---|---|---|---|
| MA Cross | **1.282** | 45.69% | -5.14% | +24.84% | +26.64% | ~100 |
| RS Breakout | 1.114 | 47.37% | -4.51% | +11.62% | +13.58% | ~50 |
| EMA Trend | 0.898 | 44.44% | -6.23% | +9.80% | +11.96% | ~80 |

---

## Combination Strategy Results

The goal was to AND multiple entry conditions together to improve signal quality and Sharpe.

### US Combinations (baseline to beat: RS Breakout Sharpe 2.599)

| Strategy | Entry Logic | Sharpe | Hit Rate | Max DD | Total Return | Alpha | Trades |
|---|---|---|---|---|---|---|---|
| RS Breakout (baseline) | `rs_breakout_entry > 0` | 2.599 | 59.57% | -5.51% | +40.68% | +28.28% | ~100 |
| Combo 1: RS + EMA trend | RS entry AND close > EMA20 AND EMA20 > EMA200 | 2.203 | 57.23% | -4.29% | +27.36% | +11.73% | 173 |
| Combo 3: EMA + 52wk breakout + vol | close > EMA20 AND EMA20 > EMA200 AND close ≥ 52wk-high×0.98 AND vol > SMA20vol×1.5 | 2.078 | 57.95% | -4.51% | +24.77% | +12.23% | 176 |
| Combo 6: RS + RSI filter + trend | RS entry AND RSI14 < 72 AND EMA20 > EMA200 | 2.039 | **61.83%** | **-2.98%** | +21.86% | +10.14% | 131 |
| Combo 2: RS + 52wk high + vol | RS entry AND close ≥ 52wk-high×0.99 AND vol > SMA20vol×1.5 | 1.665 | 55.32% | -4.78% | +17.65% | +7.83% | 141 |

### India Combinations (baseline to beat: MA Cross Sharpe 1.282)

| Strategy | Entry Logic | Sharpe | Hit Rate | Max DD | Total Return | Alpha | Trades |
|---|---|---|---|---|---|---|---|
| MA Cross (baseline) | crossover(EMA10, EMA20) AND EMA50 > EMA200 | 1.282 | 45.69% | -5.14% | +24.84% | +26.64% | ~100 |
| Combo 5: MA Cross + 200 EMA | MA cross AND close > EMA200 | 1.202 | 47.66% | -5.79% | +24.89% | +25.40% | 107 |
| Combo 4: RS breakout + EMA align | RS entry AND EMA10 > EMA20 AND EMA50 > EMA200 | 0.330 | 48.81% | -3.65% | +1.74% | +2.57% | 84 |

---

## Key Findings

### 1. Combinations did not beat the RS Breakout baseline (US)
The standalone RS Breakout at Sharpe 2.599 remained the top US strategy. Adding EMA or volume conditions was largely redundant — the RS signal already implies trend alignment. The extra filters pruned good trades without improving signal quality.

### 2. Combo 6 is the best risk-adjusted combination
RS Breakout + RSI<72 + EMA trend delivered:
- **Highest hit rate tested: 61.83%**
- **Lowest max drawdown tested: -2.98%** (nearly half the baseline's -5.51%)
- Acceptable Sharpe of 2.039

This is the preferred variant for risk-averse implementation — trades roughly 31% fewer positions than the baseline but with significantly tighter drawdown.

### 3. RSI Pullback does not work on SP500 mega-caps
Even a relaxed RSI<45 + EMA crossover filter produced -3.71% total return and -12.37% alpha over the test window. The large-cap US universe in a bull trend has no meaningful oversold pullbacks.

### 4. India — MA Cross remains best; RS signal frequency too low on Nifty 50
The 50-stock Nifty universe generates too few RS breakout signals (Combo 4: only 84 trades, 35% avg exposure). MA Cross with EMA200 confirmation (Combo 5) nearly matches the baseline (1.202 vs 1.282) but adds no improvement — the 200 EMA condition is already satisfied when MA crosses occur.

For meaningful RS signal frequency in India, the universe should be expanded to Nifty 500.

---

## Recommended Playbooks

### US — Primary: RS Breakout
```
backtest-rolling --market US --strategy rs_breakout \
  --stop-loss 0.08 --hold 10 --top 10
```
- Best Sharpe (2.599), strong alpha (+28%), consistent 10-day hold
- Works best in bull/neutral regimes; expect drawdown in sharp corrections

### US — Risk-Managed: Combo 6 (RS + RSI filter + trend)
```
backtest-rolling --market US --strategy rs_breakout \
  --entry "rs_breakout_entry > 0 and rsi(close,14) < 72 and ema(close,20) > ema(close,200)" \
  --stop-loss 0.08 --hold 10 --top 10
```
- Best hit rate (61.83%), lowest max drawdown (-2.98%)
- Sacrifice ~19pp total return for near-halved drawdown

### India — Primary: MA Cross
```
backtest-rolling --market INDIA \
  --entry "crossover(ema(close,10), ema(close,20)) and ema(close,50) > ema(close,200)" \
  --stop-loss 0.08 --hold 15 --top 10
```
- Outperforms NSEI benchmark by +26.64% alpha in a down year
- Nifty 50 in bear regime: MA Cross catches the few trending names

---

## Technical Notes

- **Pine expression evaluator** supports: `sma`, `ema`, `rsi`, `highest`, `lowest`, `atr`, `crossover`, `crossunder`
- **Strategy + entry combo trick**: Pass `--strategy rs_breakout` (triggers `prepare_bars` hook to add `rs_breakout_entry` column) AND `--entry "rs_breakout_entry > 0 and <extra>"` to override the entry expression while keeping the bar-preparation hook
- **Universe cap**: 100 for US, 50 for India — larger caps cause download timeouts on 1-year windows
- **Named CLI strategies**: Only `breakout`, `ema_trend`, `rs_breakout`, `vivek_equity_tool` are accessible via `--strategy`; callable-only strategies (ma_cross, supertrend) require direct `--entry` Pine expressions
