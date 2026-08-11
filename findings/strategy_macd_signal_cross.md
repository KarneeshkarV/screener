# MACD Signal-Line Cross (12, 26, 9)

**Strategy key:** `macd_signal_cross`

## Research source

**Book:** Appel, *"Technical Analysis: Power Tools for Active Investors"*, 2005, Prentice Hall.

- Publisher: https://www.pearson.com/en-us/subject-catalog/p/technical-analysis-power-tools-for-active-investors/P200000005747
- Amazon: https://www.amazon.com/Technical-Analysis-Power-Active-Investors/dp/0131479024

Gerald Appel introduced the Moving Average Convergence/Divergence oscillator in
the late 1970s. MACD is the difference between a fast (12) and slow (26) EMA;
the "signal line" is a 9-period EMA of MACD. Appel's timing rule — buy on a
bullish MACD/signal cross, sell on a bearish cross — is one of the most widely
used momentum-oscillator systems and the canonical implementation in every
major charting platform (TradingView, Bloomberg, MetaTrader).

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `crossover(ema(close,12) − ema(close,26), ema(ema(close,12) − ema(close,26), 9))` |
| Exit | `crossunder(...)` — same operands, opposite cross |

## Implementation notes

- Pure Pine expression — nested `ema` calls are supported by the engine.
- `required_lookback = 26` bars (the slow EMA window).
- Distinct from the repo's existing `macd_oscillator` (SMA 10/21 cross) and
  `macd_rsi` (MACD + RSI confirmation): this is the canonical Appel signal-line
  rule with no extra confirmation.

## Expected behaviour

- More responsive than the golden cross, so it catches turns earlier and
  trades more often.
- Whipsaws in sideways markets (the classic MACD failure mode) — churn without
  trend. Expected to shine in trending years (2021, 2023-24) and bleed in
  ranges (2022 bear, 2025 chop).
- Momentum-oscillator profile: moderate holding periods, balanced win rate.
