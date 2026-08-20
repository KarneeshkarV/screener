# Stochastic Oscillator %K/%D Cross — George Lane

**Strategy key:** `stochastic_cross`

## Research source

**Origin:** George C. Lane, stochastic oscillator seminars (1950s-60s) — no formal paper; the canonical modern reference is StockCharts chartSchool.

- https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

Lane's oscillator is one of the three classic momentum oscillators (with RSI and MACD). %K measures where today's close sits inside the recent high-low range; %D is its smoothed version. Lane's original rule: buy when %K crosses above %D in the oversold zone, sell on the cross below %D in the overbought zone — mean reversion within a momentum envelope.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `crossover(%K, %D) and %K < 30` — bullish cross in oversold |
| Exit | `crossunder(%K, %D) and %K > 70` — bearish cross in overbought |

with `%K = 100·(close − lowest(low,14)) / (highest(high,14) − lowest(low,14))` and `%D = sma(%K, 3)`.

## Implementation notes

- Pure Pine expression (highest/lowest/sma all supported).
- `required_lookback = 14`.
- Suggested sizing: momentum oscillator — 15 slots, short holds.

## Expected behaviour

- Catches pullback entries inside uptrends and exits into strength.
- Whipsaws in choppy markets; %K/%D cross frequently in ranges.
- Complements the repo's existing RSI rules: same family, different construction (range position vs smoothed momentum).
