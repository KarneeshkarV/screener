# Keltner Channel Breakout — Chester Keltner

**Strategy key:** `keltner_breakout`

## Research source

**Book:** Keltner, *"How to Make Money in Commodities"*, 1960, Keltner Statistical Service (10-day MA ± ATR variant); the modern EMA20 ± 2×ATR(20) form is the StockCharts standard.

- https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

Keltner's channels predate Bollinger's by four decades and differ in construction: where Bollinger uses a standard-deviation width, Keltner widths by *true range* (volatility), which makes the bands track breakouts without widening infinitely in trends.

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `crossover(close, ema(close,20) + 2·atr(20))` — volatility breakout above the upper band |
| Exit | `crossunder(close, ema(close,20) − 2·atr(20))` — ride until the lower band |

## Implementation notes

- Pure Pine expression — `atr()` is natively supported.
- `required_lookback = 20`.
- Suggested sizing: trend following — 10 slots, longer holds.

## Expected behaviour

- Volatility-adjusted trend breakout: fewer false breaks than raw-price channels in high-vol names.
- Full-channel exit gives back gains in fast reversals (the classic Keltner critique) — expect lower hit rate, larger winners.
- Distinct from the repo's `bb_breakout` (std-width) and `bll_trading_range_break` (fixed 150-day price channel).
