# CCI Mean Reversion — Donald Lambert

**Strategy key:** `cci_reversion`

## Research source

**Article:** Lambert, *"Commodity Channel Index: Tool for Trading Cyclic Trends"*, Commodities magazine (now Futures), 1980.

- Modern reference: https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci

CCI measures how far the typical price deviates from its 20-period mean, normalized by the mean absolute deviation — a statistical z-score built without assuming normality. Lambert's ±100 thresholds mark one mean-deviation overshoots; his original framing was counter-trend at the extremes (buy < −100, sell > +100).

## Rule

| Leg | Signal |
|-----|--------|
| Entry | `cci_20 < −100` — typical price one mean-deviation below average |
| Exit | `cci_20 > +100` — typical price one mean-deviation above average |

with `cci = (tp − sma(tp,20)) / (0.015 · mean(|tp − sma(tp,20)|, 20))`, `tp = (high+low+close)/3`.

## Implementation notes

- Engine Pine has no absolute value / mean deviation, so `prepare_bars` precomputes the `cci_20` column.
- `required_lookback = 20`.
- Suggested sizing: mean reversion — 20 slots, short holds.

## Expected behaviour

- Mean reversion at statistical extremes — analogous to Bollinger but on typical price with absolute deviation scaling.
- High frequency; the ±100 band is tighter than Bollinger's ±2σ, so expect more trades and more noise than `bollinger_mean_reversion`.
