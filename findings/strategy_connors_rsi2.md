# Connors RSI-2 Mean Reversion

**Strategy keys:** `connors_rsi2`, `connors_rsi2_bull`

## Research source

**Book:** Connors & Alvarez, *"Short Term Trading Strategies That Work"*, 2009, Connors Research LLC.

- Publisher: https://www.connorsresearch.com/products/short-term-trading-strategies-that-work/
- Amazon: https://www.amazon.com/Short-Term-Trading-Strategies-That-Work/dp/0615281788

The RSI-2 rule is the flagship of the Connors Research family and one of the
most widely backtested short-term mean-reversion rules in retail quant circles.
RSI(2) is extremely reactive — a reading below 5 means the stock has fallen for
two straight sessions into genuine short-term capitulation. Connors' published
equity curves show the bounce back toward RSI 60 being profitable across US
equities, ETFs, and futures, especially when combined with a bull-regime
filter.

## Rule

| Variant | Entry | Exit |
|---------|-------|------|
| `connors_rsi2` | `rsi(close, 2) < 5` | `rsi(close, 2) > 60` |
| `connors_rsi2_bull` | `rsi(close, 2) < 5 and close > sma(close, 200)` | `rsi(close, 2) > 60` |

The `_bull` variant is the book's bear-market protection: only buy oversold
bounces while price holds above the 200-day SMA.

## Implementation notes

- Pure Pine expression; RSI with `length=2` is fully supported by the engine.
- No `required_lookback` beyond the RSI warmup (2 bars + Wilder seed).
- Suggested holding semantics: mean reversion — short `--hold` (default 20
  trading days is fine).

## Expected behaviour

- High trade frequency, high win rate, small average win.
- Fails in sustained downtrends (falling knife: RSI(2) can stay < 5 for many
  sessions). The `_bull` variant is expected to dominate in bear-heavy windows
  (e.g., 2022) and lag in strong bull years where the 200-day filter is rarely
  the binding constraint.
- Needs liquid names — a liquidity filter is recommended (the backtest's
  default min-ADV filter covers this).
