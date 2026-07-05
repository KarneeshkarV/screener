# Live Illustrative Backtest (real market data)

**Research, not financial advice. Network-dependent and NOT pinned by tests** —
unlike `strategy_report.md` (deterministic/synthetic), the numbers below come
from live yfinance OHLCV and will drift over time.

## Setup

- **Universe:** 35 curated large/liquid US names
  (`AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, JNJ, PG, XOM, KO, PEP, WMT, HD, MA,
  V, UNH, DIS, CSCO, INTC, VZ, T, MRK, PFE, CVX, ABT, MCD, NKE, ORCL, IBM, WFC,
  BA, CAT, GE, MMM`).
- **Window:** trailing 3 years (`--years 3`), daily rolling backtest.
- **Portfolio:** top 5 equal-weight slots, 21-day hold, benchmark `SPY`,
  no commission/slippage.
- **Captured:** 2026-06-19.

### Known biases (why this is "illustrative" only)

- **Survivorship bias:** the universe is today's survivors, not the
  point-in-time membership of any index. Use
  `backtest-rolling --universe sp500 --point-in-time` for a survivorship-aware
  run (heavier: ~500 names).
- **Small, curated universe** → concentrated, not a broad factor portfolio.
- Live data revisions / splits can change results between runs.

## Results vs SPY (benchmark total return +77.3%)

| Strategy | Sharpe | Total Return | Sortino | CAGR | Max DD | Trades | Hit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12-1 Momentum | 1.48 | +110.0% | 2.26 | 28.2% | -15.7% | 175 | 61% |
| Low Volatility | 0.77 | +128.8% | 4.35 | 32.0% | -12.3% | 175 | 51% |
| Momentum + Low-Vol Combo | 0.76 | +127.8% | 4.15 | 31.8% | -11.1% | 175 | 49% |

All three beat the `SPY` benchmark (+77.3%). The combo achieves the shallowest
drawdown (-11.1%) — the volatility brake behaving as the literature predicts —
while momentum has the highest Sharpe.

## Reproduce (numbers will differ — live data)

```bash
TICKERS="AAPL,MSFT,AMZN,GOOGL,META,NVDA,JPM,JNJ,PG,XOM,KO,PEP,WMT,HD,MA,V,UNH,DIS,CSCO,INTC,VZ,T,MRK,PFE,CVX,ABT,MCD,NKE,ORCL,IBM,WFC,BA,CAT,GE,MMM"
uv run screener backtest-rolling -m us --strategy momentum_12_1   --tickers "$TICKERS" --top 5 --hold 21 --years 3 --min-price 0 --min-avg-dollar-volume 0
uv run screener backtest-rolling -m us --strategy low_volatility  --tickers "$TICKERS" --top 5 --hold 21 --years 3 --min-price 0 --min-avg-dollar-volume 0
uv run screener backtest-rolling -m us --strategy mom_lowvol_combo --tickers "$TICKERS" --top 5 --hold 21 --years 3 --min-price 0 --min-avg-dollar-volume 0
```
