---
name: techofundo
description: Use when the user invokes /use techofundo or asks for technofundamental stock or portfolio analysis with stop-loss and take-profit levels. Inputs may be plain tickers, comma-separated tickers, or portfolio entries like SYMBOL:ENTRY:MARKET_VALUE. The output should combine technical and fundamental evidence from the repo/providers and explain the logic behind stop-loss and take-profit levels.
---

# Techofundo

Analyze the stock or portfolio input the user supplies (plain tickers, comma-separated tickers, or `SYMBOL:ENTRY:MARKET_VALUE`).

Produce an accurate technofundamental analysis with stop-loss and take-profit levels. Treat this as a current-market, high-accuracy task.

## Workflow

1. Parse the input as one or more holdings. Accept plain tickers, comma-separated tickers, or `SYMBOL:ENTRY:MARKET_VALUE`. If quantity, average price, timeframe, or risk style is supplied, use it. If not supplied, make a conservative swing-trade assumption and state it.
2. Read the project docs and inspect the codebase enough to identify available data providers, screeners, indicators, backtesters, FMP/yfinance integrations, and portfolio tooling. Use the repository freely: run existing CLI commands, import modules in small scripts, or add temporary/ad-hoc analysis code if useful. Do not assume any single helper or command is mandatory.
3. Verify current/recent market data. Prefer the repo’s configured providers and caches. Use FMP when `FMP_API_KEY` is available, especially for quote, valuation, rating, profile, analyst/target, insider, or fundamental context. Fall back to yfinance/repo OHLCV when FMP is unavailable. Browse only when needed for current facts, and cite links if external web data materially affects the answer.
4. Run a technical pass:
   - Current price, entry P&L when entry is provided.
   - EMA/SMA trend, RSI, ATR/volatility, SuperTrend or equivalent if available.
   - Relative strength versus an appropriate benchmark, usually `SPY` for US.
   - Support/resistance from recent lows/highs, breakout/failure zones, volume or buildup confirmation.
   - Backtest or strategy signal checks if the repo makes that easy.
5. Run a fundamental/context pass:
   - Valuation/rating/financial quality from FMP or repo sources when available.
   - Insider/promoter/ownership signal when relevant.
   - Earnings/event risk if discoverable.
   - Any contradiction between technicals and fundamentals.
6. Convert the evidence into levels:
   - A primary close-based stop-loss.
   - A hard invalidation stop for gaps or strict risk control.
   - Take-profit 1 for partial exit.
   - Take-profit 2 for larger trim or final target.
   - Position action: hold, trim, sell bounce, exit, or add only above a reclaim level.

## Output

Keep the answer concise but decision-grade:

- State data date/time and data sources used.
- Provide a table with: Symbol, Latest, Entry, P&L%, Trend/Score, Stop, Hard Stop, TP1, TP2, Action.
- Under the table, give short per-symbol reasoning explaining why those levels were chosen.
- End with portfolio-level priorities: strongest hold, weakest/risk cut, where to take partial profits first.
- Include a brief “not financial advice” note, but do not let the disclaimer replace the analysis.

Be explicit when data is stale, missing, or conflicting. Do not invent unavailable fundamentals or analyst targets.
