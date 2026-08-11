"""Connors RSI-2 mean reversion: buy deep RSI(2) oversold, sell overbought.

Source: Connors & Alvarez, "Short Term Trading Strategies That Work", 2009
(Quantitative Research, Connors Research LLC). The RSI-2 rule is the flagship
strategy of the book: RSI(2) is so reactive that readings below 5 mark genuine
2-day capitulation, and the bounce back toward 60 is historically profitable
when applied to liquid names.

Book rule:

    entry = rsi(close, 2) < 5                       # deep 2-day oversold
    exit  = rsi(close, 2) > 60                      # 2-day overbought

``connors_rsi2_bull`` adds the book's bull-regime refinement: only take longs
while price holds above the 200-day SMA (stay out of bear markets, where
oversold bounces fail):

    entry = rsi(close, 2) < 5 and close > sma(close, 200)
    exit  = rsi(close, 2) > 60

Both are mean-reversion strategies: they need liquid, mean-reverting names and
suffer when a stock keeps falling (RSI(2) can stay < 5 for days). Expected
behaviour: high trade frequency, high win rate, small average wins, and a
left tail of losing streaks in sustained downtrends.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_ENTRY = 5
_EXIT = 60
_BULL_MA = 200

register_expression_strategy(
    "connors_rsi2",
    entry=f"rsi(close, 2) < {_ENTRY}",
    exit=f"rsi(close, 2) > {_EXIT}",
)

register_expression_strategy(
    "connors_rsi2_bull",
    entry=f"rsi(close, 2) < {_ENTRY} and close > sma(close, {_BULL_MA})",
    exit=f"rsi(close, 2) > {_EXIT}",
)
