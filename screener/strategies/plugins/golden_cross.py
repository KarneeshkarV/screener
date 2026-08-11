"""Golden Cross: SMA50 over SMA200 long-term trend-following.

Paper: Han, Yang & Zhou, "A New Anomaly: The Cross-Sectional Profitability of
Technical Analysis", Journal of Financial and Quantitative Analysis 48(5),
2013. https://doi.org/10.1017/S0022109013000586

The paper studies the cross-sectional profitability of moving-average trading
rules and shows that stocks whose price is above their long-term moving average
(the 50/200 configuration being the canonical long-term MA pair, popularised
decades earlier in the Dow theory / "golden cross" folklore) earn positive
abnormal returns that reverse at longer horizons.

Signal (as-of bar ``t``):

    entry = crossover(sma(close, 50), sma(close, 200))   # golden cross
    exit  = crossunder(sma(close, 50), sma(close, 200))  # death cross

Entry fires exactly on the bar where the fast MA crosses above the slow MA; the
position is held until the fast MA crosses back below. This is a low-frequency
trend rule: expect long holding periods, deep drawdowns in bear phases, and the
bulk of returns concentrated in bull regimes.
"""

from __future__ import annotations

from screener.strategies.spec import register_expression_strategy

_FAST = 50
_SLOW = 200

register_expression_strategy(
    "golden_cross_50_200",
    entry=f"crossover(sma(close, {_FAST}), sma(close, {_SLOW}))",
    exit=f"crossunder(sma(close, {_FAST}), sma(close, {_SLOW}))",
    required_lookback=lambda: _SLOW,
)
