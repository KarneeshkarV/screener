"""Earnings-drift backtest module.

Entry: buy at close of E-N (N = days_before, default 1).
Exit:  sell at close of earnings day E.
Sentiment strategies filter which earnings events are traded.

PEAD mode (``earnings-pead``): select events by EPS-surprise threshold,
enter at the next trading day's open, hold N trading days (default 40).
"""
