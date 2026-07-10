"""Constants for the research Pine runner."""

from screener.markets import MARKETS


BENCHMARKS = {name: market.benchmark for name, market in MARKETS.items()}
