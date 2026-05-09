"""Canonical strategy package."""

from screener.strategies.registry import STRATEGIES, get_strategy, iter_strategies

__all__ = ["STRATEGIES", "get_strategy", "iter_strategies"]
