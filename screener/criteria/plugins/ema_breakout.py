"""EMA bullish stack + 52-week breakout — composition of two criteria."""

from __future__ import annotations

from typing import cast

from screener.criteria import combine, criterion
from screener.criteria.plugins.breakout import near_52w_breakout
from screener.criteria.plugins.ema import ema_bullish_stack


@criterion("ema_breakout")
def ema_with_breakout() -> list:
    # combine returns CriterionFn (Callable[..., Any]); the call yields a list.
    return cast(list, combine(ema_bullish_stack, near_52w_breakout)())
