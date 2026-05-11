"""Numpy/Pandas indicator helpers shared by strategy research code.

Implementations now live as individual plugin modules under
``screener.indicators.plugins`` and are registered via
``screener.indicators.registry``. This module keeps the legacy underscore-
prefixed names as re-exports so existing imports keep working.
"""

from __future__ import annotations

from screener.indicators.plugins.atr import atr as _atr
from screener.indicators.plugins.ema import ema as _ema
from screener.indicators.plugins.rma import rma as _rma
from screener.indicators.plugins.rsi import rsi as _rsi
from screener.indicators.plugins.sma import sma as _sma
from screener.indicators.plugins.stdev import stdev as _stdev
from screener.indicators.plugins.supertrend import supertrend_dir as _supertrend_dir

__all__ = [
    "_atr",
    "_ema",
    "_rma",
    "_rsi",
    "_sma",
    "_stdev",
    "_supertrend_dir",
]
