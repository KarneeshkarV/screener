"""One shared price-only feature and score layer.

Both consumers of a ranking score in this repo used to own their own copy of
the formula:

* the live screen wrote ``setup_score`` from a TradingView snapshot row;
* the rolling backtester wrote ``rank_score`` from OHLCV bars.

Two implementations of one named factor is a defect (see ``CONTEXT.md``), so
every *price-only* recipe now lives here exactly once and both paths consume it
through a thin adapter:

* backtest -> :mod:`screener.strategies.factor_adapter` (writes ``rank_score``)
* screen   -> :mod:`screener.scoring.bar_scores` (writes ``setup_score``)

Contract for anything registered here:

* Inputs are bars only: ``close``, ``high``, ``low``, ``volume`` and an
  optional ``benchmark_close``. No TradingView snapshot column ever reaches
  this layer, so nothing here can be contaminated by a vendor field that only
  carries *today's* value.
* Every function is causal: the value at bar ``t`` uses only data through
  ``t``. That is what makes one number valid both as today's screen score and
  as a historical backtest rank.
* NaN means *ineligible*, never "rank last". Recipes here must not ``fillna``;
  a name with too little history has no score and is dropped by both adapters.
  (The legacy snapshot recipes in ``screener.scoring.components`` keep their
  historical ``fillna(0)`` behaviour; only this unified path is strict.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

from screener._registry import Registry


@dataclass(frozen=True)
class BarFeatures:
    """The only inputs a shared price-only recipe may read.

    ``high`` / ``low`` / ``volume`` / ``benchmark_close`` are optional so a
    close-only recipe can be evaluated from a frame that carries nothing else.
    """

    close: pd.Series
    high: pd.Series | None = None
    low: pd.Series | None = None
    volume: pd.Series | None = None
    benchmark_close: pd.Series | None = None

    @classmethod
    def from_frame(
        cls,
        bars: pd.DataFrame,
        *,
        benchmark_close: pd.Series | None = None,
    ) -> BarFeatures:
        """Build features from an OHLCV frame with lowercase columns."""

        def _column(name: str) -> pd.Series | None:
            if name not in bars.columns:
                return None
            return pd.to_numeric(bars[name], errors="coerce").astype(float)

        close = _column("close")
        if close is None:
            raise KeyError("bar features require a 'close' column")
        aligned_benchmark = None
        if benchmark_close is not None:
            aligned_benchmark = (
                pd.to_numeric(benchmark_close, errors="coerce")
                .astype(float)
                .reindex(bars.index)
            )
        return cls(
            close=close,
            high=_column("high"),
            low=_column("low"),
            volume=_column("volume"),
            benchmark_close=aligned_benchmark,
        )


PriceScoreFn = Callable[[BarFeatures], pd.Series]
_RegisteredPriceScoreFn = TypeVar("_RegisteredPriceScoreFn", bound=PriceScoreFn)


@dataclass(frozen=True)
class PriceScoreSpec:
    """A price-only ranking recipe plus the history it needs."""

    name: str
    score_fn: PriceScoreFn
    required_lookback: int
    description: str = ""
    #: Column name the backtest adapter writes the raw score under, so an
    #: entry expression can gate on it (``rank_score`` is always written too).
    aux_column: str | None = None


registry: Registry[PriceScoreSpec] = Registry("price score")


def price_score(
    name: str,
    *,
    required_lookback: int,
    description: str = "",
    aux_column: str | None = None,
) -> Callable[[_RegisteredPriceScoreFn], _RegisteredPriceScoreFn]:
    """Register a causal, price-only ranking recipe under ``name``."""

    def _wrap(fn: _RegisteredPriceScoreFn) -> _RegisteredPriceScoreFn:
        registry.add(
            name,
            PriceScoreSpec(
                name=name,
                score_fn=fn,
                required_lookback=int(required_lookback),
                description=description,
                aux_column=aux_column,
            ),
        )
        return fn

    return _wrap


def get_price_score(name: str) -> PriceScoreSpec:
    """Look up one registered price-only recipe."""
    return registry.get(name)


def get_price_score_optional(name: str | None) -> PriceScoreSpec | None:
    """Look up one recipe, or ``None`` when the name is not registered."""
    return registry.get_optional(name)


def score_bars(
    spec: PriceScoreSpec,
    bars: pd.DataFrame,
    *,
    benchmark_close: pd.Series | None = None,
) -> pd.Series:
    """Evaluate ``spec`` over one symbol's bars, aligned to ``bars.index``."""
    if bars is None or bars.empty:
        return pd.Series(dtype=float)
    features = BarFeatures.from_frame(bars, benchmark_close=benchmark_close)
    return spec.score_fn(features).reindex(bars.index).astype(float)


def score_panel(
    spec: PriceScoreSpec,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    benchmark_close: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Evaluate ``spec`` across a panel, one causal series per symbol."""
    return {
        symbol: score_bars(spec, bars, benchmark_close=benchmark_close)
        for symbol, bars in bars_by_symbol.items()
        if bars is not None and not bars.empty
    }


def _register_recipes() -> None:
    from screener.factors import recipes  # noqa: F401


_register_recipes()

__all__ = [
    "BarFeatures",
    "PriceScoreFn",
    "PriceScoreSpec",
    "get_price_score",
    "get_price_score_optional",
    "price_score",
    "registry",
    "score_bars",
    "score_panel",
]
