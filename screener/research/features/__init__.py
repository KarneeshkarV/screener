"""Causal, point-in-time feature library for trend-filter research.

Importing this package registers every feature. Use :data:`registry` to look
one up by name, or :func:`compute_features` to build a frame for one ticker.

Nothing here reads a future bar. That claim is enforced by
``tests/test_feature_causality.py``, which recomputes every feature on
truncated history and compares; see :mod:`.base` for the rules that follow.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from screener.research.features import (  # noqa: F401  (import registers features)
    acceleration,
    experimental,
    liquidity,
    quality,
    relative,
    trend,
    volatility,
)
from screener.research.features.base import (
    Category,
    FeatureCtx,
    FeatureSpec,
    feature,
    registry,
)
from screener.research.features.relative import (
    cross_sectional_ranks,
    rank_consistency,
)


def feature_names(category: Category | None = None) -> tuple[str, ...]:
    """Registered feature names, optionally restricted to one category."""
    if category is None:
        return tuple(registry)
    return tuple(spec.name for spec in registry.by_category(category))


def compute_features(
    ctx: FeatureCtx,
    names: tuple[str, ...] | None = None,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compute features for one ticker into a ``date x feature`` frame.

    Features whose inputs are unavailable (a benchmark-relative feature with no
    benchmark in ``ctx``) are skipped rather than filled with NaN, so a caller
    can tell "not applicable" from "not yet defined".
    """
    overrides = overrides or {}
    selected = names or tuple(registry)
    columns: dict[str, pd.Series] = {}
    for name in selected:
        spec = registry[name]
        if spec.needs_benchmark and ctx.benchmark is None:
            continue
        if spec.needs_sector and ctx.sector is None:
            continue
        columns[name] = spec.compute(ctx, **overrides.get(name, {}))
    return pd.DataFrame(columns, index=ctx.bars.index)


__all__ = [
    "Category",
    "FeatureCtx",
    "FeatureSpec",
    "compute_features",
    "cross_sectional_ranks",
    "feature",
    "feature_names",
    "rank_consistency",
    "registry",
]
