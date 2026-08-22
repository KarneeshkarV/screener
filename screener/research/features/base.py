"""Registry and shared contract for causal, point-in-time trend features.

Every feature here is a function of one ticker's OHLCV history that returns a
Series aligned to the input index. The contract is deliberately narrow, because
the whole value of this library is that a feature value at bar ``t`` is
computable from bars ``<= t`` and nothing else.

Causality is enforced, not documented. ``tests/test_feature_causality.py``
re-evaluates every registered feature against history truncated at bar ``t`` and
asserts the value at ``t`` is unchanged. A centered window, a forward shift, a
``filtfilt``-style backward pass or a whole-sample fit all fail that test.

Rules that follow from it, for anyone adding a feature:

- ``rolling(n)`` and ``ewm(adjust=False)`` are trailing and therefore fine.
  ``rolling(n, center=True)`` is not.
- ``shift(k)`` with ``k > 0`` looks back and is fine. ``shift(-k)`` is not.
- ``scipy.signal.savgol_filter`` and ``filtfilt`` are centered/bidirectional.
  Do not call them; fit the local polynomial on a trailing window instead.
- Any transform whose output at ``t`` depends on the length of the input (a
  whole-sample FFT, a whole-sample detrend, a global normalization) is banned.
  Normalize against a trailing window instead.

Each spec carries a ``grid`` of alternative parameter settings. That is not
decoration: the research plan ranks a feature on whether it works across a
broad parameter range, so the grid is the unit of the stability test and every
feature must declare one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Category = Literal[
    "trend",
    "quality",
    "volatility",
    "acceleration",
    "relative",
    "liquidity",
    "experimental",
]


@dataclass(frozen=True)
class FeatureCtx:
    """Everything a feature may read. Nothing else is in scope.

    ``bars`` is one ticker's OHLCV, ascending, with a unique DatetimeIndex.
    ``benchmark`` and ``sector`` are close series already reindexed onto
    ``bars.index`` by the caller, so a feature never has to align anything and
    can never accidentally align against a future bar.
    """

    bars: pd.DataFrame
    benchmark: pd.Series | None = None
    sector: pd.Series | None = None

    @property
    def close(self) -> pd.Series:
        return self.bars["close"].astype(float)

    @property
    def log_close(self) -> pd.Series:
        # Guard non-positive prices: a corrupt zero would otherwise poison every
        # log-based slope with -inf rather than a clean NaN.
        close = self.close
        return pd.Series(np.log(close.where(close > 0.0)), index=close.index)

    @property
    def returns(self) -> pd.Series:
        return self.close.pct_change()

    @property
    def log_returns(self) -> pd.Series:
        return self.log_close.diff()


FeatureFn = Callable[..., pd.Series]


@dataclass(frozen=True)
class FeatureSpec:
    """One named feature: how to compute it, and how to stress its parameters."""

    name: str
    fn: FeatureFn
    category: Category
    doc: str
    params: dict[str, Any] = field(default_factory=dict)
    # Alternative settings used by the parameter-stability test. A feature that
    # only works at one point of this grid is reported as likely overfit.
    grid: tuple[dict[str, Any], ...] = ()
    # Bars of history needed before the feature is defined. Used to size warmup
    # so a backtest never reads a value the feature could not really have had.
    min_lookback: int = 0
    needs_benchmark: bool = False
    needs_sector: bool = False
    # True when a larger value means "more of the thing the name says".
    higher_is_stronger: bool = True

    def compute(self, ctx: FeatureCtx, **overrides: Any) -> pd.Series:
        params = {**self.params, **overrides}
        out = self.fn(ctx, **params)
        out.name = self.name
        return out

    def settings(self) -> tuple[dict[str, Any], ...]:
        """Default settings first, then the rest of the stability grid."""
        seen: list[dict[str, Any]] = [dict(self.params)]
        for candidate in self.grid:
            merged = {**self.params, **candidate}
            if merged not in seen:
                seen.append(merged)
        return tuple(seen)


class FeatureRegistry(Mapping[str, FeatureSpec]):
    """``name -> FeatureSpec``, populated by the ``@feature`` decorator."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def add(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self._specs:
            raise ValueError(f"feature already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def __getitem__(self, key: str) -> FeatureSpec:
        return self._specs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._specs))

    def __len__(self) -> int:
        return len(self._specs)

    def by_category(self, category: Category) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.values() if s.category == category)


registry = FeatureRegistry()


def feature(
    name: str,
    *,
    category: Category,
    doc: str,
    params: dict[str, Any] | None = None,
    grid: tuple[dict[str, Any], ...] = (),
    min_lookback: int = 0,
    needs_benchmark: bool = False,
    needs_sector: bool = False,
    higher_is_stronger: bool = True,
) -> Callable[[FeatureFn], FeatureFn]:
    """Register ``fn`` as a causal feature. Returns ``fn`` unchanged."""

    def decorate(fn: FeatureFn) -> FeatureFn:
        registry.add(
            FeatureSpec(
                name=name,
                fn=fn,
                category=category,
                doc=doc,
                params=dict(params or {}),
                grid=grid,
                min_lookback=min_lookback,
                needs_benchmark=needs_benchmark,
                needs_sector=needs_sector,
                higher_is_stronger=higher_is_stronger,
            )
        )
        return fn

    return decorate


# ── shared causal primitives ─────────────────────────────────────────
#
# These exist so features do not each hand-roll a rolling regression and drift
# apart on edge cases (short windows, NaN warmup, zero variance).


def rolling_ols_slope(
    y: pd.Series, window: int, *, with_stats: bool = False
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Trailing OLS of ``y`` on ``0..window-1``.

    Returns ``(slope, t_stat, r_squared)``. The regressor is a fixed ramp, so
    the design matrix is constant and every moment can be accumulated with
    rolling sums: no per-bar ``polyfit``, and the window is strictly trailing.
    ``t_stat`` and ``r_squared`` are NaN when the fit is degenerate (zero
    residual variance or zero total variance) rather than raising.
    """
    if window < 3:
        raise ValueError("rolling_ols_slope needs window >= 3")
    n = float(window)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    s_xx = float(((x - x_mean) ** 2).sum())

    roll = y.rolling(window, min_periods=window)
    y_sum = roll.sum()
    y_mean = y_sum / n
    # sum(x_i * y_i) via a dot product against the fixed ramp.
    xy = y.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(x, values)), raw=True
    )
    s_xy = xy - x_mean * y_sum
    slope = s_xy / s_xx

    if not with_stats:
        empty = pd.Series(np.nan, index=y.index)
        return slope, empty, empty

    y_sq_sum = (y * y).rolling(window, min_periods=window).sum()
    s_yy = y_sq_sum - n * (y_mean**2)
    explained = slope * s_xy
    residual = s_yy - explained
    # Clamp tiny negative residuals from floating-point cancellation.
    residual = residual.where(residual > 0.0, 0.0)
    r_squared = (explained / s_yy).where(s_yy > 0.0)
    sigma_sq = residual / (n - 2.0)
    se = np.sqrt(sigma_sq / s_xx)
    t_stat = (slope / se).where(se > 0.0)
    return slope, t_stat, r_squared


def rolling_percentile(x: pd.Series, window: int) -> pd.Series:
    """Where the current value sits inside its own trailing window, in [0, 1].

    Uses the window ending at the current bar and includes the current value,
    which is information available at ``t``.
    """
    return x.rolling(window, min_periods=window).apply(
        lambda values: float((values <= values[-1]).mean()), raw=True
    )


def zscore(x: pd.Series, window: int) -> pd.Series:
    """Trailing z-score. NaN where the trailing standard deviation is zero."""
    mean = x.rolling(window, min_periods=window).mean()
    std = x.rolling(window, min_periods=window).std(ddof=1)
    return ((x - mean) / std).where(std > 0.0)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """``numerator / denominator``, NaN wherever the denominator is <= 0."""
    return (numerator / denominator).where(denominator > 0.0)


__all__ = [
    "Category",
    "FeatureCtx",
    "FeatureRegistry",
    "FeatureSpec",
    "feature",
    "registry",
    "rolling_ols_slope",
    "rolling_percentile",
    "safe_ratio",
    "zscore",
]
