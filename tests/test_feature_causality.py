"""Causality guarantees for the trend-filter feature library.

The load-bearing test is :func:`test_every_feature_is_truncation_invariant`.
For every registered feature at every setting in its stability grid, it
recomputes the feature on history truncated at bar ``t`` and asserts the value
at ``t`` is unchanged. A centered window, a negative shift, a whole-sample fit,
a bidirectional filter or a Fourier reconstruction over the full sample all
change that value, so all of them fail here rather than silently inflating a
backtest.

The static checks below are cheaper and catch the same mistakes at the source
level, so a future edit gets a precise error instead of a numeric one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from screener.research.features import FeatureCtx, compute_features, registry

_FEATURE_DIR = pathlib.Path(registry["ema_slope"].fn.__code__.co_filename).parent

# Bars must comfortably exceed the largest `min_lookback` in the registry, or
# the truncation test would only ever compare NaN against NaN.
_N_BARS = 900
_SEED = 20260818


def _synthetic_bars(n: int = _N_BARS, seed: int = _SEED) -> pd.DataFrame:
    """Trending, noisy, gappy OHLCV: enough structure to make features defined.

    A pure random walk leaves several quality features degenerate, so the series
    carries a drift, a volatility regime shift and occasional shocks.
    """
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0008, -0.0004, n)
    vol = np.where(np.arange(n) < n // 2, 0.012, 0.022)
    shocks = rng.normal(0.0, 1.0, n) * vol + drift
    shocks[rng.integers(0, n, 12)] *= 6.0  # gap days
    close = 100.0 * np.exp(np.cumsum(shocks))
    index = pd.bdate_range("2019-01-01", periods=n)
    spread = close * 0.01
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.001, n)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.lognormal(12.0, 0.6, n),
        },
        index=index,
    )


def _synthetic_reference(bars: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(bars)
    steps = rng.normal(0.0004, 0.009, n)
    return pd.Series(500.0 * np.exp(np.cumsum(steps)), index=bars.index)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _synthetic_bars()


@pytest.fixture(scope="module")
def ctx(bars: pd.DataFrame) -> FeatureCtx:
    return FeatureCtx(
        bars=bars,
        benchmark=_synthetic_reference(bars, _SEED + 1),
        sector=_synthetic_reference(bars, _SEED + 2),
    )


def _truncated(ctx: FeatureCtx, upto: int) -> FeatureCtx:
    """The same context as if history had simply stopped at ``upto``."""
    return FeatureCtx(
        bars=ctx.bars.iloc[: upto + 1],
        benchmark=None if ctx.benchmark is None else ctx.benchmark.iloc[: upto + 1],
        sector=None if ctx.sector is None else ctx.sector.iloc[: upto + 1],
    )


# The bars checked for truncation invariance. Spread across the series so the
# test covers early warmup, the volatility regime change and the tail.
_PROBES = (620, 731, 848, 899)


@pytest.mark.parametrize("name", sorted(registry))
def test_every_feature_is_truncation_invariant(ctx: FeatureCtx, name: str) -> None:
    spec = registry[name]
    for params in spec.settings():
        full = spec.compute(ctx, **params)
        for probe in _PROBES:
            truncated = spec.compute(_truncated(ctx, probe), **params)
            expected = full.iloc[probe]
            actual = truncated.iloc[probe]
            if pd.isna(expected) and pd.isna(actual):
                continue
            assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
                f"{name} at {params} leaks future information: bar {probe} is "
                f"{actual!r} on truncated history but {expected!r} on full history"
            )


@pytest.mark.parametrize("name", sorted(registry))
def test_every_feature_is_defined_somewhere_and_finite(
    ctx: FeatureCtx, name: str
) -> None:
    """A feature that is all-NaN would pass the causality test vacuously."""
    spec = registry[name]
    values = spec.compute(ctx)
    defined = values.iloc[spec.min_lookback :].dropna()
    assert len(defined) > 0, f"{name} is never defined on {_N_BARS} bars"
    assert np.isfinite(defined.to_numpy(dtype=float)).all(), f"{name} emits inf"


@pytest.mark.parametrize("name", sorted(registry))
def test_feature_warmup_is_not_understated(ctx: FeatureCtx, name: str) -> None:
    """``min_lookback`` must really cover the warmup, or a backtest reads noise."""
    spec = registry[name]
    values = spec.compute(ctx)
    first_defined = values.notna().idxmax() if values.notna().any() else None
    assert first_defined is not None
    position = ctx.bars.index.get_loc(first_defined)
    assert position <= spec.min_lookback, (
        f"{name} first becomes defined at bar {position}, "
        f"beyond its declared min_lookback of {spec.min_lookback}"
    )


def test_every_feature_declares_a_stability_grid() -> None:
    """The research plan ranks on parameter stability, so a grid is mandatory."""
    missing = [spec.name for spec in registry.values() if spec.params and not spec.grid]
    assert not missing, f"features with parameters but no stability grid: {missing}"


def test_compute_features_skips_what_it_cannot_compute(bars: pd.DataFrame) -> None:
    # No benchmark and no sector: relative features are absent, not NaN, so a
    # caller can tell "not applicable" from "not yet defined".
    frame = compute_features(FeatureCtx(bars=bars))
    assert "relative_momentum" not in frame.columns
    assert "sector_relative_momentum" not in frame.columns
    assert "efficiency_ratio" in frame.columns


# ── static bans ──────────────────────────────────────────────────────


_BANNED = {
    "center=True": "centered rolling windows read future bars",
    "filtfilt": "filtfilt runs a backward pass over the whole series",
    "savgol_filter": "scipy's savgol_filter is centered; fit a trailing window",
    "shift(-": "a negative shift reads a later bar",
    "[::-1]": "reversing the series is a backward pass in disguise",
    "bfill": "backfill propagates a later value into an earlier bar",
}


def _executable_source(path: pathlib.Path) -> str:
    """Source with comments and string literals removed.

    The module docstrings name the very idioms being banned, so a raw text
    search would flag the documentation that explains the rule. Tokenizing and
    dropping comments and strings leaves only code that actually runs.
    """
    import tokenize

    kept: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept)


@pytest.mark.parametrize("path", sorted(_FEATURE_DIR.glob("*.py")))
def test_feature_sources_contain_no_lookahead_idioms(path: pathlib.Path) -> None:
    code = _executable_source(path)
    for idiom, reason in _BANNED.items():
        # Tokenization separates operators, so compare against a whitespace-free
        # form as well as the literal idiom.
        squashed = code.replace(" ", "")
        needle = idiom.replace(" ", "")
        assert needle not in squashed, f"{path.name} uses {idiom!r}: {reason}"
