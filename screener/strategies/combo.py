"""Generic multi-factor combiner.

Supports strategy names of the form::

    combo:momentum_12_1=0.6,low_volatility=0.4

Each component must be a registered expression strategy that writes
``rank_score`` via ``prepare_bars``. Component scores are z-scored
cross-sectionally per day, then combined as a weighted sum into the final
``rank_score``.

Resolved dynamically (not a static registry entry) so arbitrary weight mixes
work without pre-registering every combination. See :func:`resolve_combo_spec`
and :func:`is_combo_strategy`.
"""

from __future__ import annotations

import math
import re
from typing import Sequence, cast

import numpy as np
import pandas as pd

from screener.strategies.spec import (
    ExpressionStrategySpec,
    LookbackFn,
    PrepareBarsFn,
    PrepareCtx,
    discover_plugins,
    registry,
)

_COMBO_PREFIX = "combo:"
# name=weight pairs separated by commas; names are registry keys (no commas/'=').
_COMPONENT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def is_combo_strategy(name: str | None) -> bool:
    return bool(name) and str(name).startswith(_COMBO_PREFIX)


def parse_combo_spec(name: str) -> list[tuple[str, float]]:
    """Parse ``combo:factor=w,...`` into ``[(factor, weight), ...]``.

    Raises ``ValueError`` on malformed input or invalid weights.
    """
    if not is_combo_strategy(name):
        raise ValueError(f"not a combo strategy name: {name!r}")
    body = name[len(_COMBO_PREFIX) :].strip()
    if not body:
        raise ValueError("combo strategy requires at least one component")
    components: list[tuple[str, float]] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        match = _COMPONENT_RE.match(part)
        if not match:
            raise ValueError(
                f"invalid combo component {part!r}; expected name=weight "
                f"(e.g. momentum_12_1=0.6)"
            )
        factor_name, weight_s = match.group(1), match.group(2)
        weight = float(weight_s)
        if not math.isfinite(weight):
            raise ValueError(f"combo weight for {factor_name!r} must be finite")
        components.append((factor_name, weight))
    if not components:
        raise ValueError("combo strategy requires at least one component")
    return components


def validate_combo_components(
    components: Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Ensure every component exists and can produce a rank_score."""
    discover_plugins()
    validated: list[tuple[str, float]] = []
    for factor_name, weight in components:
        if not math.isfinite(weight):
            raise ValueError(f"combo weight for {factor_name!r} must be finite")
        if is_combo_strategy(factor_name):
            raise ValueError("nested combo strategies are not supported")
        spec = registry.get_optional(factor_name)
        if spec is None:
            raise ValueError(
                f"unknown combo component {factor_name!r}. "
                f"Known: {sorted(registry.names())}"
            )
        if getattr(spec, "prepare_bars", None) is None:
            raise ValueError(
                f"combo component {factor_name!r} has no prepare_bars hook "
                "(cannot produce rank_score)"
            )
        if getattr(spec, "entry", None) is None:
            raise ValueError(f"combo component {factor_name!r} has no entry expression")
        validated.append((factor_name, weight))
    if not validated:
        raise ValueError("combo strategy requires at least one component")
    return validated


def cross_sectional_zscore(score_mat: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row (day) across columns (tickers); NaNs ignored.

    Days with population std 0 (or a single non-NaN name) become 0 for
    non-NaN inputs; input NaNs stay NaN.
    """
    if score_mat.empty:
        return score_mat
    mu = score_mat.mean(axis=1)
    # Population std so constant / single-name rows yield sigma 0.
    sigma = score_mat.sub(mu, axis=0).pow(2).mean(axis=1).pow(0.5)
    centered = score_mat.sub(mu, axis=0)
    # Divide only where sigma > 0; zero-sigma rows become NaN then 0 for
    # valid inputs, while original NaN scores stay NaN.
    z = centered.div(sigma.where(sigma > 0), axis=0)
    need_zero = score_mat.notna() & z.isna()
    return z.mask(need_zero, 0.0)


def combine_rank_scores(
    component_scores: Sequence[tuple[pd.DataFrame, float]],
) -> pd.DataFrame:
    """Weighted sum of per-day cross-sectional z-scores of each component.

    A name only receives a blended score on days where every component is
    defined, so a missing leg does not silently reweight the others.
    """
    if not component_scores:
        raise ValueError("at least one component score matrix is required")
    # Align all component matrices to a common index/columns union.
    all_index = component_scores[0][0].index
    all_columns = component_scores[0][0].columns
    for score_mat, _weight in component_scores[1:]:
        all_index = all_index.union(score_mat.index)
        all_columns = all_columns.union(score_mat.columns)

    blended = pd.DataFrame(0.0, index=all_index, columns=all_columns)
    defined = pd.DataFrame(True, index=all_index, columns=all_columns)
    for score_mat, weight in component_scores:
        aligned = score_mat.reindex(index=all_index, columns=all_columns)
        z = cross_sectional_zscore(aligned.astype(float))
        blended = blended + z.fillna(0.0) * float(weight)
        defined = defined & aligned.notna()
    return blended.where(defined, np.nan)


def _component_score_matrix(
    prepared: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for tv, bars in prepared.items():
        if bars is None or bars.empty or "rank_score" not in bars.columns:
            continue
        cols[tv] = bars["rank_score"].astype(float)
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def make_combo_prepare(components: Sequence[tuple[str, float]]) -> PrepareBarsFn:
    validated = list(components)

    def _prepare(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
        component_mats: list[tuple[pd.DataFrame, float]] = []
        prepared_by_factor: list[dict[str, pd.DataFrame]] = []
        for factor_name, weight in validated:
            spec = cast(ExpressionStrategySpec, registry.get(factor_name))
            assert spec.prepare_bars is not None
            child_ctx = PrepareCtx(
                market=ctx.market,
                benchmark=ctx.benchmark,
                bars_by_tv={tv: bars for tv, bars in ctx.bars_by_tv.items()},
                price_panel=ctx.price_panel,
                tv_symbols=ctx.tv_symbols,
                start=ctx.start,
                end=ctx.end,
                fetcher=ctx.fetcher,
                warnings=ctx.warnings,
            )
            prepared = spec.prepare_bars(child_ctx)
            prepared_by_factor.append(prepared)
            score_mat = _component_score_matrix(prepared)
            if score_mat.empty:
                ctx.warnings.append(
                    f"combo component {factor_name!r} produced no rank_score values"
                )
            component_mats.append((score_mat, weight))

        blended = combine_rank_scores(component_mats)
        out: dict[str, pd.DataFrame] = {}
        for tv, bars in ctx.bars_by_tv.items():
            if bars is None or bars.empty:
                out[tv] = bars
                continue
            frame = bars.copy()
            # Merge aux columns from every component so multi-factor entry
            # gates (e.g. mom_12_1 and vol_252) still resolve.
            for prepared in prepared_by_factor:
                aux = prepared.get(tv)
                if aux is None or aux.empty:
                    continue
                for col in aux.columns:
                    if col == "rank_score":
                        continue
                    if col not in frame.columns:
                        frame[col] = aux[col]
            if tv in blended.columns:
                frame["rank_score"] = blended[tv].reindex(frame.index)
            else:
                frame["rank_score"] = np.nan
            out[tv] = frame
        return out

    return _prepare


def make_combo_lookback(components: Sequence[tuple[str, float]]) -> LookbackFn:
    validated = list(components)

    def _lookback() -> int:
        floor = 0
        for factor_name, _weight in validated:
            spec = cast(ExpressionStrategySpec, registry.get(factor_name))
            if spec.required_lookback is not None:
                floor = max(floor, int(spec.required_lookback()))
        return floor

    return _lookback


def make_combo_entry(components: Sequence[tuple[str, float]]) -> str:
    validated = list(components)
    parts: list[str] = []
    for factor_name, _weight in validated:
        spec = cast(ExpressionStrategySpec, registry.get(factor_name))
        parts.append(f"({spec.entry})")
    return " and ".join(parts)


def resolve_combo_spec(name: str) -> ExpressionStrategySpec:
    """Build an :class:`ExpressionStrategySpec` for a ``combo:...`` name."""
    components = validate_combo_components(parse_combo_spec(name))
    return ExpressionStrategySpec(
        name=name,
        entry=make_combo_entry(components),
        exit=None,
        prepare_bars=make_combo_prepare(components),
        required_lookback=make_combo_lookback(components),
    )
