"""Eligibility is declared on the recipe, not restated per adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from screener.factors import (
    eligible_mask,
    entry_gate_expression,
    get_price_score,
)
from screener.strategies.plugins.momentum_12_1 import ENTRY_PURE, ENTRY_TREND


def test_momentum_12_1_declares_a_positive_raw_floor() -> None:
    spec = get_price_score("momentum_12_1")
    assert spec.aux_column == "mom_12_1"
    assert spec.eligible_above == 0.0
    assert entry_gate_expression(spec) == "mom_12_1 > 0"
    assert ENTRY_PURE == "mom_12_1 > 0"
    assert ENTRY_TREND.startswith("mom_12_1 > 0 and ")


def test_eligible_mask_drops_nan_and_non_positive_momentum() -> None:
    spec = get_price_score("momentum_12_1")
    scores = pd.Series([0.4, 0.0, -0.1, float("nan")], index=list("ABCD"))
    assert eligible_mask(spec, scores).tolist() == [True, False, False, False]


def test_entry_gate_expression_refuses_a_recipe_with_no_floor() -> None:
    spec = get_price_score("momentum_12_1")
    bare = spec.__class__(
        name=spec.name,
        score_fn=spec.score_fn,
        required_lookback=spec.required_lookback,
        description=spec.description,
        aux_column=None,
        eligible_above=None,
    )
    with pytest.raises(ValueError, match="declares no entry gate"):
        entry_gate_expression(bare)
