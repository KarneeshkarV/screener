from __future__ import annotations

import click
import pytest

from screener._registry import Registry
from screener.backtester.cli_common import (
    build_slippage_model,
    parse_partial_exits,
    resolve_min_filters,
    resolve_strategy_exprs,
)
from screener.backtester.slippage import (
    CompositeSlippage,
    FixedBpsSlippage,
    HalfSpreadSlippage,
    VolumeImpactSlippage,
)
from screener.criteria import (
    CRITERIA,
    FilterCriteriaSelection,
    combine,
    registry,
    resolve_criteria,
)
from screener.strategies.expressions import NamedStrategy


def test_registry_exposes_snapshots_and_errors():
    reg: Registry[int] = Registry("thing")
    decorated = reg.register("one", group="core")(1)

    assert decorated == 1
    assert reg.get("one") == 1
    assert reg.get_optional("one") == 1
    assert reg.get_optional(None) is None
    assert reg.names() == ["one"]
    assert list(reg.items()) == [("one", 1)]
    assert list(reg) == ["one"]
    assert "one" in reg
    assert len(reg) == 1
    assert reg.meta("one") == {"group": "core"}
    assert reg.meta("missing") == {}
    assert reg.as_dict() == {"one": 1}

    with pytest.raises(ValueError, match="already has 'one'"):
        reg.add("one", 2)
    with pytest.raises(KeyError, match="Unknown thing 'missing'"):
        reg.get("missing")


def test_resolve_strategy_exprs_uses_named_strategy(monkeypatch):
    monkeypatch.setattr(
        "screener.strategies.expressions.resolve_strategy",
        lambda name: NamedStrategy(entry="close > ema(close, 20)", exit="close < open"),
    )

    assert resolve_strategy_exprs("trend", None, None) == (
        "close > ema(close, 20)",
        "close < open",
    )
    assert resolve_strategy_exprs("trend", "close > 0", "close < 0") == (
        "close > 0",
        "close < 0",
    )


def test_resolve_strategy_exprs_reports_usage_errors(monkeypatch):
    def fail(_: str) -> NamedStrategy:
        raise KeyError("not here")

    monkeypatch.setattr("screener.strategies.expressions.resolve_strategy", fail)

    with pytest.raises(click.UsageError, match="not here"):
        resolve_strategy_exprs("missing", None, None)
    with pytest.raises(click.UsageError, match="--entry"):
        resolve_strategy_exprs(None, None, None)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("fixed", FixedBpsSlippage),
        ("half-spread", HalfSpreadSlippage),
        ("vol-impact", VolumeImpactSlippage),
        ("composite", CompositeSlippage),
    ],
)
def test_build_slippage_model_variants(name, expected_type):
    model = build_slippage_model(name, 4, 2, 0.15)

    assert isinstance(model, expected_type)


def test_parse_partial_exits_and_min_filter_defaults():
    assert parse_partial_exits(()) == ()
    assert parse_partial_exits(("0.10:0.50", "0.20:0.25")) == (
        (0.10, 0.50),
        (0.20, 0.25),
    )
    with pytest.raises(click.UsageError, match="PROFIT_FRAC:SHARES_FRAC"):
        parse_partial_exits(("bad",))

    assert resolve_min_filters("us", None, None) == (1.0, 1_000.0)
    assert resolve_min_filters("india", 0, 0) == (None, None)
    assert resolve_min_filters("custom", None, None) == (None, None)
    assert resolve_min_filters("us", 5.0, 2_500.0) == (5.0, 2_500.0)


def test_criteria_registry_and_combine():
    assert registry.get("ema") is CRITERIA["ema"]
    assert registry.get_optional("does-not-exist") is None
    assert "garp" not in CRITERIA

    def first() -> list[int]:
        return [1, 2]

    def second() -> list[int]:
        return [3]

    assert combine(first, second)() == [1, 2, 3]


def test_criteria_selection_resolves_filter_names():
    filters = resolve_criteria(("ema", "value"))
    assert isinstance(filters, FilterCriteriaSelection)
    assert filters.names == ("ema", "value")
    assert filters.label == "ema+value"
    assert filters.filters


def test_all_filter_only_criteria_build_filter_lists():
    for name, fn in CRITERIA.items():
        filters = fn()

        assert isinstance(filters, list), name
        assert filters, name
