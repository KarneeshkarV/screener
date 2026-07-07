from __future__ import annotations

import types
from datetime import date
from types import SimpleNamespace
from unittest.mock import ANY

import click
import pytest

from screener._registry import Registry, autodiscover
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
    CriteriaSelectionError,
    FilterCriteriaSelection,
    PipelineCriteriaSelection,
    combine,
    definitions,
    filter_criteria,
    is_pipeline,
    pipeline_criteria,
    registry,
    resolve_criteria,
)
from screener.criteria.plugins import garp, obv_trend, promoter_buys, rs_breakout
from screener.criteria.plugins import unusual_volume, vol_breakout
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


def test_autodiscover_rejects_non_package():
    with pytest.raises(TypeError, match="expects a package"):
        autodiscover(types.ModuleType("plain_module"))


def test_resolve_strategy_exprs_uses_named_strategy(monkeypatch):
    monkeypatch.setattr(
        "screener.backtester.strategies.resolve_strategy",
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

    monkeypatch.setattr("screener.backtester.strategies.resolve_strategy", fail)

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


def test_criteria_registry_pipeline_flags_and_combine():
    assert registry.get("ema") is CRITERIA["ema"]
    assert registry.get_optional("does-not-exist") is None
    assert is_pipeline("garp") is True
    assert is_pipeline("ema") is False

    def first() -> list[int]:
        return [1, 2]

    def second() -> list[int]:
        return [3]

    assert combine(first, second)() == [1, 2, 3]


def test_criteria_interface_distinguishes_filters_and_pipelines():
    defs = definitions()
    assert defs["ema"].is_filter is True
    assert defs["garp"].is_pipeline is True
    assert "ema" in filter_criteria()
    assert "garp" not in filter_criteria()
    assert "garp" in pipeline_criteria()
    assert "ema" not in pipeline_criteria()

    filters = resolve_criteria(("ema", "value"))
    assert isinstance(filters, FilterCriteriaSelection)
    assert filters.names == ("ema", "value")
    assert filters.label == "ema+value"
    assert filters.filters

    pipeline = resolve_criteria(("garp",))
    assert isinstance(pipeline, PipelineCriteriaSelection)
    assert pipeline.name == "garp"


def test_criteria_interface_rejects_pipeline_filter_combinations():
    with pytest.raises(CriteriaSelectionError, match="cannot be combined"):
        resolve_criteria(("garp", "ema"))


def test_all_filter_only_criteria_build_filter_lists():
    for name, fn in CRITERIA.items():
        if is_pipeline(name):
            continue

        filters = fn()

        assert isinstance(filters, list), name
        assert filters, name


def test_garp_pipeline_prints_no_results(monkeypatch, capsys):
    monkeypatch.setattr(garp, "parse_ttl", lambda raw, default: 123)
    monkeypatch.setattr(garp, "run_garp_screen", lambda *args, **kwargs: None)

    garp.garp_pipeline(
        market="us",
        limit=5,
        output_csv=False,
        refresh=True,
        cache_ttl="1d",
    )

    assert "No tickers returned" in capsys.readouterr().out


def test_garp_pipeline_outputs_csv_or_rich_results(monkeypatch):
    calls: list[tuple[str, object]] = []
    rows = [SimpleNamespace(symbol="AAA")]

    def fake_run(*args, on_universe, **kwargs):
        on_universe(["AAA", "BBB"])
        calls.append(("run", (args, kwargs)))
        return rows

    monkeypatch.setattr(garp, "parse_ttl", lambda raw, default: 321)
    monkeypatch.setattr(garp, "run_garp_screen", fake_run)
    monkeypatch.setattr(
        garp, "print_csv", lambda results: calls.append(("csv", results))
    )
    monkeypatch.setattr(
        garp,
        "print_garp_results",
        lambda results, market: calls.append(("rich", (results, market))),
    )

    garp.garp_pipeline(
        market="india",
        limit=3,
        output_csv=True,
        refresh=False,
        cache_ttl="5m",
    )
    garp.garp_pipeline(
        market="india",
        limit=3,
        output_csv=False,
        refresh=False,
        cache_ttl="5m",
    )

    assert ("csv", rows) in calls
    assert ("rich", (rows, "india")) in calls


def test_pipeline_criteria_delegate_to_command_runners(monkeypatch):
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        "screener.commands.live_strategies.run_obv_trend_live",
        lambda **kwargs: calls.append(("obv", kwargs)),
    )
    monkeypatch.setattr(
        "screener.commands.live_strategies.run_vol_breakout_live",
        lambda **kwargs: calls.append(("vol", kwargs)),
    )
    monkeypatch.setattr(
        "screener.commands.insiders.run_promoter_buys",
        lambda **kwargs: calls.append(("promoter", kwargs)),
    )
    monkeypatch.setattr(
        "screener.unusual_volume.cli.run_unusual_volume",
        lambda **kwargs: calls.append(("unusual", kwargs)),
    )

    obv_trend.obv_trend_pipeline(market="us", limit=4)
    vol_breakout.vol_breakout_pipeline(market="india", limit=5)
    promoter_buys.promoter_buys_pipeline(
        market="india",
        limit=6,
        output_csv=True,
        refresh=True,
        cache_ttl="1h",
    )
    unusual_volume.unusual_volume_pipeline(market="us", limit=7, refresh=False)

    assert calls[0] == ("obv", {"market": "us", "as_of": date.today(), "limit": 4})
    assert calls[1] == (
        "vol",
        {"market": "india", "as_of": date.today(), "limit": 5},
    )
    assert calls[2][0] == "promoter"
    assert calls[2][1]["market"] == "india"
    assert calls[2][1]["limit"] == 6
    assert calls[2][1]["output_csv"] is True
    assert calls[3] == (
        "unusual",
        {"market": "us", "as_of": date.today(), "limit": 7, "refresh": False},
    )


def test_rs_breakout_pipeline_renders_and_writes(monkeypatch):
    calls: list[tuple[str, object]] = []
    result = SimpleNamespace(rows=[])

    monkeypatch.setattr(rs_breakout, "parse_ttl", lambda raw, default: 456)
    monkeypatch.setattr(
        "screener.commands.rs_breakout.run_rs_breakout_screen",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "screener.commands.rs_breakout.write_default_outputs",
        lambda *args, **kwargs: ("out.json", "out.md"),
    )
    monkeypatch.setattr(
        rs_breakout,
        "render_result",
        lambda *args, **kwargs: calls.append(("render", (args, kwargs))),
    )

    rs_breakout.rs_breakout_pipeline(
        market="india",
        limit=8,
        refresh=True,
        cache_ttl="15m",
    )

    assert calls == [
        (
            "render",
            (
                (result, ANY),
                {"limit": 8, "market": "india"},
            ),
        )
    ]
