"""Stage 1 guards for StrategyProfile (docs/plans/unify-screen-backtest.md).

The point of the stage is anti-drift: ``StrategyProfile`` must mirror
``SignalPanelInputs`` field for field for everything that gates candidate
eligibility, so these tests fail loudly when the two drift apart.
"""

from __future__ import annotations

from datetime import date
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from screener.backtester.models import BacktestConfig
from screener.backtester.signal_panel import (
    NON_PANEL_PROFILE_FIELDS,
    RUN_SCOPED_SIGNAL_PANEL_FIELDS,
    SIGNAL_PANEL_INPUT_FIELDS,
    SignalPanelInputs,
)
from screener.strategies.spec import (
    DEFAULT_STRATEGY_PROFILE,
    ExpressionStrategySpec,
    StrategyProfile,
    discover_plugins,
    registry,
    resolve_strategy_profile,
)


def _minimal_config() -> BacktestConfig:
    return BacktestConfig(
        market="us",
        as_of=date(2024, 6, 3),
        benchmark="SPY",
        entry_expr="close > 0",
        exit_expr=None,
        hold=5,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        top=2,
        initial_capital=100_000.0,
    )


def test_profile_fields_partition_signal_panel_inputs():
    # Every SignalPanelInputs field is either mirrored on StrategyProfile or
    # explicitly classified as run-scoped. A new gate added to the panel
    # inputs lands in neither set until someone decides, which fails here.
    # NON_PANEL_PROFILE_FIELDS is the third bucket: a profile field that is not
    # a gate at all. ``tv_prefilter`` is one - it narrows the field before bars
    # are fetched, and the panel judges whatever names it is handed.
    gate_fields = frozenset(StrategyProfile.model_fields) - NON_PANEL_PROFILE_FIELDS
    assert not (gate_fields & RUN_SCOPED_SIGNAL_PANEL_FIELDS)
    assert gate_fields | RUN_SCOPED_SIGNAL_PANEL_FIELDS == SIGNAL_PANEL_INPUT_FIELDS
    assert NON_PANEL_PROFILE_FIELDS <= frozenset(StrategyProfile.model_fields)


def test_profile_annotations_match_signal_panel_inputs_field_for_field():
    # Same names must mean same types: the profile feeds SignalPanelInputs in
    # later stages, so a silent annotation change would corrupt eligibility.
    panel_hints = get_type_hints(SignalPanelInputs)
    profile_hints = get_type_hints(StrategyProfile)
    inherited_rules = {"entry_expr", "exit_expr"}
    # The rule fields stay required on ExpressionStrategySpec (D14); on the
    # profile None means "the spec's own entry/exit stand", so both are the
    # optional form of the panel's exit_expr annotation.
    assert panel_hints["exit_expr"] == str | None
    assert profile_hints["entry_expr"] == str | None
    assert profile_hints["exit_expr"] == str | None
    skip = inherited_rules | set(NON_PANEL_PROFILE_FIELDS)
    for name in set(StrategyProfile.model_fields) - skip:
        assert profile_hints[name] == panel_hints[name], name


def test_profile_defaults_equal_backtest_config_effective_gates():
    # An attached profile encodes "today's effective defaults", so its values
    # must equal what SignalPanelInputs.from_config sees on a default config.
    inputs = SignalPanelInputs.from_config(_minimal_config())
    profile = StrategyProfile()
    assert profile.min_price == inputs.min_price
    assert profile.min_avg_dollar_volume == inputs.min_avg_dollar_volume
    assert profile.avg_dollar_volume_window == inputs.avg_dollar_volume_window
    assert profile.regime_filter == inputs.regime_filter
    assert profile.earnings_blackout_days == inputs.earnings_blackout_days
    assert profile.sector_neutral == inputs.sector_neutral


def test_default_profile_is_the_shared_baseline():
    assert DEFAULT_STRATEGY_PROFILE == StrategyProfile()
    # Total resolution: even a missing spec resolves to something valid.
    assert resolve_strategy_profile(None) == DEFAULT_STRATEGY_PROFILE


def test_spec_without_profile_resolves_to_default_unchanged():
    spec = ExpressionStrategySpec(name="no_profile", entry=" close > 0 ")

    assert spec.profile is None
    assert resolve_strategy_profile(spec) == DEFAULT_STRATEGY_PROFILE


def test_attached_profile_wins_over_default_and_overrides_win_last():
    attached = StrategyProfile(min_price=10.0, sector_neutral=True)
    spec = ExpressionStrategySpec(
        name="with_profile", entry="close > 0", profile=attached
    )

    assert resolve_strategy_profile(spec) == attached
    assert resolve_strategy_profile(spec, {"min_price": 99.0}) == StrategyProfile(
        min_price=99.0, sector_neutral=True
    )
    # Overrides also apply over the defaults when no profile is attached.
    assert resolve_strategy_profile(None, {"regime_filter": ("bull",)}) == (
        StrategyProfile(regime_filter=("bull",))
    )


def test_override_rejects_unknown_keys_by_name():
    with pytest.raises(ValueError, match="min_price"):
        resolve_strategy_profile(None, {"minn_price": 5.0})


def test_override_values_are_validated_by_the_model():
    with pytest.raises(ValidationError):
        resolve_strategy_profile(None, {"avg_dollar_volume_window": "not-an-int"})


def test_every_registered_expression_plugin_keeps_current_effective_defaults():
    # Plugin parity on the gates: a profile whose *gate* values deviate from
    # the effective defaults would move candidates. ``tv_prefilter`` is exempt
    # because it is not a gate; stage 6 attaches one to every strategy that has
    # a TradingView spelling, and doing so must not move a candidate.
    discover_plugins()
    for name, spec in registry.items():
        if not isinstance(spec, ExpressionStrategySpec):
            continue
        resolved = resolve_strategy_profile(spec)
        gates = resolved.model_copy(
            update=dict.fromkeys(NON_PANEL_PROFILE_FIELDS, None)
        )
        assert gates == DEFAULT_STRATEGY_PROFILE, name


def test_only_strategies_with_a_tradingview_spelling_declare_a_prefilter():
    # A prefilter is an optimisation, so it must name a real criterion. A typo
    # would otherwise fail at screen time, on a live run, rather than here.
    from screener.criteria import registry as criteria_registry

    discover_plugins()
    declared = {
        name: resolve_strategy_profile(spec).tv_prefilter
        for name, spec in registry.items()
        if isinstance(spec, ExpressionStrategySpec)
        and resolve_strategy_profile(spec).tv_prefilter is not None
    }

    assert set(declared) == {"breakout", "mark_minervini", "momentum_12_1"}
    for strategy_name, criterion in declared.items():
        assert criteria_registry.get_optional(criterion) is not None, strategy_name
