"""Registration of the India momentum ablation stages as strategies."""

from screener.strategies.plugins.india_momentum import STAGES, strategy_name
from screener.strategies.spec import (
    ExpressionStrategySpec,
    discover_plugins,
    resolve_strategy_spec,
)


def test_every_stage_registers_a_plain_and_a_lower_circuit_strategy():
    discover_plugins()
    for stage, spec in STAGES.items():
        plain = resolve_strategy_spec(strategy_name(stage))
        locked = resolve_strategy_spec(strategy_name(stage, lower_circuit_exit=True))
        assert isinstance(plain, ExpressionStrategySpec)
        assert isinstance(locked, ExpressionStrategySpec)
        assert plain.entry == spec.entry
        assert plain.exit is None
        assert locked.entry == f"{spec.entry} and lower_lock_recent == 0"
        assert locked.exit == "lower_lock > 0"
        assert plain.required_lookback is not None
        assert plain.required_lookback() > 252


def test_strategy_name_tags_the_lower_circuit_variant():
    assert strategy_name("s4_volpen") == "india_mom_s4_volpen"
    assert (
        strategy_name("s4_volpen", lower_circuit_exit=True) == "india_mom_s4_volpen_lc"
    )
