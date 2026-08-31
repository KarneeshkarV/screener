"""The screen's gate flags reach the candidate layer, or are refused.

The flags themselves are pinned for parity with the backtest in
``tests/correctness/test_screen_backtest_reconciliation.py``. What is pinned
here is the wiring on the screen's own side: a flag typed on the command line
has to arrive at the gates, and a flag that this run cannot honour has to be
refused rather than ignored.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from screener import api
from screener.cli import cli
from screener.gate_options import gate_overrides
from screener.screen_candidates import (
    IntervalNotScreenableError,
    ScreenStrategy,
    resolve_screen_gates,
)
from screener.screen_workflow import ScreenRequest, run_screen_workflow
from screener.strategies.spec import discover_plugins, resolve_strategy_spec


def _strategy(name: str = "breakout") -> ScreenStrategy:
    discover_plugins()
    return ScreenStrategy(criterion=name, spec=resolve_strategy_spec(name))


class TestTypedFlagsReachTheGates:
    def test_a_typed_flag_wins_over_the_declared_gate(self) -> None:
        gates = resolve_screen_gates(
            _strategy(),
            market="india",
            overrides=gate_overrides(min_price=250.0, min_score=80.0),
        )

        assert gates.min_price == 250.0
        assert gates.min_score == 80.0

    def test_an_untyped_flag_leaves_the_market_floor_standing(self) -> None:
        # Nothing typed means the venue minimum, which is what the backtest
        # would have applied. A screen with no floor names penny stocks no
        # backtest would have entered.
        gates = resolve_screen_gates(_strategy(), market="india")

        assert gates.min_price == 10.0
        assert gates.min_avg_dollar_volume == 100_000.0

    def test_an_explicit_zero_disables_the_gate(self) -> None:
        gates = resolve_screen_gates(
            _strategy(), market="india", overrides=gate_overrides(min_price=0.0)
        )

        assert gates.min_price is None

    def test_an_explicit_default_adv_window_remains_an_override(self) -> None:
        overrides = gate_overrides(adv_window=20, adv_window_was_explicit=True)

        assert overrides["avg_dollar_volume_window"] == 20


class TestRefusals:
    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--min-price", "5"),
            ("--min-score", "50"),
            ("--adv-window", "20"),
            ("--max-universe", "10"),
        ],
    )
    def test_a_gate_flag_is_refused_on_a_filters_only_criterion(
        self, flag: str, value: str
    ) -> None:
        # ``dividend`` names TradingView filters and no strategy, so
        # there is no bar rule for a gate to gate. Accepting the flag quietly
        # would return a result that ignored it with nothing to say so.
        result = CliRunner().invoke(cli, ["screen", "-c", "dividend", flag, value])

        assert result.exit_code != 0
        assert "names TradingView filters only" in result.output

    def test_the_interval_flag_is_refused_on_a_filters_only_criterion(self) -> None:
        result = CliRunner().invoke(
            cli, ["screen", "-c", "dividend", "--interval", "1h"]
        )

        assert result.exit_code != 0
        assert "--interval" in result.output

    def test_an_intraday_interval_is_refused_with_an_earnings_blackout(self) -> None:
        # A blackout suppresses whole calendar days, which has no intraday
        # meaning. Running anyway would apply it to the wrong bars.
        request = ScreenRequest(
            market="india",
            criteria_names=("breakout",),
            limit=10,
            order_by="setup_score",
            output_csv=True,
            detail=False,
            refresh=False,
            cache_ttl="off",
            report_path=None,
            universe="nifty50",
            interval="1h",
            gate_overrides=gate_overrides(earnings_blackout_days=3),
        )

        with pytest.raises(IntervalNotScreenableError, match="earnings-blackout"):
            run_screen_workflow(request)

    def test_a_negative_universe_cap_is_refused_by_the_api(self) -> None:
        with pytest.raises(ValueError, match="max_universe"):
            api.screen(max_universe=-1)


def test_the_api_takes_a_profile_as_the_gates_outright() -> None:
    # A caller holding the profile a backtest ran with can hand it straight
    # over, which is the point: screen exactly what the backtest entered.
    from screener.strategies.spec import StrategyProfile

    request_gates = StrategyProfile(min_price=123.0, sector_neutral=True)
    resolved = resolve_screen_gates(
        _strategy(), market="us", overrides=request_gates.model_dump()
    )

    assert resolved.min_price == 123.0
    assert resolved.sector_neutral is True
    # Unset gates still pick up the venue floor, as they do for a declared one.
    assert resolved.min_avg_dollar_volume == 1_000.0
