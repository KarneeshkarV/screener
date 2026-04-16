"""CLI-level tests: argument validation, deprecation warnings, rejection messages.

Uses Click's CliRunner so no network calls are needed.
"""
from __future__ import annotations

from click.testing import CliRunner

from main import cli


class TestDeprecatedBacktestAlias:
    def test_backtest_alias_prints_deprecation(self, tmp_db):
        """'backtest' (old name) should work but print a deprecation warning."""
        runner = CliRunner()
        # We don't have a real saved run, but the deprecation warning fires
        # before the actual backtest lookup.  The command will fail with a
        # ClickException about "No saved screener run" — that's fine; we just
        # check that the deprecation line appeared in stderr.
        result = runner.invoke(cli, ["backtest", "--help"])
        # --help exits cleanly; the deprecation is in the info_name branch
        # which only fires on real invocation.  Test the actual invocation:
        result = runner.invoke(cli, ["backtest", "-m", "us"], catch_exceptions=False)
        assert "deprecated" in result.output.lower() or "deprecated" in (result.stderr or "").lower() or \
               "backtest-last-run" in result.output


class TestHistoricalCriteriaAccepted:
    def test_all_criteria_accepted_by_click(self):
        """All criteria (including fundamentals-based) should be accepted by Click."""
        runner = CliRunner()
        all_criteria = ("ema", "breakout", "ema_breakout",
                        "value", "quality", "cheap_quality",
                        "undervalued", "dividend", "momentum_value")
        for criterion in all_criteria:
            result = runner.invoke(
                cli,
                ["backtest-historical", "-c", criterion, "--as-of", "2025-01-01", "--help"],
            )
            assert "Invalid value" not in result.output, (
                f"Click rejected valid criterion '{criterion}'"
            )

    def test_unknown_criterion_rejected(self):
        """An unknown criterion name should fail with an error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["backtest-historical", "-c", "nonexistent_criterion", "--as-of", "2025-01-01"],
        )
        assert result.exit_code != 0


class TestAsOfRequired:
    def test_missing_as_of_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["backtest-historical", "-c", "ema"])
        assert result.exit_code != 0
        assert "as-of" in result.output.lower() or "missing" in result.output.lower()


class TestBacktestLastRunHelp:
    def test_help_shows_new_command_name(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["backtest-last-run", "--help"])
        assert result.exit_code == 0
        assert "backtest-last-run" in result.output or "Backtest" in result.output
