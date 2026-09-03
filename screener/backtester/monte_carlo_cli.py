"""``backtest-monte-carlo``: a rolling backtest plus a bootstrap of its risk.

A single backtest is one draw from the distribution of outcomes the strategy
could have produced. This command runs the same rolling simulation as
``backtest-rolling`` (identical flags, identical engine), then resamples the
resulting equity curve in blocks to answer what a *typical* and a *bad* run of
the same strategy look like: the 5th-percentile return, the drawdown
percentiles, and the odds of ending in profit or at half the starting capital.

The equity curve is the thing resampled, not the trade list, because a rolling
run holds ``--top`` positions at once. See ``EquityMonteCarloResult`` for why
chaining overlapping trades would misstate the drawdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from screener import agentio
from screener.backtester.cli_common import resolve_report_path, write_tearsheet
from screener.backtester.display import print_backtest, print_ledger_csv
from screener.backtester.rolling import rolling_run_options
from screener.backtester.rolling_simulation import run_rolling_backtest
from screener.backtester.workflow import BacktestRequest, resolve_backtest_run

__all__ = ["backtest_monte_carlo"]

# The engine states its bounds in terms of its own argument names and tags
# each rejection with the field it names, so the bounds themselves live in one
# place instead of being restated here. This is the whole translation: look up
# the flag that carries the field, and let the error restate itself.
_FLAG_FOR_FIELD = {
    "iterations": "--iterations",
    "block": "--block",
    "seed": "--seed",
    "keep_paths": "--paths",
    "ruin_threshold": "--ruin-threshold",
}


def _flag_message(error: ValueError) -> str:
    """Restate an engine rejection in the flag the user actually typed.

    Anything the engine raises that is not about one of these arguments (an
    unresampleable equity curve, say) has no flag to name and reaches the user
    unchanged.
    """
    from screener.backtester.optimization.monte_carlo import MonteCarloArgumentError

    if isinstance(error, MonteCarloArgumentError):
        flag = _FLAG_FOR_FIELD.get(error.field)
        if flag is not None:
            return error.named(flag)
    return str(error)


@click.command(name="backtest-monte-carlo")
@rolling_run_options
@click.option(
    "--iterations",
    "mc_iterations",
    type=int,
    default=5000,
    show_default=True,
    help="Number of synthetic equity paths to draw.",
)
@click.option(
    "--block",
    "mc_block",
    type=int,
    default=20,
    show_default=True,
    help=(
        "Length in bars of each resampled block. Blocks keep the short-horizon "
        "autocorrelation that drives drawdown; 1 makes the draw i.i.d. and "
        "understates it."
    ),
)
@click.option(
    "--seed",
    "mc_seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed, so a run is reproducible.",
)
@click.option(
    "--ruin-threshold",
    "mc_ruin_threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Fraction of starting capital that counts as ruin when touched.",
)
@click.option(
    "--paths",
    "mc_keep_paths",
    type=int,
    default=1000,
    show_default=True,
    help=(
        "Simulated paths retained for the fan chart's faint individual lines. "
        "The percentile bands are taken over every iteration rather than over "
        "these, so raising this does not tighten them and the chart is drawn "
        "either way; 0 keeps the bands and the realized run and drops only "
        "the lines."
    ),
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the raw Monte Carlo result to this JSON file.",
)
def backtest_monte_carlo(**params: Any) -> None:
    """Run a rolling backtest, then block-bootstrap its equity curve."""
    from screener.backtester.optimization.monte_carlo import (
        equity_monte_carlo_metrics,
        simulate_equity_monte_carlo_paths,
        validate_equity_monte_carlo_flags,
    )

    iterations = params.pop("mc_iterations")
    block = params.pop("mc_block")
    seed = params.pop("mc_seed")
    ruin_threshold = params.pop("mc_ruin_threshold")
    keep_paths = params.pop("mc_keep_paths")
    json_path = params.pop("json_path")

    # Checked before the backtest, not inside the simulation: the rolling run
    # takes minutes, and a bad flag must not surface as a traceback after it.
    try:
        validate_equity_monte_carlo_flags(
            iterations=iterations,
            block=block,
            seed=seed,
            keep_paths=keep_paths,
            ruin_threshold=ruin_threshold,
        )
    except ValueError as exc:
        raise click.UsageError(f"{_flag_message(exc)}.") from exc

    ctx = click.get_current_context()
    request = BacktestRequest(
        mode="rolling",
        context_obj=ctx.obj,
        adv_window_was_explicit=(
            ctx.get_parameter_source("adv_window")
            == click.core.ParameterSource.COMMANDLINE
        ),
        point_in_time_was_explicit=(
            ctx.get_parameter_source("point_in_time")
            is not click.core.ParameterSource.DEFAULT
        ),
        **params,
    )
    run = resolve_backtest_run(request)
    assert run.start_date is not None and run.end_date is not None
    result = run_rolling_backtest(
        run.config,
        run.price_fetcher,
        start_date=run.start_date,
        end_date=run.end_date,
        fundamental_fetcher=run.fundamental_fetcher,
    )
    # Whether the block fits, and whether the curve is resampleable at all,
    # can only be known once the run has produced it. Raised as a plain error
    # rather than a usage error: the flags were legal, and a usage dump after
    # a multi-minute backtest buries the one line that says what to change.
    try:
        mc, mc_paths = simulate_equity_monte_carlo_paths(
            result.equity_curve,
            iterations=iterations,
            block=block,
            seed=seed,
            ruin_threshold=ruin_threshold,
            keep_paths=keep_paths,
        )
    except ValueError as exc:
        raise click.ClickException(_flag_message(exc)) from exc
    # Merge into the run's own metrics so the terminal table and the tear-sheet
    # show the realized run and its bootstrap side by side, in one place.
    result.metrics.update(equity_monte_carlo_metrics(mc))

    generated_report = resolve_report_path(
        params["report_path"], params["output_csv"], "backtest-monte-carlo"
    )
    if generated_report:
        write_tearsheet(
            result,
            generated_report,
            title="Monte Carlo Backtest Tear Sheet",
            extra_notes=[run.universe_note] if run.universe_note else [],
            monte_carlo=(mc, mc_paths),
        )
    if json_path:
        from screener.backtester.optimization.reporting import write_json_report

        write_json_report(mc.model_dump(mode="json"), json_path)
    if params["output_csv"]:
        print_ledger_csv(result)
        return

    console = agentio.get_console()
    console.print(
        f"[dim]Rolling window: {run.start_date.isoformat()} "
        f"to {run.end_date.isoformat()}[/dim]"
    )
    if run.universe_note:
        console.print(f"[dim]Universe: {run.universe_note}[/dim]")
    console.print(
        f"[dim]Monte Carlo: {mc.iterations:,} iterations, {mc.bars:,} bars, "
        f"block {mc.block}, seed {mc.seed}[/dim]"
    )
    print_backtest(result, show_ledger=False)
    if generated_report:
        from screener.reporting import windows_report_path

        console.print(f"[green]Report:[/green] {generated_report}")
        windows_report = windows_report_path(generated_report)
        if windows_report:
            console.print(f"[green]Windows:[/green] {windows_report}")
        if params["open_report"]:
            from screener.reporting import open_report as open_report_file

            open_report_file(generated_report)
    if json_path:
        console.print(f"[green]JSON:[/green] {json_path}")
