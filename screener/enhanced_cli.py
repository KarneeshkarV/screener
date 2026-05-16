"""CLI for the enhanced RS Breakout scanner with ML + Regime + Kelly."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import click

from screener.ml_scanner import run_enhanced_scan, print_enhanced_results
from screener.universes import UNIVERSES


@click.group()
def cli() -> None:
    """Enhanced RS Breakout scanner with ML confidence, regime detection, and Kelly sizing."""
    pass


@cli.command()
@click.option("--market", "-m", type=click.Choice(["us", "india"]), default="us")
@click.option("--as-of", type=click.DateTime(["%Y-%m-%d"]), default=str(date.today()))
@click.option("--universe", "-u", type=click.Choice(list(UNIVERSES.keys())), default="sp500")
@click.option("--model", type=click.Path(exists=True, path_type=Path), help="Path to baseline ML model (.pkl)")
@click.option("--regime-model", type=click.Path(exists=True, path_type=Path), help="Path to regime-aware ML model (.pkl)")
@click.option("--no-regime", is_flag=True, help="Disable regime-aware model, use baseline only")
@click.option("--confidence-threshold", type=float, default=None, help="Min ML confidence (0-1)")
@click.option("--top", "-n", type=int, default=20, help="Show top N results")
def scan(
    market: str,
    as_of: date,
    universe: str,
    model: Path | None,
    regime_model: Path | None,
    no_regime: bool,
    confidence_threshold: float | None,
    top: int,
) -> None:
    """Run enhanced RS breakout scan with ML confidence, regime, and Kelly sizing."""
    tickers = UNIVERSES.get(universe, [])
    if not tickers:
        click.echo(f"Unknown universe: {universe}")
        sys.exit(1)

    use_regime = not no_regime

    # Default model paths
    if model is None:
        default_model = Path(__file__).parent / "training_data" / "model_baseline.pkl"
        if default_model.exists():
            model = default_model
    if regime_model is None and use_regime:
        default_regime = Path(__file__).parent / "training_data" / "model_regime.pkl"
        if default_regime.exists():
            regime_model = default_regime

    click.echo(f"Scanning {len(tickers)} stocks for {market} market, as-of {as_of.date()}...")
    click.echo(f"  Baseline model: {model}")
    if use_regime:
        click.echo(f"  Regime model: {regime_model}")
    if confidence_threshold:
        click.echo(f"  Confidence threshold: {confidence_threshold}")

    rows = run_enhanced_scan(
        tickers=tickers,
        as_of=as_of.date(),
        market=market,
        model_path=model,
        regime_model_path=regime_model,
        use_regime=use_regime,
        confidence_threshold=confidence_threshold,
    )

    print_enhanced_results(rows[:top])


if __name__ == "__main__":
    cli()
