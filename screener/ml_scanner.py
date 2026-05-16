"""RS Breakout scanner enhanced with all 3 ML features:
1. ML Signal Confidence (v3 model)
2. Regime Detection (bull/bear/chop)
3. Kelly Position Sizing
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from screener.rs_breakout import (
    RsBreakoutResult,
    RsBreakoutRow,
    fetch_price_data,
    scan_rs_breakouts,
    normalize_bars,
)
from screener.ml_signal_v3 import SimpleSignalModel, SimpleFeatureExtractor
from screener.ml_signal_regime import RegimeAwareModel
from screener.ml_kelly import confidence_to_size
from screener.regime import RegimeDetector, TrendRegime, stress_position_multiplier


@dataclass(frozen=True)
class EnhancedBreakoutRow:
    """RS Breakout row with ML confidence, regime, and Kelly sizing."""

    base: RsBreakoutRow
    regime: str
    stress: float
    ml_confidence: float
    kelly_size: float
    is_tradeable: bool


def enhance_scan(
    result: RsBreakoutResult,
    bars_by_symbol: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    baseline_model: Optional[SimpleSignalModel] = None,
    regime_model: Optional[RegimeAwareModel] = None,
    use_regime: bool = True,
) -> list[EnhancedBreakoutRow]:
    """Add ML confidence, regime, and Kelly sizing to breakout rows."""
    if not result.full:
        return []

    # Current regime
    bench_norm = normalize_bars(benchmark_bars, result.as_of)
    regime = RegimeDetector.classify(bench_norm, bench_norm)
    stress = regime.stress
    trend = regime.trend_regime
    is_tradeable = regime.is_tradeable

    extractor = SimpleFeatureExtractor()
    enhanced = []

    for row in result.full:
        bars = bars_by_symbol.get(row.symbol)
        if bars is None or bars.empty:
            continue

        df = normalize_bars(bars, result.as_of)
        if df.empty:
            continue

        features = extractor.extract(df, benchmark_bars=bench_norm)
        if features.empty:
            ml_conf = 0.5
        else:
            try:
                if use_regime and regime_model is not None:
                    ml_conf = float(regime_model.predict(features.iloc[[-1]], regime=trend)[0])
                elif baseline_model is not None:
                    ml_conf = float(baseline_model.predict(features.iloc[[-1]])[0])
                else:
                    ml_conf = 0.5
            except Exception:
                ml_conf = 0.5

        kelly = confidence_to_size(ml_conf, regime_stress=stress)

        enhanced.append(EnhancedBreakoutRow(
            base=row,
            regime=trend,
            stress=round(stress, 4),
            ml_confidence=round(ml_conf, 4),
            kelly_size=round(kelly, 4),
            is_tradeable=is_tradeable,
        ))

    # Sort by ML confidence descending
    enhanced.sort(key=lambda r: r.ml_confidence, reverse=True)
    return enhanced


def run_enhanced_scan(
    tickers: list[str],
    as_of: date,
    market: str = "us",
    model_path: Optional[Path] = None,
    regime_model_path: Optional[Path] = None,
    use_regime: bool = True,
    confidence_threshold: Optional[float] = None,
) -> list[EnhancedBreakoutRow]:
    """Full pipeline: fetch data, run RS breakout, enhance with ML + regime + Kelly."""
    from screener.backtester.data import YFinancePriceFetcher

    fetcher = YFinancePriceFetcher()
    bars_by_symbol, benchmark_bars = fetch_price_data(
        tickers, market, as_of, fetcher, history_days=220
    )

    # Load models
    baseline_model = None
    regime_model = None
    if model_path is not None and model_path.exists():
        try:
            baseline_model = SimpleSignalModel.load(model_path)
        except Exception as e:
            print(f"Warning: could not load baseline model: {e}")
    if use_regime and regime_model_path is not None and regime_model_path.exists():
        try:
            regime_model = RegimeAwareModel.load(regime_model_path)
        except Exception as e:
            print(f"Warning: could not load regime model: {e}")

    # Run base scan
    result = scan_rs_breakouts(
        bars_by_symbol,
        benchmark_bars,
        as_of,
        require_delivery=True,
    )

    # Enhance
    enhanced = enhance_scan(
        result, bars_by_symbol, benchmark_bars,
        baseline_model=baseline_model,
        regime_model=regime_model,
        use_regime=use_regime,
    )

    # Apply confidence threshold if specified
    if confidence_threshold is not None:
        enhanced = [r for r in enhanced if r.ml_confidence >= confidence_threshold]

    return enhanced


def print_enhanced_results(rows: list[EnhancedBreakoutRow]) -> None:
    """Pretty-print enhanced breakout results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if not rows:
        console.print("[yellow]No breakout signals found.[/yellow]")
        return

    table = Table(title="Enhanced RS Breakout Scan")
    table.add_column("Symbol", style="bold")
    table.add_column("Close", justify="right")
    table.add_column("RS 55d", justify="right")
    table.add_column("Vol Ratio", justify="right")
    table.add_column("Regime", justify="center")
    table.add_column("Stress", justify="right")
    table.add_column("ML Conf", justify="right")
    table.add_column("Kelly", justify="right")
    table.add_column("Tradeable", justify="center")

    for row in rows[:30]:
        base = row.base
        conf_color = "green" if row.ml_confidence >= 0.6 else "yellow" if row.ml_confidence >= 0.5 else "red"
        kelly_color = "green" if row.kelly_size >= 1.0 else "yellow"
        regime_color = "green" if row.regime == "UPTREND" else "red" if row.regime == "DOWNTREND" else "yellow"
        tradeable = "✓" if row.is_tradeable else "✗"

        table.add_row(
            base.symbol,
            f"{base.close:.2f}",
            f"{base.rs_55:.2f}",
            f"{base.volume_ratio:.2f}",
            f"[{regime_color}]{row.regime}[/{regime_color}]",
            f"{row.stress:.2f}",
            f"[{conf_color}]{row.ml_confidence:.2f}[/{conf_color}]",
            f"[{kelly_color}]{row.kelly_size:.2f}x[/{kelly_color}]",
            tradeable,
        )

    console.print(table)
    console.print(f"\nTotal signals: {len(rows)}")
    console.print(f"Tradeable: {sum(1 for r in rows if r.is_tradeable)}")
    if rows:
        avg_conf = np.mean([r.ml_confidence for r in rows])
        avg_kelly = np.mean([r.kelly_size for r in rows])
        console.print(f"Avg confidence: {avg_conf:.2%}")
        console.print(f"Avg Kelly size: {avg_kelly:.2f}x")
