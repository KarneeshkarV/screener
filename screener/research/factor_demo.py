"""Deterministic, fully-offline demonstration harness for the factor strategies.

Both ``scripts/run_strategy_report.py`` (which writes ``results/strategy_report.md``)
and ``tests/test_strategy_report.py`` (which asserts hardcoded metrics) build their
numbers from this single source of truth, so the report is reproducible by one
command and the test pins exactly what the report shows.

The price panel is a closed-form synthetic dataset — geometric drift plus a fixed
sinusoidal wiggle — so every metric is bit-stable across runs and machines. It is
engineered so the three factor strategies select genuinely different baskets:

* momentum favours the strongest trends (MOMA / MOMB / WILD),
* low-volatility favours the calmest names (CALM / STDY),
* the combo favours strong-but-calm names (MOMB / STDY).

This is a research/demonstration dataset, NOT real market data — it exists to
prove the *machinery* (causal factor signals, cross-sectional ranking, metrics)
deterministically. It is not evidence about live factor returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from screener.backtester.models import (
    BacktestConfig,
    BacktestResult,
    ExecutionPolicy,
    PortfolioPolicy,
    SignalPolicy,
    UniversePolicy,
)
from screener.backtester.rolling import run_rolling_backtest

# ── panel geometry ───────────────────────────────────────────────────
_PANEL_START = "2022-01-03"
_PANEL_BARS = 470  # ~22 months of business days; >252 warmup before the window
_INDEX = pd.bdate_range(_PANEL_START, periods=_PANEL_BARS)

BACKTEST_START = date(2023, 1, 2)
BACKTEST_END = _INDEX[-1].date()
BENCHMARK = "SPY"


@dataclass(frozen=True)
class _Profile:
    """One synthetic symbol: geometric daily ``growth`` + sinusoidal ``noise``."""

    ticker: str
    growth: float
    noise: float
    volume: float


# Distinct (growth, noise) profiles so the factors disagree on the ranking.
_PROFILES: tuple[_Profile, ...] = (
    _Profile("MOMA", growth=0.0018, noise=0.020, volume=820_000.0),
    _Profile("MOMB", growth=0.0016, noise=0.010, volume=760_000.0),
    _Profile("STDY", growth=0.0008, noise=0.002, volume=540_000.0),
    _Profile("CALM", growth=0.0004, noise=0.0015, volume=480_000.0),
    _Profile("CHOP", growth=0.0010, noise=0.030, volume=300_000.0),
    _Profile("WILD", growth=0.0015, noise=0.045, volume=220_000.0),
    _Profile("DOWNA", growth=-0.0010, noise=0.015, volume=900_000.0),
    _Profile("DOWNB", growth=-0.0004, noise=0.008, volume=650_000.0),
)
_BENCHMARK_PROFILE = _Profile(BENCHMARK, growth=0.0005, noise=0.004, volume=1_000_000.0)

DEMO_TICKERS: tuple[str, ...] = tuple(p.ticker for p in _PROFILES)


def _frame(profile: _Profile) -> pd.DataFrame:
    n = _PANEL_BARS
    base = 50.0
    drift = base * (1.0 + profile.growth) ** np.arange(n)
    wiggle = profile.noise * base * np.sin(np.arange(n))
    close = pd.Series(drift + wiggle, index=_INDEX)
    openp = close.shift(1).fillna(close.iloc[0])
    pad = abs(profile.noise) * base + 0.1
    high = pd.concat([openp, close], axis=1).max(axis=1) + pad
    low = pd.concat([openp, close], axis=1).min(axis=1) - pad
    return pd.DataFrame(
        {
            "open": openp,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(profile.volume, index=_INDEX, dtype=float),
        }
    )


def build_demo_panel() -> dict[str, pd.DataFrame]:
    """Return the fixed OHLCV panel keyed by ticker (includes the benchmark)."""
    panel = {p.ticker: _frame(p) for p in _PROFILES}
    panel[_BENCHMARK_PROFILE.ticker] = _frame(_BENCHMARK_PROFILE)
    return panel


class _PanelFetcher:
    """Minimal in-memory ``PriceFetcher`` over the demo panel (offline)."""

    def __init__(self, panel: dict[str, pd.DataFrame]) -> None:
        self._panel = panel

    def fetch(
        self, tickers: Iterable[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        out: dict[str, pd.DataFrame] = {}
        for t in tickers:
            frame = self._panel.get(t, pd.DataFrame())
            out[t] = (
                frame
                if frame.empty
                else frame.loc[(frame.index >= s) & (frame.index <= e)]
            )
        return out


@dataclass(frozen=True)
class StrategySpec:
    """Report metadata for one strategy run."""

    name: str
    title: str
    paper: str
    signal: str


REPORT_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="momentum_12_1",
        title="12-1 Momentum",
        paper="Jegadeesh & Titman (1993), J. Finance 48(1)",
        signal="rank by close[t-21]/close[t-252]-1; long positive-momentum winners",
    ),
    StrategySpec(
        name="low_volatility",
        title="Low Volatility",
        paper="Ang, Hodrick, Xing & Zhang (2006), J. Finance 61(1)",
        signal="rank by -stdev(daily returns, 252); long the calmest names",
    ),
    StrategySpec(
        name="mom_lowvol_combo",
        title="Momentum + Low-Vol Combo",
        paper="Blitz & van Vliet (2007) defensive-factor blend; JT (1993) + AHXZ (2006)",
        signal="rank by 0.5*pct_rank(mom) + 0.5*pct_rank(-vol); long calm winners",
    ),
)

DEMO_TOP = 3
DEMO_HOLD = 21


def run_demo_backtest(
    strategy_name: str, *, top: int = DEMO_TOP, hold: int = DEMO_HOLD
) -> BacktestResult:
    """Run one strategy on the fixed demo panel and return its result."""
    panel = build_demo_panel()
    cfg = BacktestConfig(
        market="us",
        as_of=BACKTEST_END,
        benchmark=BENCHMARK,
        universe=UniversePolicy(tickers=DEMO_TICKERS),
        signals=SignalPolicy(
            strategy_name=strategy_name,
            entry_expr="",  # filled from the registry below
            exit_expr=None,
        ),
        execution=ExecutionPolicy(
            hold=hold,
            stop_loss=None,
            take_profit=None,
            trailing_stop=None,
            slippage_bps=0.0,
            commission_bps=0.0,
        ),
        portfolio=PortfolioPolicy(top=top, initial_capital=100_000.0),
    )
    from screener.strategies.spec import discover_plugins
    from screener.strategies.spec import registry as strategy_registry

    discover_plugins()
    from screener.strategies.spec import ExpressionStrategySpec

    spec = strategy_registry.get(strategy_name)
    if not isinstance(spec, ExpressionStrategySpec):
        raise ValueError(f"Demo strategy {strategy_name!r} is not expression-based")
    cfg = cfg.model_copy(update={"entry_expr": spec.entry, "exit_expr": spec.exit})
    return run_rolling_backtest(
        cfg,
        _PanelFetcher(panel),
        start_date=BACKTEST_START,
        end_date=BACKTEST_END,
    )
