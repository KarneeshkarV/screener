#!/usr/bin/env python
"""Regenerate ``results/strategy_report.md`` from the deterministic demo panel.

One command reproduces every number in the report:

    uv run python scripts/run_strategy_report.py

The numbers come from :mod:`screener.research.factor_demo`, a closed-form
synthetic price panel, so the output is bit-stable across runs and machines.
``tests/test_strategy_report.py`` asserts the same metrics with hardcoded
expected values, so the report and the test can never silently drift apart.

This is research, not financial advice. The panel is synthetic and exists to
prove the factor machinery deterministically, not to forecast live returns.
"""

from __future__ import annotations

from pathlib import Path

from screener.backtester.models import BacktestResult
from screener.research.factor_demo import (
    BACKTEST_END,
    BACKTEST_START,
    BENCHMARK,
    DEMO_HOLD,
    DEMO_TICKERS,
    DEMO_TOP,
    REPORT_STRATEGIES,
    StrategySpec,
    run_demo_backtest,
)

_REPORT_PATH = Path(__file__).resolve().parent.parent / "results" / "strategy_report.md"


def _cli_equivalent(spec: StrategySpec) -> str:
    return (
        "uv run screener backtest-rolling -m us "
        f"--strategy {spec.name} "
        f"--tickers {','.join(DEMO_TICKERS)} "
        f"--top {DEMO_TOP} --hold {DEMO_HOLD} "
        f"--start {BACKTEST_START.isoformat()} --end {BACKTEST_END.isoformat()} "
        "--min-price 0 --min-avg-dollar-volume 0"
    )


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _metrics_row(spec: StrategySpec, result: BacktestResult) -> str:
    m = result.metrics
    held = ", ".join(sorted({t.ticker for t in result.trades}))
    return (
        f"| {spec.title} | {m['sharpe']:.2f} | {_pct(m['total_return'])} "
        f"| {_pct(m['benchmark_return'])} | {m['sortino']:.2f} "
        f"| {_pct(m['max_drawdown'])} | {m['trade_count']} | {held} |"
    )


def build_report() -> str:
    results = {s.name: run_demo_backtest(s.name) for s in REPORT_STRATEGIES}

    lines: list[str] = []
    lines.append("# Paper-Backed Factor Strategy Report")
    lines.append("")
    lines.append(
        "Deterministic, fully-offline verification of three paper-backed equity "
        "factor strategies implemented in this screener. **Research, not financial "
        "advice.**"
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python scripts/run_strategy_report.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Every metric below is computed from "
        "`screener.research.factor_demo`, a closed-form synthetic price panel "
        f"({len(DEMO_TICKERS)} names + `{BENCHMARK}` benchmark), so the numbers are "
        "bit-stable. `tests/test_strategy_report.py` pins them with hardcoded "
        "expected values."
    )
    lines.append("")
    lines.append(
        f"- Window: **{BACKTEST_START.isoformat()} → {BACKTEST_END.isoformat()}** "
        f"(daily rolling backtest, >252 trading-day warmup before the window)"
    )
    lines.append(
        f"- Portfolio: **top {DEMO_TOP}** equal-weight slots, **{DEMO_HOLD}-day** hold, "
        f"benchmark `{BENCHMARK}`, no commission/slippage"
    )
    lines.append(
        "- Selection: cross-sectional **factor ranking** via the `rank_score` hook "
        "(not signal-day dollar volume)"
    )
    lines.append("")
    lines.append("## Results vs benchmark")
    lines.append("")
    lines.append(
        "| Strategy | Sharpe | Total Return | Benchmark | Sortino | Max DD | Trades | Held |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for spec in REPORT_STRATEGIES:
        lines.append(_metrics_row(spec, results[spec.name]))
    lines.append("")
    lines.append("## Per-strategy detail")
    lines.append("")
    for spec in REPORT_STRATEGIES:
        m = results[spec.name].metrics
        lines.append(f"### {spec.title}")
        lines.append("")
        lines.append(f"- **Paper:** {spec.paper}")
        lines.append(f"- **Signal:** {spec.signal}")
        lines.append(
            f"- **Sharpe:** {m['sharpe']:.4f} | **Total return:** {_pct(m['total_return'])} "
            f"| **Benchmark:** {_pct(m['benchmark_return'])} | **Trades:** {m['trade_count']}"
        )
        lines.append(
            "- **Reproduction (this report):** `uv run python scripts/run_strategy_report.py`"
        )
        lines.append(f"- **CLI form (live universes):** `{_cli_equivalent(spec)}`")
        lines.append("")
    lines.append("## Skipped (documented)")
    lines.append("")
    lines.append(
        "- **Basu 1/PE value (1977):** requires point-in-time fundamentals "
        "(trailing P/E history), which this codebase does not provide for "
        "backtests (data is daily OHLCV only). The `value` criterion "
        "(`screener/criteria/plugins/value.py`) is available for **live screening** "
        "only — `uv run screener screen -m us -c value`. Backtest intentionally "
        "skipped to avoid look-ahead/survivorship bias."
    )
    lines.append(
        "- **Magic Formula / Piotroski / Sloan / Fama-French:** same reason — no "
        "PIT fundamentals; skipped."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(report)
    print(f"Wrote {_REPORT_PATH}")


if __name__ == "__main__":
    main()
