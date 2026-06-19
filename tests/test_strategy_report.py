"""Acceptance gate: hardcoded Sharpe / total-return / trade-count per strategy.

These pin the EXACT numbers shown in ``results/strategy_report.md`` against the
deterministic demo panel (``screener.research.factor_demo``). If a strategy's
behaviour or the selection machinery changes, this test fails and the report must
be regenerated via ``uv run python scripts/run_strategy_report.py``.
"""

from __future__ import annotations

import pytest

from screener.research.factor_demo import run_demo_backtest
from scripts.run_strategy_report import build_report

# (strategy, expected_sharpe, expected_total_return, expected_trades, held)
_EXPECTED = [
    ("momentum_12_1", 2.564212, 0.328129, 30, {"MOMA", "MOMB", "WILD"}),
    ("low_volatility", 7.528227, 0.193363, 30, {"CALM", "MOMB", "STDY"}),
    ("mom_lowvol_combo", 5.322611, 0.288754, 30, {"MOMA", "MOMB", "STDY"}),
]

_BENCHMARK_RETURN = 0.104718


@pytest.mark.parametrize(
    "name,exp_sharpe,exp_total_return,exp_trades,exp_held", _EXPECTED
)
def test_strategy_report_metrics(
    name: str,
    exp_sharpe: float,
    exp_total_return: float,
    exp_trades: int,
    exp_held: set[str],
) -> None:
    result = run_demo_backtest(name)
    m = result.metrics
    assert m["trade_count"] == exp_trades
    assert m["sharpe"] == pytest.approx(exp_sharpe, rel=1e-4)
    assert m["total_return"] == pytest.approx(exp_total_return, rel=1e-4)
    assert m["benchmark_return"] == pytest.approx(_BENCHMARK_RETURN, rel=1e-4)
    # Every strategy beats the benchmark on this panel...
    assert m["total_return"] > m["benchmark_return"]
    # ...and selects its engineered factor basket (proves cross-sectional ranking).
    assert {t.ticker for t in result.trades} == exp_held


def test_report_is_reproducible() -> None:
    """The generated markdown contains the pinned headline numbers."""
    report = build_report()
    assert "# Paper-Backed Factor Strategy Report" in report
    assert "| 12-1 Momentum | 2.56 | 32.81% | 10.47%" in report
    assert "| Low Volatility | 7.53 | 19.34% | 10.47%" in report
    assert "| Momentum + Low-Vol Combo | 5.32 | 28.88% | 10.47%" in report
    # The skipped fundamentals strategies are documented, not silently dropped.
    assert "Basu 1/PE value (1977)" in report
