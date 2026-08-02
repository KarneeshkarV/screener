"""Agent-mode resolution, spill paths, and the bounded backtest digest."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from rich.console import Console

from screener import agentio


@pytest.fixture(autouse=True)
def _clean_agent_state(monkeypatch):
    """Every test starts from no explicit override and no agent env vars."""
    agentio.reset()
    for name in agentio._AGENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SCREENER_AGENT", raising=False)
    monkeypatch.delenv("SCREENER_AGENT_DETAIL", raising=False)
    yield
    agentio.reset()


def test_explicit_flag_beats_env_and_autodetect(monkeypatch):
    monkeypatch.setenv("SCREENER_AGENT", "1")
    monkeypatch.setenv("CLAUDECODE", "1")
    agentio.configure(enabled=False)
    assert agentio.is_agent_mode() is False


def test_env_beats_autodetect(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("SCREENER_AGENT", "0")
    assert agentio.is_agent_mode() is False


@pytest.mark.parametrize("name", agentio._AGENT_ENV_VARS)
def test_autodetect_from_each_harness_var(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert agentio.is_agent_mode() is True


def test_pytest_guard_blocks_autodetect(monkeypatch):
    """A suite run *from* an agent harness must not inherit agent mode.

    Without this guard `just test` behaves differently locally than in CI,
    because CLAUDECODE=1 leaks into all 121 CliRunner invocations.
    """
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_x")
    assert agentio.is_agent_mode() is False


def test_pytest_guard_does_not_block_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_x")
    monkeypatch.setenv("SCREENER_AGENT", "1")
    assert agentio.is_agent_mode() is True


def test_agent_mode_off_by_default():
    assert agentio.is_agent_mode() is False


def test_detail_defaults_to_head():
    """`head` beat both alternatives on graded backtest questions."""
    assert agentio.detail_level() == "head"
    assert agentio.DEFAULT_DETAIL == "head"


def test_detail_from_env(monkeypatch):
    monkeypatch.setenv("SCREENER_AGENT_DETAIL", "SUMMARY")
    assert agentio.detail_level() == "summary"


def test_unknown_detail_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SCREENER_AGENT_DETAIL", "verbose")
    assert agentio.detail_level() == agentio.DEFAULT_DETAIL


def test_spill_dir_defaults_under_home(monkeypatch):
    monkeypatch.delenv("SCREENER_AGENT_DIR", raising=False)
    assert agentio.spill_dir().name == "tmp"


def test_spill_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENER_AGENT_DIR", str(tmp_path))
    assert agentio.spill_dir() == tmp_path


def test_run_key_ignores_presentation_flags(monkeypatch):
    """Detail level changes the digest, never the spilled data."""
    base = ["backtest-rolling", "-m", "us"]
    monkeypatch.setattr("sys.argv", ["screener", *base])
    plain = agentio.run_key()
    monkeypatch.setattr(
        "sys.argv", ["screener", "--agent", "--agent-detail", "full", *base]
    )
    assert agentio.run_key() == plain
    monkeypatch.setattr("sys.argv", ["screener", "--agent-detail=head", *base])
    assert agentio.run_key() == plain


def test_run_key_tracks_real_arguments(monkeypatch):
    monkeypatch.setattr("sys.argv", ["screener", "backtest-rolling", "-m", "us"])
    us = agentio.run_key()
    monkeypatch.setattr("sys.argv", ["screener", "backtest-rolling", "-m", "india"])
    assert agentio.run_key() != us


def test_spill_writes_csv_and_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SCREENER_AGENT_DIR", str(tmp_path / "nested"))
    df = pd.DataFrame({"ticker": ["AAPL"], "return_pct": [0.05]})
    path = agentio.spill(df, "backtest-us")
    assert path.exists()
    assert pd.read_csv(path)["ticker"].tolist() == ["AAPL"]


def test_kv_line_packs_pairs():
    lines = agentio.kv_line([("a", 1), ("b", 2), ("c", 3)], per_line=2)
    assert lines == ["a=1 b=2", "c=3"]


def _attribution_output(trades: pd.DataFrame) -> str:
    """Render the per-ticker attribution block into a capturable console."""
    from rich.console import Console

    from screener.backtester.display import _print_ticker_attribution

    console = Console(width=200, no_color=True, record=True)
    _print_ticker_attribution(trades, console)
    return console.export_text()


def test_agent_digest_includes_every_shared_metric(monkeypatch):
    from screener.backtester.display import print_backtest
    from screener.backtester.models import BacktestResult

    output = Console(width=200, no_color=True, record=True)
    monkeypatch.setattr(agentio, "get_console", lambda: output)
    agentio.configure(enabled=True)
    result = BacktestResult.model_construct(
        config=SimpleNamespace(
            market="us",
            as_of="2024-03-01",
            hold=5,
            top=2,
            benchmark="SPY",
        ),
        trades=[],
        metrics={"starting_equity": 100_000.0, "final_equity": 110_000.0},
        warnings=[],
    )

    print_backtest(result)

    text = output.export_text()
    assert "Starting Capital" in text
    assert "Final Equity" in text


def test_attribution_ranks_worst_ticker_first():
    """The measured failure was naming the wrong ticker as the biggest loser."""
    trades = pd.DataFrame(
        {
            "ticker": ["NVDA", "NVDA", "MSFT", "AAPL"],
            "pnl": [-10_000.0, -5_308.15, -11_786.87, 13_290.29],
            "return_pct": [-0.05, -0.02, -0.14, 0.08],
        }
    )
    lines = [line for line in _attribution_output(trades).splitlines() if line.strip()]
    assert lines[0].startswith("pnl_by_ticker")
    assert lines[1].split()[0] == "NVDA"
    assert "-15,308.15" in lines[1]
    assert lines[-1].split()[0] == "AAPL"


def test_attribution_reports_win_rate_per_ticker():
    trades = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "pnl": [1.0, 1.0, -1.0, -1.0],
            "return_pct": [0.01, 0.01, -0.01, -0.01],
        }
    )
    assert "50.0" in _attribution_output(trades)


def test_attribution_stays_bounded_for_large_universes():
    """A 500-name universe must not turn the digest into 500 lines."""
    size = 60
    trades = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(size)],
            "pnl": [float(i) for i in range(size)],
            "return_pct": [0.01] * size,
        }
    )
    output = _attribution_output(trades)
    rows = [line for line in output.splitlines() if line.startswith("  ")]
    assert len(rows) == 10
    assert "worst/best 5 of 60" in output


def test_attribution_skips_when_pnl_absent():
    trades = pd.DataFrame({"ticker": ["AAPL"], "return_pct": [0.01]})
    assert _attribution_output(trades).strip() == ""
