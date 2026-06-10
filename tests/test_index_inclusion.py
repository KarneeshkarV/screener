"""Tests for the S&P 500 index-inclusion event study."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

import screener.commands.index_inclusion as index_inclusion_module
from screener.cli import cli
from screener.index_inclusion import run_inclusion_study


def _frame(index: pd.DatetimeIndex, closes: pd.Series | float) -> pd.DataFrame:
    if isinstance(closes, (int, float)):
        closes = pd.Series(float(closes), index=index)
    return pd.DataFrame({"close": closes.to_numpy()}, index=index)


def _growth_closes(index: pd.DatetimeIndex, daily: float) -> pd.Series:
    return pd.Series(
        [100.0 * (1.0 + daily) ** i for i in range(len(index))], index=index
    )


IDX = pd.bdate_range("2020-01-01", periods=300)
AS_OF = IDX[-1].date()


def test_run_inclusion_study_computes_excess_vs_flat_benchmark(stub_fetcher_factory):
    added = IDX[100].date()
    fetcher = stub_fetcher_factory(
        {
            "NEWCO": _frame(IDX, _growth_closes(IDX, 0.01)),
            "SPY": _frame(IDX, 100.0),
        }
    )
    study = run_inclusion_study({"NEWCO": added}, fetcher, years=5, as_of=AS_OF)

    assert study.skipped == 0
    assert len(study.events) == 1
    event = study.events[0]
    assert event.symbol == "NEWCO"
    assert event.date_added == added
    # Baseline is 5 trading days before the addition; SPY is flat so the
    # excess equals the raw cumulative return over (horizon + 5) sessions.
    for horizon in (5, 20, 60):
        assert event.excess[horizon] == pytest.approx(1.01 ** (horizon + 5) - 1.0)
    assert [s.horizon for s in study.summaries] == [5, 20, 60]
    assert study.summaries[0].hit_rate == 1.0
    assert study.summaries[0].mean == pytest.approx(1.01**10 - 1.0)
    assert study.summaries[0].median == pytest.approx(1.01**10 - 1.0)


def test_run_inclusion_study_subtracts_benchmark_return(stub_fetcher_factory):
    added = IDX[100].date()
    fetcher = stub_fetcher_factory(
        {
            "NEWCO": _frame(IDX, _growth_closes(IDX, 0.01)),
            "SPY": _frame(IDX, _growth_closes(IDX, 0.005)),
        }
    )
    study = run_inclusion_study({"NEWCO": added}, fetcher, years=5, as_of=AS_OF)

    assert len(study.events) == 1
    for horizon in (5, 20, 60):
        expected = (1.01 ** (horizon + 5) - 1.0) - (1.005 ** (horizon + 5) - 1.0)
        assert study.events[0].excess[horizon] == pytest.approx(expected)


def test_run_inclusion_study_skips_insufficient_data(stub_fetcher_factory):
    good_added = IDX[100].date()
    recent_added = IDX[-10].date()  # fewer than 60 sessions after the addition
    fetcher = stub_fetcher_factory(
        {
            "GOOD": _frame(IDX, _growth_closes(IDX, 0.01)),
            "RECENT": _frame(IDX, 100.0),
            # "NODATA" intentionally absent from the stub.
            "SPY": _frame(IDX, 100.0),
        }
    )
    membership: dict[str, date | None] = {
        "GOOD": good_added,
        "RECENT": recent_added,
        "NODATA": IDX[120].date(),
    }
    study = run_inclusion_study(membership, fetcher, years=5, as_of=AS_OF)

    assert [event.symbol for event in study.events] == ["GOOD"]
    assert study.skipped == 2


def test_run_inclusion_study_ignores_old_and_undated_additions(stub_fetcher_factory):
    fetcher = stub_fetcher_factory({"SPY": _frame(IDX, 100.0)})
    membership: dict[str, date | None] = {
        "ANCIENT": date(1990, 1, 2),  # outside the trailing window
        "UNDATED": None,
    }
    study = run_inclusion_study(membership, fetcher, years=5, as_of=AS_OF)

    assert study.events == []
    assert study.skipped == 0
    assert study.summaries == []


# ── CLI ─────────────────────────────────────────────────────────────────────


def _patch_cli_inputs(monkeypatch, stub_fetcher_factory):
    """Wire the command to synthetic membership + prices anchored to today."""
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)
    added = idx[100].date()
    fetcher = stub_fetcher_factory(
        {
            "NEWCO": _frame(idx, _growth_closes(idx, 0.01)),
            "SPY": _frame(idx, 100.0),
        }
    )
    membership: dict[str, date | None] = {
        "NEWCO": added,
        "NODATA": idx[120].date(),
        "UNDATED": None,
    }
    monkeypatch.setattr(
        index_inclusion_module, "load_sp500_membership", lambda **kwargs: membership
    )
    monkeypatch.setattr(
        index_inclusion_module, "build_price_fetcher", lambda *a, **kw: fetcher
    )
    return added


def test_cli_prints_summary_table_and_limitation(monkeypatch, stub_fetcher_factory):
    _patch_cli_inputs(monkeypatch, stub_fetcher_factory)
    runner = CliRunner()
    res = runner.invoke(cli, ["index-inclusion", "-m", "us"], catch_exceptions=False)

    assert res.exit_code == 0
    assert "+60d" in res.output
    assert "Events: 1" in res.output
    assert "Skipped (insufficient price data): 1" in res.output
    assert "date added" in res.output


def test_cli_csv_outputs_per_event_rows(monkeypatch, stub_fetcher_factory):
    added = _patch_cli_inputs(monkeypatch, stub_fetcher_factory)
    runner = CliRunner()
    res = runner.invoke(
        cli, ["index-inclusion", "-m", "us", "--csv"], catch_exceptions=False
    )

    assert res.exit_code == 0
    assert "symbol,date_added,excess_5d,excess_20d,excess_60d" in res.output
    assert f"NEWCO,{added.isoformat()}" in res.output


def test_cli_rejects_non_positive_years(monkeypatch, stub_fetcher_factory):
    _patch_cli_inputs(monkeypatch, stub_fetcher_factory)
    runner = CliRunner()
    res = runner.invoke(cli, ["index-inclusion", "-m", "us", "--years", "0"])

    assert res.exit_code != 0
    assert "--years must be a positive integer." in res.output
