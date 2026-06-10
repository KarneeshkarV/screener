from __future__ import annotations

import io

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.seasonality import (
    OTHER_DAYS_LABEL,
    TURN_OF_MONTH_LABEL,
    compute_seasonality,
    report_to_csv,
)
from tests.conftest import StubPriceFetcher


def _bars_from_close(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close.values,
            "high": close.values * 1.01,
            "low": close.values * 0.99,
            "close": close.values,
            "volume": 10_000.0,
        },
        index=close.index,
    )


def _monthly_pattern_close(start: str = "2021-01-01", end: str = "2023-12-31"):
    """Close held constant within each month; +10% in Jan, -5% in Feb."""
    idx = pd.bdate_range(start, end)
    factors = {1: 1.10, 2: 0.95}
    levels = []
    level = 100.0
    seen: list[pd.Period] = []
    for period in idx.to_period("M"):
        if not seen or period != seen[-1]:
            level *= factors.get(period.month, 1.0)
            seen.append(period)
        levels.append(level)
    return pd.Series(levels, index=idx)


def test_monthly_stats_known_pattern():
    close = _monthly_pattern_close()
    report = compute_seasonality(_bars_from_close(close), ticker="AAA")
    by_label = {s.label: s for s in report.monthly}

    jan = by_label["Jan"]
    # First January (2021) has no prior month-end, so only 2022/2023 count.
    assert jan.count == 2
    assert jan.mean == pytest.approx(0.10)
    assert jan.median == pytest.approx(0.10)
    assert jan.best == pytest.approx(0.10)
    assert jan.worst == pytest.approx(0.10)
    assert jan.win_rate == pytest.approx(1.0)

    feb = by_label["Feb"]
    assert feb.count == 3
    assert feb.mean == pytest.approx(-0.05)
    assert feb.win_rate == pytest.approx(0.0)

    mar = by_label["Mar"]
    assert mar.mean == pytest.approx(0.0, abs=1e-12)


def test_day_of_week_known_pattern():
    idx = pd.bdate_range("2022-01-03", periods=260)
    returns = pd.Series(0.0, index=idx)
    returns[idx.dayofweek == 0] = 0.01
    returns[idx.dayofweek == 4] = -0.01
    close = 100.0 * (1.0 + returns).cumprod()

    report = compute_seasonality(_bars_from_close(close), ticker="AAA")
    by_label = {s.label: s for s in report.day_of_week}

    assert by_label["Monday"].mean == pytest.approx(0.01, rel=1e-9)
    assert by_label["Monday"].win_rate == pytest.approx(1.0)
    assert by_label["Friday"].mean == pytest.approx(-0.01, rel=1e-9)
    assert by_label["Friday"].win_rate == pytest.approx(0.0)
    assert by_label["Wednesday"].mean == pytest.approx(0.0, abs=1e-12)
    assert "Saturday" not in by_label and "Sunday" not in by_label


def test_turn_of_month_known_pattern():
    idx = pd.bdate_range("2022-01-03", "2022-12-30")
    # Independently mark the last 3 + first 3 trading days of each month.
    mask = pd.Series(False, index=idx)
    dates = pd.Series(idx, index=idx)
    for _, group in dates.groupby(idx.to_period("M")):
        selected = list(group.index[:3]) + list(group.index[-3:])
        mask.loc[selected] = True
    returns = pd.Series(0.0, index=idx)
    returns[mask] = 0.005
    close = 100.0 * (1.0 + returns).cumprod()

    report = compute_seasonality(_bars_from_close(close), ticker="AAA")
    by_label = {s.label: s for s in report.turn_of_month}

    tom = by_label[TURN_OF_MONTH_LABEL]
    other = by_label[OTHER_DAYS_LABEL]
    assert tom.mean == pytest.approx(0.005, rel=1e-9)
    assert tom.win_rate == pytest.approx(1.0)
    # First trading day's return is dropped by pct_change.
    assert tom.count == int(mask.sum()) - 1
    assert other.mean == pytest.approx(0.0, abs=1e-12)
    assert other.count == int((~mask).sum())


def test_compute_seasonality_rejects_short_history():
    close = pd.Series([100.0], index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError, match="Not enough price history"):
        compute_seasonality(_bars_from_close(close), ticker="AAA")


def test_report_to_csv_sections():
    close = _monthly_pattern_close()
    report = compute_seasonality(_bars_from_close(close), ticker="AAA")
    df = pd.read_csv(io.StringIO(report_to_csv(report)))
    assert set(df["section"]) == {"monthly", "turn_of_month", "day_of_week"}
    assert (df[df["section"] == "monthly"]["label"] == "Jan").any()


def _recent_bars(periods: int = 600) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    returns = pd.Series(0.0, index=idx)
    returns[idx.dayofweek == 0] = 0.01
    close = 100.0 * (1.0 + returns).cumprod()
    return _bars_from_close(close)


def test_cli_seasonality_tables():
    fetcher = StubPriceFetcher({"AAA": _recent_bars()})
    res = CliRunner().invoke(
        cli, ["seasonality", "AAA", "-m", "us", "--years", "2"], obj=fetcher
    )
    assert res.exit_code == 0, res.output
    assert "Seasonality" in res.output
    assert "Monthly returns" in res.output
    assert "Turn-of-month" in res.output
    assert "Day-of-week" in res.output


def test_cli_seasonality_csv():
    fetcher = StubPriceFetcher({"AAA": _recent_bars()})
    res = CliRunner().invoke(
        cli, ["seasonality", "AAA", "--years", "2", "--csv"], obj=fetcher
    )
    assert res.exit_code == 0, res.output
    df = pd.read_csv(io.StringIO(res.output))
    assert set(df["section"]) == {"monthly", "turn_of_month", "day_of_week"}
    monday = df[(df["section"] == "day_of_week") & (df["label"] == "Monday")]
    assert monday["mean_return"].iloc[0] == pytest.approx(0.01, rel=1e-9)


def test_cli_seasonality_notes_short_span():
    fetcher = StubPriceFetcher({"AAA": _recent_bars(periods=300)})
    res = CliRunner().invoke(cli, ["seasonality", "AAA", "--years", "10"], obj=fetcher)
    assert res.exit_code == 0, res.output
    assert "years of data available" in res.stderr


def test_cli_seasonality_india_symbol_mapping():
    fetcher = StubPriceFetcher({"RELIANCE.NS": _recent_bars()})
    res = CliRunner().invoke(
        cli, ["seasonality", "RELIANCE", "-m", "india", "--years", "1"], obj=fetcher
    )
    assert res.exit_code == 0, res.output
    assert "RELIANCE" in res.output


def test_cli_seasonality_no_data_errors():
    fetcher = StubPriceFetcher({})
    res = CliRunner().invoke(cli, ["seasonality", "ZZZ", "--years", "1"], obj=fetcher)
    assert res.exit_code != 0
    assert "No price data" in res.output
