"""Hand-calculated basket returns, research limitations, and CLI/report contracts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.sector_rotation.analysis import (
    build_sector_basket,
    compute_sector_rotation,
    fetch_sector_rotation,
)
from screener.sector_rotation.catalogue import (
    SectorCatalogue,
    load_sector_catalogue,
    select_sector_groups,
)
from screener.sector_rotation.report import render_sector_report


def _catalogue(**overrides) -> SectorCatalogue:
    data = {
        "version": 1,
        "market": "india",
        "benchmark": "^CRSLDX",
        "benchmark_name": "Nifty 500 price index",
        "observed_on": "2026-10-20",
        "source": "https://www.youtube.com/watch?v=d-I6BG5F-Yg",
        "membership_basis": "curated_research",
        "membership_note": "Current curated research baskets; survivorship bias applies.",
        "groups": [
            {
                "id": "machines",
                "name": "Machines",
                "parent": "Industrials",
                "kind": "industry",
                "video_at": "5:00",
                "members": ["NSE:AAA", "NSE:BBB"],
            },
            {
                "id": "pumps",
                "name": "Pumps",
                "parent": "Industrials",
                "kind": "industry",
                "video_at": "5:00",
                "members": ["NSE:CCC"],
            },
        ],
    }
    data.update(overrides)
    return SectorCatalogue.model_validate(data)


def _prices():
    index = pd.bdate_range("2025-01-01", "2026-01-06")
    return {
        symbol: pd.DataFrame({"close": float(value)}, index=index)
        for symbol, value in {
            "^CRSLDX": 100,
            "AAA.NS": 10,
            "BBB.NS": 1000,
            "CCC.NS": 100,
        }.items()
    }


def _result():
    prices = _prices()
    prices["AAA.NS"].loc["2026-01-06", "close"] = 12
    prices["BBB.NS"].loc["2026-01-06", "close"] = 900
    prices["CCC.NS"].loc["2026-01-06", "close"] = 110
    prices["^CRSLDX"].loc["2026-01-06", "close"] = 102
    return compute_sector_rotation(
        select_sector_groups(_catalogue()),
        prices,
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )


def test_bundled_catalogue_has_video_groups_and_no_duplicate_members():
    catalogue = load_sector_catalogue()
    groups = {group.id for group in catalogue.groups}
    assert {
        "cnc_machines",
        "shipbuilding",
        "niche_cdmo",
        "generic_cdmo",
        "quick_service_restaurants",
        "home_furnishings",
        "transformers",
        "vfx_post_production",
    } <= groups
    assert len(groups) >= 59
    assert catalogue.benchmark == "^CRSLDX"
    assert catalogue.membership_basis == "curated_research"
    assert all(
        len(set(group.members)) == len(group.members) for group in catalogue.groups
    )


def test_selection_rejects_unknown_ids_and_keeps_catalogue_order():
    catalogue = _catalogue()
    assert [
        g.id for g in select_sector_groups(catalogue, ("pumps", "machines")).groups
    ] == ["machines", "pumps"]
    with pytest.raises(ValueError, match="unknown groups: typo"):
        select_sector_groups(catalogue, ("typo",))


def test_catalogue_validation_rejects_duplicates_and_unsafe_ids():
    data = _catalogue().model_dump()
    with pytest.raises(ValueError, match="duplicate group ID"):
        _catalogue(groups=[data["groups"][0], data["groups"][0]])
    for field, value in (
        ("id", "benchmark"),
        ("id", "search"),
        ("id", "<script>"),
        ("members", ["NSE:AAA", "NSE:AAA"]),
        ("members", ["aaa.ns"]),
    ):
        group = dict(data["groups"][0], **{field: value})
        with pytest.raises(ValueError):
            _catalogue(groups=[group])


def test_load_replacement_catalogue_and_invalid_yaml(tmp_path):
    path = tmp_path / "groups.yaml"
    path.write_text(_catalogue().model_dump_json())
    assert load_sector_catalogue(path).groups[0].id == "machines"
    path.write_text("groups: [")
    with pytest.raises(ValueError, match="Sector catalogue: invalid configuration"):
        load_sector_catalogue(path)


def test_equal_weight_uses_returns_not_raw_prices_and_weights_drift():
    index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    closes = pd.DataFrame({"a": [10, 20, 20], "b": [1000, 1000, 2000]}, index=index)
    curve, coverage = build_sector_basket(closes)
    np.testing.assert_allclose(curve, [100, 150, 200])
    np.testing.assert_allclose(coverage, [1, 1, 1])
    # Daily equal-weight rebalancing would incorrectly produce 225 at the end.


def test_month_end_rebalance_uses_prior_close_and_no_future_prices():
    index = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"])
    closes = pd.DataFrame(
        {"a": [10, 20, 20, 40], "b": [1000, 1000, 2000, 2000]}, index=index
    )
    curve, _ = build_sector_basket(closes)
    np.testing.assert_allclose(curve, [100, 150, 225, 300])
    truncated, _ = build_sector_basket(closes.iloc[:3])
    pd.testing.assert_series_equal(curve.iloc[:3], truncated)


def test_missing_held_price_stops_curve_not_zero_return_or_forward_fill():
    closes = pd.DataFrame(
        {"a": [10, 11, np.nan, 13], "b": [100, 100, 110, 110]},
        index=pd.bdate_range("2026-01-05", periods=4),
    )
    curve, coverage = build_sector_basket(closes, minimum_coverage=0.5)
    assert curve.iloc[1] == pytest.approx(105)
    assert curve.iloc[2:].isna().all()
    assert coverage.iloc[2] == 0.5


@pytest.mark.parametrize("invalid", [0.0, -10.0, float("inf")])
def test_invalid_prices_cannot_become_extreme_basket_returns(invalid):
    closes = pd.DataFrame(
        {"a": [10.0, invalid]}, index=pd.bdate_range("2026-01-05", periods=2)
    )
    curve, coverage = build_sector_basket(closes)
    assert pd.isna(curve.iloc[-1])
    assert coverage.iloc[-1] == 0


def test_unsorted_basket_sessions_fail():
    closes = pd.DataFrame(
        {"a": [10, 11]}, index=pd.to_datetime(["2026-01-06", "2026-01-05"])
    )
    with pytest.raises(ValueError, match="unique and sorted"):
        build_sector_basket(closes)


def test_new_listing_enters_only_after_month_end_rebalance():
    closes = pd.DataFrame(
        {"a": [10, 10, 10, 10], "b": [np.nan, 10, 20, 40]},
        index=pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03"]),
    )
    curve, _ = build_sector_basket(closes, minimum_coverage=0.5)
    np.testing.assert_allclose(curve, [100, 100, 150, 250])


def test_insufficient_initial_coverage_never_rebases_late_to_hide_missing_history():
    closes = pd.DataFrame(
        {"a": [10, 11], "b": [np.nan, 20]},
        index=pd.bdate_range("2026-01-05", periods=2),
    )
    curve, _ = build_sector_basket(closes)
    assert curve.isna().all()
    with pytest.raises(ValueError, match="coverage"):
        build_sector_basket(closes, minimum_coverage=0)


def test_relative_metrics_breadth_and_rank_match_hand_calculation():
    result = _result()
    rows = result.summary.set_index("group_id")
    assert rows.loc["machines", "return_pct"] == pytest.approx(5)
    assert rows.loc["machines", "excess_return_pp"] == pytest.approx(3)
    assert rows.loc["machines", "relative_strength_pct"] == pytest.approx(
        (1.05 / 1.02 - 1) * 100
    )
    assert rows.loc["pumps", "rank"] == 1
    assert rows.loc["machines", "rank"] == 2
    assert rows.loc["machines", "breadth_sma50_pct"] == 50
    assert rows.loc["machines", "breadth_sma200_members"] == 2
    assert result.members.shape[0] == 3
    assert any("survivorship" in warning for warning in result.warnings)
    assert result.curves.iloc[0].eq(100).all()


def test_missing_group_member_stays_unranked_and_unknown_breadth_is_not_false():
    prices = _prices()
    del prices["CCC.NS"]
    result = compute_sector_rotation(
        select_sector_groups(_catalogue()),
        prices,
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    row = result.summary.set_index("group_id").loc["pumps"]
    assert pd.isna(row["rank"])
    assert pd.isna(row["breadth_sma50_pct"])
    assert row["breadth_sma50_members"] == 0
    assert row["status"] == "insufficient_data"
    assert (
        result.members.loc[result.members.ticker == "NSE:CCC", "above_sma50"].iloc[0]
        is None
    )


@pytest.mark.parametrize("invalid", [0, 1.1, -0.5])
def test_invalid_coverage_rejected(invalid):
    with pytest.raises(ValueError, match="minimum coverage"):
        compute_sector_rotation(
            select_sector_groups(_catalogue()),
            _prices(),
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
            minimum_coverage=invalid,
        )


def test_invalid_benchmark_and_dates_fail_explicitly():
    selection = select_sector_groups(_catalogue())
    for prices, start, end, message in (
        ({}, date(2026, 1, 2), date(2026, 1, 6), "benchmark prices"),
        (_prices(), date(2026, 1, 6), date(2026, 1, 2), "start must be before"),
        (_prices(), date(2024, 1, 1), date(2026, 1, 6), "requested start"),
        (_prices(), date(2026, 1, 2), date(2026, 2, 6), "benchmark end is stale"),
    ):
        with pytest.raises(ValueError, match=message):
            compute_sector_rotation(selection, prices, start=start, end=end)


def test_future_bars_do_not_change_as_of_result():
    selection = select_sector_groups(_catalogue())
    prices = _prices()
    first = compute_sector_rotation(
        selection, prices, start=date(2026, 1, 2), end=date(2026, 1, 5)
    )
    for frame in prices.values():
        frame.loc["2026-01-06", "close"] = 99999
    second = compute_sector_rotation(
        selection, prices, start=date(2026, 1, 2), end=date(2026, 1, 5)
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.curves, second.curves)


class StubSectorFetcher:
    def __init__(self):
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append((tickers, start, end))
        index = pd.bdate_range(start, end)
        return {
            ticker: pd.DataFrame(
                {
                    "close": 100
                    + np.arange(len(index)) * (0.2 if ticker.startswith("^") else 0.4)
                },
                index=index,
            )
            for ticker in tickers
        }


def test_fetch_deduplicates_symbols_and_includes_breadth_warmup():
    fetcher = StubSectorFetcher()
    result = fetch_sector_rotation(
        select_sector_groups(_catalogue()),
        fetcher,
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    tickers, start, end = fetcher.calls[0]
    assert len(tickers) == len(set(tickers)) == 4
    assert start == date(2024, 11, 28)
    assert end == date(2026, 1, 6)
    assert result.summary.breadth_sma200_members.sum() == 3


def test_report_is_offline_searchable_and_escapes_catalogue_text(tmp_path):
    catalogue = _catalogue()
    groups = list(catalogue.groups)
    groups[0] = groups[0].model_copy(update={"name": '<img src=x onerror="alert(1)">'})
    catalogue = catalogue.model_copy(update={"groups": tuple(groups)})
    result = compute_sector_rotation(
        select_sector_groups(catalogue),
        _prices(),
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    path = tmp_path / "report.html"
    render_sector_report(result, path)
    text = path.read_text()
    assert "<img src=x" not in text
    assert "&lt;img src=x" in text
    assert "<svg" in text and "<details>" in text
    assert 'id="search"' in text and 'id="parent"' in text
    assert "survivorship" in text
    assert "<script src=" not in text
    assert "benchmark" in text


def test_cli_list_is_offline_and_invalid_group_fails():
    runner = CliRunner()
    listed = runner.invoke(cli, ["sector-rotation", "--list-groups"])
    assert listed.exit_code == 0, listed.output
    assert "cnc_machines" in listed.output and "NSE:JYOTICNC" in listed.output
    bad = runner.invoke(cli, ["sector-rotation", "--group", "not_real"])
    assert bad.exit_code != 0 and "unknown groups" in bad.output
    future = runner.invoke(cli, ["sector-rotation", "--as-of", "2099-01-01"])
    assert future.exit_code != 0 and "before today" in future.output


def test_cli_outputs_csv_html_and_member_csv_with_stub_prices(tmp_path):
    report = tmp_path / "report.html"
    members = tmp_path / "members.csv"
    result = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--group",
            "cnc_machines",
            "--as-of",
            "2026-01-06",
            "--period",
            "1m",
            "--csv",
            "--report",
            str(report),
            "--members-csv",
            str(members),
        ],
        obj=StubSectorFetcher(),
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("rank,group_id,")
    assert "survivorship" in result.stderr
    assert "<svg" in report.read_text()
    assert "NSE:JYOTICNC" in members.read_text()


def test_cli_agent_output_is_bounded_and_spills_full_metrics(tmp_path, monkeypatch):
    from screener import agentio

    monkeypatch.setenv("SCREENER_AGENT_DIR", str(tmp_path))
    result = CliRunner().invoke(
        cli,
        ["--agent", "sector-rotation", "--as-of", "2026-01-06"],
        obj=StubSectorFetcher(),
    )
    agentio.reset()
    assert result.exit_code == 0, result.output
    assert len(result.stdout.splitlines()) < 20
    files = list(Path(tmp_path).glob("sector-rotation-*.csv"))
    assert len(files) == 1
    assert len(pd.read_csv(files[0])) == len(load_sector_catalogue().groups)


def test_missing_benchmark_month_end_cannot_change_rebalance_silently():
    index = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02"])
    prices = {
        "^CRSLDX": pd.DataFrame({"close": [100, 100]}, index=index[[0, 2]]),
        "AAA.NS": pd.DataFrame({"close": [10, 20, 20]}, index=index),
        "BBB.NS": pd.DataFrame({"close": [100, 100, 200]}, index=index),
    }
    with pytest.raises(ValueError, match="benchmark sessions missing.*2026-01-30"):
        compute_sector_rotation(
            select_sector_groups(_catalogue(), ("machines",)),
            prices,
            start=date(2026, 1, 29),
            end=date(2026, 2, 2),
        )


def test_zero_volume_holiday_rows_do_not_imply_missing_benchmark_session():
    prices = _prices()
    holiday = pd.Timestamp("2026-01-05")
    prices["^CRSLDX"] = prices["^CRSLDX"].drop(holiday)
    for ticker in ("AAA.NS", "BBB.NS", "CCC.NS"):
        prices[ticker]["volume"] = 100
        prices[ticker].loc[holiday, "volume"] = 0
    result = compute_sector_rotation(
        select_sector_groups(_catalogue()),
        prices,
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    assert result.summary.status.eq("ok").all()
    prices["AAA.NS"].loc[holiday, "volume"] = 1
    with pytest.raises(ValueError, match="benchmark sessions missing"):
        compute_sector_rotation(
            select_sector_groups(_catalogue()),
            prices,
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
        )


def test_invalid_benchmark_warmup_does_not_block_valid_return_window():
    prices = _prices()
    prices["^CRSLDX"].loc["2025-01-02", "close"] = float("nan")
    result = compute_sector_rotation(
        select_sector_groups(_catalogue()),
        prices,
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    assert result.summary.status.eq("ok").all()
    prices["^CRSLDX"].loc["2026-01-05", "close"] = float("nan")
    with pytest.raises(ValueError, match="invalid in the analysis window"):
        compute_sector_rotation(
            select_sector_groups(_catalogue()),
            prices,
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
        )


@pytest.mark.parametrize(
    "as_of,expected_end", [("2026-01-04", "2026-01-02"), ("2026-01-05", "2026-01-02")]
)
def test_cli_one_day_resolves_weekend_and_holiday_end_before_start(as_of, expected_end):
    class HolidayFetcher(StubSectorFetcher):
        def fetch(self, tickers, start, end):
            return {
                ticker: frame.drop(pd.Timestamp("2026-01-05"), errors="ignore")
                for ticker, frame in super().fetch(tickers, start, end).items()
            }

    result = CliRunner().invoke(
        cli,
        [
            "sector-rotation",
            "--group",
            "cnc_machines",
            "--period",
            "1d",
            "--as-of",
            as_of,
            "--csv",
        ],
        obj=HolidayFetcher(),
    )
    assert result.exit_code == 0, result.output
    assert f"2026-01-01,{expected_end}" in result.stdout
