"""Contract tests for the shared report theme (:mod:`screener.report_theme`)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from screener import report_theme as rt
from screener.html_report import html_page

_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def _spec(html: str, chart_id: str) -> dict:
    match = re.search(
        rf'<script type="application/json" id="{chart_id}-spec">(.*?)</script>', html
    )
    assert match, f"no spec for {chart_id}"
    return json.loads(match.group(1))


def test_every_token_has_a_light_and_a_dark_step() -> None:
    light, _, dark = rt.THEME_CSS.partition("prefers-color-scheme: dark")
    for token in rt.TOKENS:
        assert f"--{token}:" in light, token
        assert f"--{token}:" in dark, token


def test_html_page_always_carries_the_theme_and_only_inlines_charts_on_request() -> (
    None
):
    plain = html_page("t", ".x{}", "<p>hi</p>")
    assert "--series-1:" in plain
    assert "echarts" not in plain

    charted = html_page(
        "t", "", rt.chart_html("c", rt.bar_chart(["a"], [1])), charts=True
    )
    assert "screenerCharts" in charted
    # The bundle is inlined, so a report opens offline.
    assert "<script src=" not in charted
    assert len(rt.echarts_js()) > 500_000


def test_chart_options_reference_tokens_never_raw_colours() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    options = [
        rt.line_chart(
            dates,
            [
                {"name": "S", "values": range(5), "area": True},
                {"name": "B", "values": range(5)},
            ],
        ),
        rt.bar_chart(
            ["a", "b"],
            [1, -1],
            colors=rt.signed_colors([1, -1]),
            horizontal=True,
            labels=True,
        ),
        rt.histogram_chart(np.linspace(-1, 1, 50)),
    ]
    for option in options:
        assert not _HEX.search(json.dumps(option)), option


def test_chart_json_is_strict_and_cannot_break_out_of_its_script() -> None:
    html = rt.chart_html(
        "x",
        rt.bar_chart(
            ["</script><img src=x onerror=1>", "b&c"], [float("nan"), np.int64(3)]
        ),
    )
    assert "</script><img" not in html
    spec = _spec(html, "x")
    categories = (
        spec["yAxis"]["data"]
        if spec["xAxis"]["type"] == "value"
        else spec["xAxis"]["data"]
    )
    assert categories[0] == "</script><img src=x onerror=1>"
    assert spec["series"][0]["data"][0]["value"] is None
    assert spec["series"][0]["data"][1]["value"] == 3


def test_horizontal_bars_draw_the_first_category_at_the_top() -> None:
    option = rt.bar_chart(["first", "second"], [2, 1], horizontal=True)
    # ECharts draws category index 0 at the bottom of a vertical category axis.
    assert option["yAxis"]["data"] == ["second", "first"]


def test_signed_histogram_colours_only_non_positive_bins_as_negative() -> None:
    option = rt.histogram_chart([-2.0, -1.0, 1.0, 2.0], bins=4)
    colours = [bar["itemStyle"]["color"] for bar in option["series"][0]["data"]]
    assert colours == ["token:neg", "token:neg", "token:series-1", "token:series-1"]
    assert sum(bar["value"] for bar in option["series"][0]["data"]) == 4


def test_diverging_cells_mix_the_pole_into_the_neutral_midpoint() -> None:
    style, cls = rt.diverging_cell_style(-0.1, 0.1)
    assert "var(--neg)" in style and "var(--div-mid)" in style
    assert cls == "strong"
    assert rt.diverging_cell_style(float("nan"), 0.1) == ("", "")


def test_a_constant_sample_is_one_bar_not_a_spread_of_invented_bins() -> None:
    option = rt.histogram_chart([0.0] * 20, bins=24)
    assert [bar["value"] for bar in option["series"][0]["data"]] == [20]


def test_a_flat_drawdown_keeps_a_small_floor_instead_of_autoscaling() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    flat = rt.drawdown_chart(dates, [0.0, 0.0, 0.0])
    assert (flat["yAxis"]["min"], flat["yAxis"]["max"]) == (-0.01, 0)
    deep = rt.drawdown_chart(dates, [0.0, -0.2, -0.1])
    assert deep["yAxis"]["min"] == -0.2


def test_end_labels_are_dropped_when_the_lines_finish_on_top_of_each_other() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    same = rt.line_chart(
        dates, [{"name": "A", "values": [1, 2, 3]}, {"name": "B", "values": [1, 2, 3]}]
    )
    apart = rt.line_chart(
        dates, [{"name": "A", "values": [1, 2, 3]}, {"name": "B", "values": [1, 1, 1]}]
    )
    assert all("endLabel" not in s for s in same["series"])
    assert all("endLabel" in s for s in apart["series"])


def test_a_page_toggle_can_force_either_scheme_over_the_os_setting() -> None:
    assert ':root[data-theme="dark"]' in rt.THEME_CSS
    assert ':root:where(:not([data-theme="light"]))' in rt.THEME_CSS
