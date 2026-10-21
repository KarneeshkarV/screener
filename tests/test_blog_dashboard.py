"""Check the offline report exports finite, self-contained result data."""

import json

import pandas as pd

from scripts import build_blog_dashboard


def test_blog_dashboard_is_self_contained_and_first_month_is_present(
    tmp_path, monkeypatch
):
    root = tmp_path
    folder = root / "reports/blog_momentum/chan/us_etfs_chan_return_12m"
    folder.mkdir(parents=True)
    row = {
        "cell": folder.name,
        "strategy": "chan_return_12m",
        "starting_equity": 100,
        "sharpe": float("nan"),
    }
    (folder.parent / "summary.json").write_text(json.dumps([row]))
    (folder / "warnings.json").write_text("[]")
    pd.Series(
        [101.0, 105.0], index=pd.to_datetime(["2020-01-31", "2020-02-28"])
    ).to_csv(folder / "equity.csv")
    (root / "scripts").mkdir()
    (root / "scripts/blog_dashboard.html").write_text("const DATA=__RESULTS__;")
    monkeypatch.setattr(build_blog_dashboard, "ROOT", root)
    output = build_blog_dashboard.build_blog_dashboard().read_text()
    payload = json.loads(output.removeprefix("const DATA=").removesuffix(";"))
    assert payload[0]["sharpe"] is None
    assert payload[0]["returns"][0] == ["2020-01-31", 1.0]
    assert "__RESULTS__" not in output


def test_render_template_injects_slate_theme_and_echarts():
    template = (
        "<style>/*__THEME_CSS__*/</style><!--__ECHARTS__-->"
        "<script>const DATA=__RESULTS__;</script>"
    )
    output = build_blog_dashboard.render_template(
        template, [{"strategy": "chan_return_12m", "sharpe": None}]
    )
    assert "--series-1:" in output
    assert "echarts" in output
    assert "__THEME_CSS__" not in output
    assert "__ECHARTS__" not in output
    assert "__RESULTS__" not in output
    assert (
        '"strategy": "chan_return_12m"' in output
        or '"strategy":"chan_return_12m"' in output
    )
