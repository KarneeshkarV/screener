"""Shared backtest result-view tests."""

from screener.backtester.metrics import (
    SIZING_COMPARISON_COLUMNS,
    result_view,
    sizing_comparison_rows,
)


def test_result_view_orders_and_formats_known_metrics():
    view = result_view(
        {
            "final_equity": 110_000.0,
            "starting_equity": 100_000.0,
            "total_return": 0.1,
        }
    )

    assert [row.key for row in view] == [
        "starting_equity",
        "final_equity",
        "total_return",
    ]
    assert [row.kind for row in view] == ["money", "money", "pct"]
    assert [row.formatted for row in view] == ["100,000.00", "110,000.00", "+10.00%"]


def test_result_view_includes_new_metrics_without_renderer_configuration():
    row = result_view({"new_signal_return": 0.1234})[0]

    assert row.key == "new_signal_return"
    assert row.label == "New Signal Return"
    assert row.kind == "pct"
    assert row.formatted == "+12.34%"


def test_sizing_comparison_covers_every_metric_in_result_view_order():
    """The comparison must not show a narrower picture than a single run does."""
    fixed = {
        "starting_equity": 100_000.0,
        "final_equity": 110_000.0,
        "sortino": 2.104,
    }
    reinvested = dict(fixed, final_equity=125_000.0, sortino=1.877)

    rows = sizing_comparison_rows(fixed, reinvested)

    assert [row[0] for row in rows] == [row.label for row in result_view(fixed)]
    assert all(len(row) == 1 + len(SIZING_COMPARISON_COLUMNS) for row in rows)
    assert ("Final Equity", "110,000.00", "125,000.00") in rows
    assert ("Sortino", "+2.104", "+1.877") in rows


def test_sizing_comparison_marks_a_metric_only_one_rule_produced():
    rows = sizing_comparison_rows({"cagr": 0.4432}, {})

    assert rows == (("CAGR", "+44.32%", "-"),)
