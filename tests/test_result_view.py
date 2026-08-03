"""Shared backtest result-view tests."""

from screener.backtester.metrics import result_view


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
