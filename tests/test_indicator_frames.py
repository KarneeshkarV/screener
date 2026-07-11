import numpy as np
import pandas as pd

from screener.indicators.frames import (
    on_balance_volume,
    true_range,
    wilder_atr,
    wilder_rsi,
)


def _ohlc() -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.date_range("2025-01-01", periods=5)
    close = pd.Series([10.0, 12.0, 11.0, 14.0, 14.0], index=index)
    high = pd.Series([11.0, 13.0, 12.0, 15.0, 15.0], index=index)
    low = pd.Series([9.0, 11.0, 10.0, 13.0, 13.0], index=index)
    return high, low, close


def test_true_range_makes_first_bar_policy_explicit() -> None:
    high, low, close = _ohlc()

    pine_range = true_range(high, low, close, first_bar="high_low")
    panel_range = true_range(high, low, close, first_bar="nan")

    assert pine_range.iloc[0] == 2.0
    assert np.isnan(panel_range.iloc[0])
    pd.testing.assert_series_equal(pine_range.iloc[1:], panel_range.iloc[1:])


def test_wilder_indicators_preserve_series_and_panel_shapes() -> None:
    high, low, close = _ohlc()
    panel = close.to_frame("AAA")

    series_rsi = wilder_rsi(close, 2, min_periods=2)
    panel_rsi = wilder_rsi(panel, 2, min_periods=2)
    series_atr = wilder_atr(high, low, close, 2, min_periods=2)
    panel_atr = wilder_atr(
        high.to_frame("AAA"),
        low.to_frame("AAA"),
        panel,
        2,
        min_periods=2,
    )

    pd.testing.assert_series_equal(series_rsi, panel_rsi["AAA"], check_names=False)
    pd.testing.assert_series_equal(series_atr, panel_atr["AAA"], check_names=False)


def test_on_balance_volume_is_shared_panel_primitive() -> None:
    _, _, close = _ohlc()
    close_panel = close.to_frame("AAA")
    volume_panel = pd.DataFrame({"AAA": [100.0, 200.0, 300.0, 400.0, 500.0]})
    volume_panel.index = close.index

    result = on_balance_volume(close_panel, volume_panel)

    np.testing.assert_array_equal(result["AAA"], [0.0, 200.0, -100.0, 300.0, 300.0])
