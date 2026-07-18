from __future__ import annotations

from datetime import date

import pandas as pd

from screener.screen_aliases import SCREEN_ALIASES
from screener.options.criteria import (
    HIGH_PCR,
    OPTIONS_CRITERIA,
    screen_options_criterion,
)


def _row(symbol: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "as_of": pd.Timestamp("2026-07-06"),
        "SYMBOL": symbol,
        "source": "fixture",
        "history_days": 6,
        "call_oi_change": float("nan"),
        "put_oi_change": float("nan"),
        "call_writing_near_spot": float("nan"),
        "put_writing_near_spot": float("nan"),
        "call_oi": float("nan"),
        "put_oi": float("nan"),
        "pcr": float("nan"),
    }
    row.update(overrides)
    return row


def _bearish_panel() -> pd.DataFrame:
    # EXACT: call writing near spot dominates put writing (defending resistance).
    exact = _row(
        "EXA",
        call_writing_near_spot=80.0,
        put_writing_near_spot=10.0,
        call_oi_change=50.0,
        put_oi_change=20.0,
    )
    # PROXY: no writing direction, call OI added faster than put OI.
    proxy = _row(
        "PRX",
        call_writing_near_spot=None,
        put_writing_near_spot=None,
        call_oi_change=250.0,
        put_oi_change=100.0,
    )
    # NEGATIVE: put writing dominates (this is bullish, not bearish).
    negative = _row(
        "NEG",
        call_writing_near_spot=10.0,
        put_writing_near_spot=90.0,
        call_oi_change=30.0,
        put_oi_change=200.0,
    )
    return pd.DataFrame([exact, proxy, negative])


def test_bearish_oi_buildup_exact_and_proxy_paths():
    result = screen_options_criterion(
        "bearish_oi_buildup",
        market="india",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=_bearish_panel(),
    )
    basis = dict(zip(result.frame["SYMBOL"], result.frame["oi_signal_basis"]))
    assert basis == {"EXA": "exact_call_writing", "PRX": "snapshot_diff_proxy"}
    # Sorted by bearish_oi_score descending: PRX (250 from call_oi_change) > EXA (80).
    assert result.frame["SYMBOL"].tolist() == ["PRX", "EXA"]
    assert result.frame.iloc[0]["bearish_oi_score"] == 250.0
    assert result.frame.iloc[1]["bearish_oi_score"] == 80.0
    assert (result.frame["signal"] == "bearish_oi_buildup").all()
    assert result.frame["coverage_days"].tolist() == [6, 6]
    assert "India uses exact" in result.message


def test_bearish_oi_buildup_no_baseline_message():
    panel = pd.DataFrame(
        [
            _row("AAA", call_writing_near_spot=None, put_writing_near_spot=None),
            _row("BBB", call_writing_near_spot=None, put_writing_near_spot=None),
        ]
    )
    result = screen_options_criterion(
        "bearish_oi_buildup",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=panel,
    )
    assert result.frame.empty
    assert "no OI-change baseline" in result.message


def _pcr_panel() -> pd.DataFrame:
    high = _row("HIG", call_oi=1000.0, put_oi=1600.0, pcr=1.6)
    mid = _row("MID", call_oi=1000.0, put_oi=1400.0, pcr=1.4)
    low = _row("LOW", call_oi=1000.0, put_oi=1000.0, pcr=1.0)
    return pd.DataFrame([high, mid, low])


def test_high_pcr_reversal_selects_and_sorts_by_pcr():
    result = screen_options_criterion(
        "high_pcr_reversal",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=_pcr_panel(),
    )
    # Only HIG and MID clear HIGH_PCR (1.3); LOW is below threshold.
    assert result.frame["SYMBOL"].tolist() == ["HIG", "MID"]
    assert result.frame.iloc[0]["pcr"] == 1.6
    assert (result.frame["signal"] == "high_pcr_reversal").all()
    assert result.frame["coverage_days"].tolist() == [6, 6]
    assert str(HIGH_PCR) in result.message
    assert "Contrarian" in result.message


def test_high_pcr_reversal_below_threshold_empty():
    panel = pd.DataFrame([_row("AAA", call_oi=1000.0, put_oi=1100.0, pcr=1.1)])
    result = screen_options_criterion(
        "high_pcr_reversal",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=panel,
    )
    assert result.frame.empty
    assert "Contrarian" in result.message


def test_high_pcr_reversal_thin_no_coverage_message():
    # PCR present but no two-sided open interest -> no coverage.
    panel = pd.DataFrame(
        [
            _row("AAA", call_oi=0.0, put_oi=0.0, pcr=2.0),
            _row("BBB", pcr=2.0),
        ]
    )
    result = screen_options_criterion(
        "high_pcr_reversal",
        market="us",
        limit=10,
        as_of=date(2026, 7, 6),
        panel=panel,
    )
    assert result.frame.empty
    assert "thin" in result.message


def test_new_criteria_are_registered_and_callable():
    for name in ("bearish_oi_buildup", "high_pcr_reversal"):
        assert name in OPTIONS_CRITERIA
        assert name in SCREEN_ALIASES
        assert callable(SCREEN_ALIASES[name])
        result = screen_options_criterion(
            name, market="us", limit=10, as_of=date(2026, 7, 6), panel=_pcr_panel()
        )
        assert isinstance(result.message, str)
