from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from screener.options.lot_history import historical_lot_sizes
from screener.options.nse_bhavcopy import normalize_bhavcopy_options


def _write_history(path: Path) -> None:
    path.write_text(
        "symbol,effective_from,lot_size\n"
        "RELIANCE,2020-01-01,505\n"
        "RELIANCE,2023-06-01,250\n"
        "TCS,2022-01-01,150\n"
    )


def test_point_in_time_selection_honors_effective_from(tmp_path: Path) -> None:
    path = tmp_path / "lot_sizes_history.csv"
    _write_history(path)
    # Before the RELIANCE change -> old lot; after -> new lot.
    early = historical_lot_sizes(date(2023, 1, 1), path=path)
    assert early == {"RELIANCE": 505.0, "TCS": 150.0}
    late = historical_lot_sizes(date(2023, 12, 31), path=path)
    assert late["RELIANCE"] == 250.0
    # A date before any TCS record omits TCS.
    assert historical_lot_sizes(date(2021, 1, 1), path=path) == {"RELIANCE": 505.0}


def test_absent_file_returns_empty(tmp_path: Path) -> None:
    assert historical_lot_sizes(date(2023, 1, 1), path=tmp_path / "missing.csv") == {}


def test_malformed_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n")
    assert historical_lot_sizes(date(2023, 1, 1), path=path) == {}


def test_embedded_lot_preferred_over_mapping() -> None:
    frame = pd.DataFrame(
        {
            "TradDt": ["2026-07-08"],
            "FinInstrmTp": ["STO"],
            "TckrSymb": ["RELIANCE"],
            "XpryDt": ["2026-07-28"],
            "StrkPric": [1250.0],
            "OptnTp": ["CE"],
            "FinInstrmNm": ["RELIANCE26JUL1250CE"],
            "ClsPric": [45.0],
            "UndrlygPric": [1275.9],
            "SttlmPric": [45.0],
            "OpnIntrst": [1000],
            "ChngInOpnIntrst": [10],
            "TtlTradgVol": [5],
            "NewBrdLotQty": [500],
        }
    )
    chains = normalize_bhavcopy_options(
        frame, as_of=date(2026, 7, 8), lot_sizes={"RELIANCE": 999.0}
    )
    # Embedded NewBrdLotQty wins over the mapping fallback.
    assert chains["RELIANCE"].contracts[0].lot_size == 500.0

    # When embedded lot is absent, the mapping fills in.
    frame.loc[0, "NewBrdLotQty"] = float("nan")
    chains2 = normalize_bhavcopy_options(
        frame, as_of=date(2026, 7, 8), lot_sizes={"RELIANCE": 999.0}
    )
    assert chains2["RELIANCE"].contracts[0].lot_size == 999.0
