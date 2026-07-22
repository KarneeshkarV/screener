from __future__ import annotations

import io
import sys
import types
import zipfile
from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.operator import cli, fetch, output, process, screen, universe


def test_latest_trading_day_uses_actual_bhavcopy_date(monkeypatch):
    requested = date(2024, 1, 7)
    actual = date(2024, 1, 5)

    monkeypatch.setattr(fetch, "is_trading_day", lambda d: d.weekday() < 6)
    monkeypatch.setattr(
        fetch,
        "_read_cash_bhavcopy_raw",
        lambda d: pd.DataFrame({"DATE1": ["05-Jan-2024"]}),
    )

    assert fetch.latest_trading_day(requested) == actual


def test_latest_trading_day_skips_bad_candidates(monkeypatch):
    requested = date(2024, 1, 8)
    calls: list[date] = []

    def fake_read(d: date) -> pd.DataFrame:
        calls.append(d)
        if len(calls) == 1:
            raise FileNotFoundError("missing")
        return pd.DataFrame({"DATE1": ["bad"]})

    monkeypatch.setattr(fetch, "is_trading_day", lambda d: True)
    monkeypatch.setattr(fetch, "_read_cash_bhavcopy_raw", fake_read)

    with pytest.raises(RuntimeError, match="no NSE cash bhavcopy"):
        fetch.latest_trading_day(requested, lookback=1)


def test_latest_trading_day_skips_empty_bhavcopy(monkeypatch):
    calls = []

    def fake_read(d: date) -> pd.DataFrame:
        calls.append(d)
        return (
            pd.DataFrame()
            if len(calls) == 1
            else pd.DataFrame({"DATE1": ["05-Jan-2024"]})
        )

    monkeypatch.setattr(fetch, "is_trading_day", lambda d: True)
    monkeypatch.setattr(fetch, "_read_cash_bhavcopy_raw", fake_read)

    assert fetch.latest_trading_day(date(2024, 1, 6), lookback=1) == date(2024, 1, 5)


def test_parse_bhavcopy_date_handles_missing_and_bad_values():
    assert fetch._parse_bhavcopy_date(pd.DataFrame()) is None
    assert fetch._parse_bhavcopy_date(pd.DataFrame({"DATE1": ["not-a-date"]})) is None
    assert fetch._parse_bhavcopy_date(pd.DataFrame({"DATE1": ["05-Jan-2024"]})) == date(
        2024, 1, 5
    )


def test_fetch_cash_bhavcopy_reads_cached_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    d = date(2024, 1, 5)
    path = fetch._cash_cache_path(d)
    path.write_text(
        " SYMBOL , SERIES ,PREV_CLOSE,CLOSE_PRICE,AVG_PRICE,TTL_TRD_QNTY,DELIV_QTY,DELIV_PER,DATE1\n"
        " AAA , EQ ,10,11,10.5,1000,600,60,05-Jan-2024\n"
        " BBB , BE ,20,21,20.5,2000,900,45,05-Jan-2024\n"
    )

    out = fetch.fetch_cash_bhavcopy(d)

    assert out["SYMBOL"].tolist() == ["AAA"]
    assert out.loc[0, "CLOSE_PRICE"] == 11
    assert out.loc[0, "DELIV_PER"] == 60


def test_read_cash_bhavcopy_downloads_when_cache_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    d = date(2024, 1, 5)

    class FakeArchives:
        def full_bhavcopy_save(self, requested: date, parent: str) -> None:
            assert requested == d
            path = fetch._cash_cache_path(requested)
            assert str(path.parent) == parent
            path.write_text("SYMBOL,SERIES,DATE1\nAAA,EQ,05-Jan-2024\n")

    monkeypatch.setitem(
        sys.modules,
        "jugaad_data.nse",
        types.SimpleNamespace(NSEArchives=FakeArchives),
    )
    monkeypatch.setattr(
        fetch, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    out = fetch._read_cash_bhavcopy_raw(d)

    assert out["SYMBOL"].tolist() == ["AAA"]


def test_read_cash_bhavcopy_raises_when_download_does_not_create_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    class FakeArchives:
        def full_bhavcopy_save(self, requested: date, parent: str) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "jugaad_data.nse",
        types.SimpleNamespace(NSEArchives=FakeArchives),
    )
    monkeypatch.setattr(
        fetch, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    with pytest.raises(FileNotFoundError):
        fetch._read_cash_bhavcopy_raw(date(2024, 1, 5))


def test_fetch_fo_bhavcopy_reads_cache_and_collapses_near_month(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    d = date(2024, 1, 5)
    path = fetch._fo_cache_path(d)
    path.write_text(
        "FinInstrmTp,TckrSymb,XpryDt,OpnIntrst\n"
        "STF,AAA,2024-01-25,100\n"
        "STF,AAA,2024-02-29,200\n"
        "STF,AAA,2024-03-28,300\n"
        "STF,BBB,2024-01-25,50\n"
        "OPTIDX,NIFTY,2024-01-25,999\n"
    )

    fo = fetch.fetch_fo_bhavcopy(d)
    collapsed = fetch.near_month_oi(fo)

    assert fo["SYMBOL"].tolist() == ["AAA", "AAA", "AAA", "BBB"]
    aaa = collapsed[collapsed["SYMBOL"] == "AAA"].iloc[0]
    bbb = collapsed[collapsed["SYMBOL"] == "BBB"].iloc[0]
    assert aaa["Current_OI"] == 100
    assert aaa["Next_OI"] == 200
    assert aaa["Cumulative_OI"] == 300
    assert bbb["Current_OI"] == 50
    assert pd.isna(bbb["Next_OI"])
    assert bbb["Cumulative_OI"] == 50


def test_fetch_fo_bhavcopy_downloads_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)
    d = date(2024, 8, 5)  # post-UDiff cutover -> UDiff archive URL
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr(
            "fo.csv",
            "FinInstrmTp,TckrSymb,XpryDt,OpnIntrst\nSTF,AAA,2024-01-25,100\n",
        )

    class Response:
        status_code = 200
        content = zip_buffer.getvalue()
        headers = {"content-type": "application/zip"}

    class FakeSession:
        def get(self, url: str, timeout: int):
            assert "20240805" in url
            assert timeout == 10
            return Response()

    class FakeArchives:
        def __init__(self) -> None:
            self.s = FakeSession()

    monkeypatch.setitem(
        sys.modules,
        "jugaad_data.nse",
        types.SimpleNamespace(NSEArchives=FakeArchives),
    )
    monkeypatch.setattr(
        fetch, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    out = fetch.fetch_fo_bhavcopy(d)

    assert out["SYMBOL"].tolist() == ["AAA"]


def test_fetch_fo_bhavcopy_reports_bad_response(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    class Response:
        status_code = 404
        content = b"nope"
        headers = {"content-type": "text/plain"}

    class FakeSession:
        def get(self, url: str, timeout: int):
            return Response()

    class FakeArchives:
        def __init__(self) -> None:
            self.s = FakeSession()

    monkeypatch.setitem(
        sys.modules,
        "jugaad_data.nse",
        types.SimpleNamespace(NSEArchives=FakeArchives),
    )
    monkeypatch.setattr(
        fetch, "call_with_resilience", lambda _d, _o, fn, fallback=None: fn()
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch.fetch_fo_bhavcopy(date(2024, 1, 5))


def test_fetch_fo_bhavcopy_reports_none_response(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_ROOT", tmp_path)

    class FakeArchives:
        def __init__(self) -> None:
            self.s = object()

    monkeypatch.setitem(
        sys.modules,
        "jugaad_data.nse",
        types.SimpleNamespace(NSEArchives=FakeArchives),
    )
    monkeypatch.setattr(fetch, "call_with_resilience", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="NSE unavailable"):
        fetch.fetch_fo_bhavcopy(date(2024, 1, 5))


def test_fifty_two_week_hl_uses_deepest_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "INDIA_OHLCV_CACHE", tmp_path)
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    short = pd.DataFrame({"date": dates[-50:], "high": 1.0, "low": 1.0, "close": 1.0})
    deep = pd.DataFrame(
        {
            "date": dates,
            "high": range(260),
            "low": range(260, 0, -1),
            "close": range(260),
        }
    )
    short.to_parquet(tmp_path / "AAA__2023-12-01__2024-01-01.parquet")
    deep.to_parquet(tmp_path / "AAA__2023-01-01__2024-01-01.parquet")

    out = fetch.fifty_two_week_hl(["AAA", "MISSING"], date(2023, 9, 17))

    aaa = out[out["SYMBOL"] == "AAA"].iloc[0]
    missing = out[out["SYMBOL"] == "MISSING"].iloc[0]
    assert aaa["_52W_High"] == 259
    assert aaa["_52W_Low"] == 1
    assert pd.isna(missing["_52W_High"])


def test_fifty_two_week_hl_handles_bad_and_short_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "INDIA_OHLCV_CACHE", tmp_path)
    bad_path = tmp_path / "BAD__2023-01-01__2024-01-01.parquet"
    bad_path.write_text("not parquet")
    short = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
        }
    )
    short_path = tmp_path / "SHORT__2024-01-01__2024-01-10.parquet"
    short.to_parquet(short_path)
    real_read_parquet = pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        if path == bad_path:
            raise OSError("corrupt")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    out = fetch.fifty_two_week_hl(["BAD", "SHORT"], date(2024, 1, 10))

    assert out["SYMBOL"].tolist() == ["BAD", "SHORT"]
    assert out["_52W_High"].isna().all()


def test_resolve_parquet_returns_none_when_cache_root_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "INDIA_OHLCV_CACHE", tmp_path / "missing")

    assert fetch._resolve_parquet("AAA") is None


def test_operator_label_classifies_all_action_buckets():
    base = {
        "_is_fno": True,
        "%_Change_Delivery": 150,
        "Dist_From_52W_High": 10,
    }
    df = pd.DataFrame(
        [
            base | {"%_Change_Price": 1, "%_Change_OI": 1},
            base | {"%_Change_Price": 1, "%_Change_OI": -1},
            base | {"%_Change_Price": -1, "%_Change_OI": 1},
            base | {"%_Change_Price": -1, "%_Change_OI": -1},
            base | {"%_Change_Price": 0, "%_Change_OI": 1},
            base | {"_is_fno": False, "%_Change_Price": 1, "%_Change_OI": 1},
            base | {"%_Change_Price": 1, "%_Change_OI": 1, "%_Change_Delivery": 99},
            base | {"%_Change_Price": None, "%_Change_OI": 1},
        ]
    )

    out = screen.label(df)

    assert out["Operator_Action"].tolist() == [
        "Long Build-up",
        "Short Covering",
        "Short Build-up",
        "Long Unwinding",
        None,
        None,
        None,
        None,
    ]
    assert out["High_Momentum_Watch"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_operator_write_csv_sorts_and_filters_actions(tmp_path):
    df = pd.DataFrame(
        [
            _operator_row("BBB", "Short Covering", True, 120),
            _operator_row("AAA", "Long Build-up", True, 130),
            _operator_row("CCC", None, False, 999),
        ]
    )

    path = output.write_csv(
        df,
        date(2024, 1, 5),
        out_path=tmp_path / "operator.csv",
        only_actions=True,
    )

    written = pd.read_csv(path)
    assert written["SYMBOL"].tolist() == ["AAA", "BBB"]
    assert list(written.columns) == output.OUTPUT_COLUMNS


def test_process_build_dataset_computes_derived_columns(monkeypatch):
    as_of = date(2024, 1, 5)
    trailing = [date(2024, 1, day) for day in [4, 3, 2, 1, 1, 4]]
    latest_calls: list[date] = []

    def fake_latest(d: date) -> date:
        latest_calls.append(d)
        if len(latest_calls) == 1:
            return as_of
        return trailing.pop(0)

    monkeypatch.setattr(process, "latest_trading_day", fake_latest)
    monkeypatch.setattr(
        process, "combined_universe", lambda d, mode: (["AAA", "BBB"], {"AAA"})
    )
    monkeypatch.setattr(
        process,
        "fetch_cash_bhavcopy",
        lambda d: pd.DataFrame(
            {
                "SYMBOL": ["AAA", "BBB"],
                "PREV_CLOSE": [100.0, 50.0],
                "CLOSE_PRICE": [110.0, 45.0],
                "AVG_PRICE": [108.0, 46.0],
                "TTL_TRD_QNTY": [1000.0, 2000.0],
                "DELIV_QTY": [200.0 if d == as_of else 100.0, 80.0],
                "DELIV_PER": [60.0, 40.0],
            }
        ),
    )

    def fake_fo(d: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SYMBOL": ["AAA"],
                "EXPIRY": [pd.Timestamp("2024-01-25")],
                "OI": [150.0 if d == as_of else 100.0],
            }
        )

    monkeypatch.setattr(process, "fetch_fo_bhavcopy", fake_fo)
    monkeypatch.setattr(process, "load_bhavcopy_chains", lambda d: {})
    monkeypatch.setattr(
        process,
        "fifty_two_week_hl",
        lambda symbols, d: pd.DataFrame(
            {
                "SYMBOL": list(symbols),
                "_52W_High": [120.0, 60.0],
                "_52W_Low": [80.0, 40.0],
            }
        ),
    )

    out, actual = process.build_dataset(as_of, universe_mode="fo")

    aaa = out[out["SYMBOL"] == "AAA"].iloc[0]
    assert actual == as_of
    assert aaa["%_Change_Price"] == pytest.approx(10.0)
    assert aaa["%_Change_OI"] == pytest.approx(50.0)
    assert aaa["%_Change_Delivery"] == pytest.approx(200.0)
    assert aaa["Dist_From_52W_High"] == pytest.approx((120 - 110) / 120 * 100)
    assert bool(aaa["_is_fno"]) is True


def test_universe_modes_and_fallback(monkeypatch):
    monkeypatch.setattr(universe, "fno_symbols", lambda d: ["BBB", "AAA"])
    monkeypatch.setattr(universe, "cash_top_500", lambda: ["CCC", "AAA"])

    assert universe.combined_universe(date(2024, 1, 5), mode="fo") == (
        ["BBB", "AAA"],
        {"AAA", "BBB"},
    )
    assert universe.combined_universe(date(2024, 1, 5), mode="fo+cash") == (
        ["AAA", "BBB", "CCC"],
        {"AAA", "BBB"},
    )

    monkeypatch.setattr(
        universe,
        "cash_top_500",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert universe.combined_universe(date(2024, 1, 5), mode="fo+cash") == (
        ["BBB", "AAA"],
        {"AAA", "BBB"},
    )
    with pytest.raises(ValueError, match="unknown universe mode"):
        universe.combined_universe(date(2024, 1, 5), mode="bad")


def test_universe_leaf_loaders(monkeypatch):
    monkeypatch.setattr(
        universe,
        "fetch_fo_bhavcopy",
        lambda d: pd.DataFrame({"SYMBOL": ["BBB", "AAA", "AAA"]}),
    )
    monkeypatch.setitem(
        sys.modules,
        "screener.research.pine_runner",
        types.SimpleNamespace(load_universe=lambda market: [f"{market}:AAA"]),
    )

    assert universe.fno_symbols(date(2024, 1, 5)) == ["AAA", "BBB"]
    assert universe.cash_top_500() == ["india:AAA"]


def test_operator_cli_writes_summary(monkeypatch, tmp_path):
    df = screen.label(
        pd.DataFrame(
            [
                _operator_row("AAA", None, False, 200)
                | {
                    "_is_fno": True,
                    "%_Change_Price": 1,
                    "%_Change_OI": 1,
                    "%_Change_Delivery": 150,
                    "Dist_From_52W_High": 10,
                }
            ]
        )
    )
    out_path = tmp_path / "out.csv"

    monkeypatch.setattr(
        cli, "build_dataset", lambda as_of, universe_mode: (df, date(2024, 1, 5))
    )

    result = CliRunner().invoke(
        cli.operator_scan,
        ["--date", "2024-01-06", "--universe", "fo", "--output", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Operator scan: trading day 2024-01-05" in result.output
    assert "Long Build-up" in result.output
    assert out_path.exists()


def _operator_row(
    symbol: str,
    action: str | None,
    high_momentum_watch: bool,
    delivery_change: float,
) -> dict:
    return {
        "SYMBOL": symbol,
        "Operator_Action": action,
        "High_Momentum_Watch": high_momentum_watch,
        "Options_OI_Confirmation": None,
        "Options_Confirms_Futures": False,
        "ATM_Call_Writing_OI": float("nan"),
        "ATM_Put_Writing_OI": float("nan"),
        "CLOSE_PRICE": 100.0,
        "AVG_PRICE": 99.0,
        "%_Change_Price": 1.0,
        "%_Change_OI": 2.0,
        "%_Change_Delivery": delivery_change,
        "Dist_From_52W_High": 10.0,
        "_52W_High": 110.0,
        "_52W_Low": 80.0,
        "DELIV_QTY": 1000.0,
        "DELIV_PER": 55.0,
        "5_Day_Avg_Delivery": 500.0,
        "Current_OI": 100.0,
        "Next_OI": 200.0,
        "Cumulative_OI": 300.0,
        "PREV_CLOSE": 99.0,
    }
