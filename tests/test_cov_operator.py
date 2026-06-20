"""Offline line-coverage tests for the ``screener.operator`` package.

All NSE / network / disk-archive interactions are monkeypatched so these
tests are fully deterministic and never touch the network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from screener.cli import cli
from screener.operator import cli as op_cli
from screener.operator import fetch as fetch_mod
from screener.operator import output as output_mod
from screener.operator import process as process_mod
from screener.operator import screen as screen_mod
from screener.operator import universe as universe_mod


# ── fetch.py ───────────────────────────────────────────────────────────


def _cash_raw_df(d: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SYMBOL": ["AAA ", "BBB", "ZZZ"],
            "SERIES": ["EQ", "EQ", "SME"],
            "DATE1": [d.strftime("%d-%b-%Y")] * 3,
            "PREV_CLOSE": ["100", "50", "1"],
            "CLOSE_PRICE": ["110", "48", "2"],
            "AVG_PRICE": ["105", "49", "1.5"],
            "TTL_TRD_QNTY": ["1000", "2000", "10"],
            "DELIV_QTY": ["500", "1000", "5"],
            "DELIV_PER": ["50", "50", "50"],
        }
    )


def test_latest_trading_day_drift_and_skip(monkeypatch):
    requested = date(2024, 6, 16)  # a Sunday
    actual_day = date(2024, 6, 14)

    calls = {"reads": []}

    def fake_is_trading_day(d):
        # Skip the Sunday so the first iteration `continue`s.
        return d != requested

    def fake_read_raw(d):
        calls["reads"].append(d)
        # First non-skipped candidate returns the drifted file.
        return _cash_raw_df(actual_day)

    monkeypatch.setattr(fetch_mod, "is_trading_day", fake_is_trading_day)
    monkeypatch.setattr(fetch_mod, "_read_cash_bhavcopy_raw", fake_read_raw)

    got = fetch_mod.latest_trading_day(requested)
    assert got == actual_day
    # Sunday skipped → first read is the Saturday candidate.
    assert requested not in calls["reads"]


def test_latest_trading_day_read_error_then_empty_then_no_date(monkeypatch):
    monkeypatch.setattr(fetch_mod, "is_trading_day", lambda d: True)

    state = {"n": 0}

    def fake_read_raw(d):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError("boom")  # exception branch -> continue
        if state["n"] == 2:
            return pd.DataFrame()  # empty -> continue
        if state["n"] == 3:
            return pd.DataFrame({"SYMBOL": ["X"]})  # no DATE1 -> parse None -> continue
        # 4th: a parseable, non-drifted file
        return _cash_raw_df(d)

    monkeypatch.setattr(fetch_mod, "_read_cash_bhavcopy_raw", fake_read_raw)
    got = fetch_mod.latest_trading_day(date(2024, 6, 14), lookback=10)
    assert isinstance(got, date)


def test_latest_trading_day_exhausts_lookback(monkeypatch):
    monkeypatch.setattr(fetch_mod, "is_trading_day", lambda d: False)
    with pytest.raises(RuntimeError, match="no NSE cash bhavcopy"):
        fetch_mod.latest_trading_day(date(2024, 6, 14), lookback=2)


def test_parse_bhavcopy_date_branches():
    assert fetch_mod._parse_bhavcopy_date(pd.DataFrame()) is None
    assert fetch_mod._parse_bhavcopy_date(pd.DataFrame({"X": [1]})) is None
    assert (
        fetch_mod._parse_bhavcopy_date(pd.DataFrame({"DATE1": ["not-a-date"]})) is None
    )
    parsed = fetch_mod._parse_bhavcopy_date(pd.DataFrame({"DATE1": ["14-Jun-2024"]}))
    assert parsed == date(2024, 6, 14)


def test_read_cash_bhavcopy_raw_download_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "CACHE_ROOT", tmp_path)
    d = date(2024, 6, 14)

    class FakeNSE:
        def full_bhavcopy_save(self, day, parent):
            _cash_raw_df(day).to_csv(fetch_mod._cash_cache_path(day), index=False)

    import jugaad_data.nse as _jn

    monkeypatch.setattr(_jn, "NSEArchives", FakeNSE, raising=False)

    captured = {}

    def fake_resilience(provider, desc, fn, fallback=None):
        captured["called"] = True
        return fn()

    monkeypatch.setattr(fetch_mod, "call_with_resilience", fake_resilience)

    df = fetch_mod._read_cash_bhavcopy_raw(d)
    assert captured["called"]
    # object columns stripped
    assert (df["SYMBOL"] == "AAA").any()
    # Second call should hit the cache (no download).
    captured["called"] = False
    df2 = fetch_mod._read_cash_bhavcopy_raw(d)
    assert not captured["called"]
    assert len(df2) == 3


def test_read_cash_bhavcopy_raw_missing_after_save(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "CACHE_ROOT", tmp_path)

    import jugaad_data.nse as _jn

    monkeypatch.setattr(_jn, "NSEArchives", lambda: object(), raising=False)
    # resilience does nothing → file never created → FileNotFoundError
    monkeypatch.setattr(fetch_mod, "call_with_resilience", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        fetch_mod._read_cash_bhavcopy_raw(date(2024, 6, 14))


def test_fetch_cash_bhavcopy_filters_and_coerces(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_read_cash_bhavcopy_raw", lambda d: _cash_raw_df(d))
    out = fetch_mod.fetch_cash_bhavcopy(date(2024, 6, 14))
    # SME row dropped; whitespace-strip happens in the raw reader (stubbed here).
    assert list(out["SYMBOL"]) == ["AAA ", "BBB"]
    assert out["CLOSE_PRICE"].dtype.kind in ("f", "i")


def _fo_zip_bytes() -> bytes:
    import io
    import zipfile

    df = pd.DataFrame(
        {
            "FinInstrmTp": ["STF", "STF", "IDF"],
            "TckrSymb": ["AAA", "AAA", "NIFTY"],
            "XpryDt": ["2024-06-27", "2024-07-25", "2024-06-27"],
            "OpnIntrst": ["1000", "500", "9"],
        }
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.csv", df.to_csv(index=False))
    return buf.getvalue()


class _FakeResp:
    def __init__(self, status_code=200, content=b"PK\x03\x04", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "application/zip"}


def test_fetch_fo_bhavcopy_download(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "CACHE_ROOT", tmp_path)
    d = date(2024, 6, 14)

    class FakeNSE:
        def __init__(self):
            self.s = self

        def get(self, url, timeout=10):
            return _FakeResp(content=_fo_zip_bytes())

    import jugaad_data.nse as _jn

    monkeypatch.setattr(_jn, "NSEArchives", FakeNSE, raising=False)
    monkeypatch.setattr(
        fetch_mod, "call_with_resilience", lambda p, desc, fn, fallback=None: fn()
    )

    out = fetch_mod.fetch_fo_bhavcopy(d)
    assert set(out.columns) == {"SYMBOL", "EXPIRY", "OI"}
    assert list(out["SYMBOL"]) == ["AAA", "AAA"]  # IDF dropped

    # Cached read on second call (file exists already).
    out2 = fetch_mod.fetch_fo_bhavcopy(d)
    assert len(out2) == 2


def test_fetch_fo_bhavcopy_resilience_none(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "CACHE_ROOT", tmp_path)
    import jugaad_data.nse as _jn

    monkeypatch.setattr(_jn, "NSEArchives", lambda: object(), raising=False)
    monkeypatch.setattr(fetch_mod, "call_with_resilience", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="NSE unavailable"):
        fetch_mod.fetch_fo_bhavcopy(date(2024, 6, 14))


def test_fetch_fo_bhavcopy_bad_status(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "CACHE_ROOT", tmp_path)
    import jugaad_data.nse as _jn

    monkeypatch.setattr(_jn, "NSEArchives", lambda: object(), raising=False)
    monkeypatch.setattr(
        fetch_mod,
        "call_with_resilience",
        lambda *a, **k: _FakeResp(status_code=404, content=b"no"),
    )
    with pytest.raises(RuntimeError, match="HTTP 404"):
        fetch_mod.fetch_fo_bhavcopy(date(2024, 6, 14))


def test_near_month_oi():
    fo = pd.DataFrame(
        {
            "SYMBOL": ["AAA", "AAA", "BBB"],
            "EXPIRY": pd.to_datetime(["2024-07-25", "2024-06-27", "2024-06-27"]),
            "OI": [500.0, 1000.0, 700.0],
        }
    )
    out = fetch_mod.near_month_oi(fo).set_index("SYMBOL")
    # AAA: sorted by expiry → current=1000 (June), next=500 (July)
    assert out.loc["AAA", "Current_OI"] == 1000.0
    assert out.loc["AAA", "Next_OI"] == 500.0
    assert out.loc["AAA", "Cumulative_OI"] == 1500.0
    # BBB single expiry → Next NaN, cumulative == current
    assert pd.isna(out.loc["BBB", "Next_OI"])
    assert out.loc["BBB", "Cumulative_OI"] == 700.0


def test_resolve_parquet_none_when_cache_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "INDIA_OHLCV_CACHE", tmp_path / "nope")
    assert fetch_mod._resolve_parquet("AAA") is None


def test_resolve_parquet_no_match(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "INDIA_OHLCV_CACHE", tmp_path)
    assert fetch_mod._resolve_parquet("AAA") is None


def test_resolve_parquet_picks_earliest(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "INDIA_OHLCV_CACHE", tmp_path)
    (tmp_path / "AAA__2020-01-01__2024-01-01.parquet").write_text("x")
    (tmp_path / "AAA__2018-01-01__2024-01-01.parquet").write_text("x")
    chosen = fetch_mod._resolve_parquet("AAA")
    assert chosen.name.startswith("AAA__2018")


def _write_parquet(path: Path, n: int, as_of: date) -> None:
    idx = pd.bdate_range(end=pd.Timestamp(as_of), periods=n)
    df = pd.DataFrame(
        {
            "date": idx,
            "high": range(100, 100 + n),
            "low": range(50, 50 + n),
            "close": range(75, 75 + n),
        }
    )
    df.to_parquet(path)


def test_fifty_two_week_hl_all_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_mod, "INDIA_OHLCV_CACHE", tmp_path)
    as_of = date(2024, 6, 14)

    # GOOD: >= 200 trading days
    _write_parquet(tmp_path / "GOOD__2020-01-01__x.parquet", 260, as_of)
    # SHORT: < 200 trading days
    _write_parquet(tmp_path / "SHORT__2023-01-01__x.parquet", 20, as_of)
    # BAD: read raises OSError (exercise the except branch deterministically).
    _write_parquet(tmp_path / "BAD__2020-01-01__x.parquet", 260, as_of)

    real_read = pd.read_parquet

    def fake_read_parquet(path, **kwargs):
        if "BAD__" in str(path):
            raise OSError("corrupt")
        return real_read(path, **kwargs)

    monkeypatch.setattr(fetch_mod.pd, "read_parquet", fake_read_parquet)

    out = fetch_mod.fifty_two_week_hl(
        ["GOOD", "SHORT", "BAD", "MISSING"], as_of
    ).set_index("SYMBOL")
    assert pd.notna(out.loc["GOOD", "_52W_High"])
    assert pd.isna(out.loc["SHORT", "_52W_High"])
    assert pd.isna(out.loc["BAD", "_52W_High"])
    assert pd.isna(out.loc["MISSING", "_52W_High"])


# ── universe.py ────────────────────────────────────────────────────────


def test_fno_symbols(monkeypatch):
    monkeypatch.setattr(
        universe_mod,
        "fetch_fo_bhavcopy",
        lambda d: pd.DataFrame({"SYMBOL": ["BBB", "AAA", "AAA"]}),
    )
    assert universe_mod.fno_symbols(date(2024, 6, 14)) == ["AAA", "BBB"]


def test_cash_top_500(monkeypatch):
    import run_pinescript_strategies as rps

    monkeypatch.setattr(rps, "load_universe", lambda mkt: ["AAA", "CCC"])
    assert universe_mod.cash_top_500() == ["AAA", "CCC"]


def test_combined_universe_fo_only(monkeypatch):
    monkeypatch.setattr(universe_mod, "fno_symbols", lambda d: ["AAA", "BBB"])
    syms, fset = universe_mod.combined_universe(date(2024, 6, 14), mode="fo")
    assert syms == ["AAA", "BBB"]
    assert fset == {"AAA", "BBB"}


def test_combined_universe_fo_cash(monkeypatch):
    monkeypatch.setattr(universe_mod, "fno_symbols", lambda d: ["AAA", "BBB"])
    monkeypatch.setattr(universe_mod, "cash_top_500", lambda: ["BBB", "CCC"])
    syms, fset = universe_mod.combined_universe(date(2024, 6, 14))
    assert syms == ["AAA", "BBB", "CCC"]
    assert fset == {"AAA", "BBB"}


def test_combined_universe_cash_failure_falls_back(monkeypatch):
    monkeypatch.setattr(universe_mod, "fno_symbols", lambda d: ["AAA"])

    def boom():
        raise ConnectionError("down")

    monkeypatch.setattr(universe_mod, "cash_top_500", boom)
    syms, fset = universe_mod.combined_universe(date(2024, 6, 14))
    assert syms == ["AAA"]
    assert fset == {"AAA"}


def test_combined_universe_unknown_mode(monkeypatch):
    monkeypatch.setattr(universe_mod, "fno_symbols", lambda d: ["AAA"])
    with pytest.raises(ValueError, match="unknown universe mode"):
        universe_mod.combined_universe(date(2024, 6, 14), mode="bogus")


# ── process.py ─────────────────────────────────────────────────────────


def test_trailing_trading_days(monkeypatch):
    seq = [date(2024, 6, 13), date(2024, 6, 12), date(2024, 6, 11)]
    it = iter(seq)
    monkeypatch.setattr(process_mod, "latest_trading_day", lambda d: next(it))
    days = process_mod._trailing_trading_days(date(2024, 6, 14), 3)
    assert days == seq


def test_five_day_avg_delivery(monkeypatch):
    days = [date(2024, 6, 13), date(2024, 6, 12)]
    monkeypatch.setattr(process_mod, "_trailing_trading_days", lambda d, n: days)

    def fake_cash(td):
        return pd.DataFrame({"SYMBOL": ["AAA", "BBB"], "DELIV_QTY": [100.0, 200.0]})

    monkeypatch.setattr(process_mod, "fetch_cash_bhavcopy", fake_cash)
    out = process_mod._five_day_avg_delivery(date(2024, 6, 14))
    assert "5_Day_Avg_Delivery" in out.columns
    assert out.set_index("SYMBOL").loc["AAA", "5_Day_Avg_Delivery"] == 100.0


def _patch_build_dataset(monkeypatch):
    today = date(2024, 6, 14)
    prev = date(2024, 6, 13)

    monkeypatch.setattr(process_mod, "latest_trading_day", lambda d: today)
    monkeypatch.setattr(
        process_mod,
        "combined_universe",
        lambda d, mode: (["AAA", "BBB"], {"AAA"}),
    )
    monkeypatch.setattr(
        process_mod,
        "fetch_cash_bhavcopy",
        lambda d: pd.DataFrame(
            {
                "SYMBOL": ["AAA", "BBB"],
                "PREV_CLOSE": [100.0, 50.0],
                "CLOSE_PRICE": [110.0, 48.0],
                "AVG_PRICE": [105.0, 49.0],
                "TTL_TRD_QNTY": [1000.0, 2000.0],
                "DELIV_QTY": [600.0, 1000.0],
                "DELIV_PER": [60.0, 50.0],
            }
        ),
    )
    monkeypatch.setattr(
        process_mod,
        "_five_day_avg_delivery",
        lambda d: pd.DataFrame(
            {"SYMBOL": ["AAA", "BBB"], "5_Day_Avg_Delivery": [300.0, 1000.0]}
        ),
    )

    def fake_fo(d):
        oi = 1000.0 if d == today else 500.0
        return pd.DataFrame(
            {
                "SYMBOL": ["AAA"],
                "EXPIRY": pd.to_datetime(["2024-06-27"]),
                "OI": [oi],
            }
        )

    monkeypatch.setattr(process_mod, "fetch_fo_bhavcopy", fake_fo)
    monkeypatch.setattr(process_mod, "near_month_oi", fetch_mod.near_month_oi)
    monkeypatch.setattr(process_mod, "_trailing_trading_days", lambda d, n: [prev])
    monkeypatch.setattr(
        process_mod,
        "fifty_two_week_hl",
        lambda syms, d: pd.DataFrame(
            {
                "SYMBOL": ["AAA", "BBB"],
                "_52W_High": [120.0, 60.0],
                "_52W_Low": [80.0, 40.0],
            }
        ),
    )
    return today


def test_build_dataset(monkeypatch):
    today = _patch_build_dataset(monkeypatch)
    df, actual = process_mod.build_dataset(date(2024, 6, 14))
    assert actual == today
    assert set(df["SYMBOL"]) == {"AAA", "BBB"}
    assert "%_Change_Price" in df.columns
    assert "%_Change_OI" in df.columns
    assert "Dist_From_52W_High" in df.columns
    assert df.set_index("SYMBOL").loc["AAA", "_is_fno"]
    assert not df.set_index("SYMBOL").loc["BBB", "_is_fno"]


def test_build_dataset_default_today(monkeypatch):
    _patch_build_dataset(monkeypatch)
    # as_of=None path -> uses date.today() then latest_trading_day (stubbed)
    df, actual = process_mod.build_dataset(None)
    assert actual == date(2024, 6, 14)


# ── screen.py ──────────────────────────────────────────────────────────


def _row(**kw):
    base = {
        "_is_fno": True,
        "%_Change_Price": 1.0,
        "%_Change_OI": 1.0,
        "%_Change_Delivery": 150.0,
    }
    base.update(kw)
    return base


def test_classify_all_branches():
    assert screen_mod._classify(_row(_is_fno=False)) is None
    assert screen_mod._classify(_row(**{"%_Change_Price": float("nan")})) is None
    assert screen_mod._classify(_row(**{"%_Change_Delivery": 50.0})) is None  # gate
    assert screen_mod._classify(_row()) == "Long Build-up"
    assert screen_mod._classify(_row(**{"%_Change_OI": -1.0})) == "Short Covering"
    assert (
        screen_mod._classify(_row(**{"%_Change_Price": -1.0, "%_Change_OI": 1.0}))
        == "Short Build-up"
    )
    assert (
        screen_mod._classify(_row(**{"%_Change_Price": -1.0, "%_Change_OI": -1.0}))
        == "Long Unwinding"
    )
    # flat price -> None
    assert screen_mod._classify(_row(**{"%_Change_Price": 0.0})) is None


def test_label():
    df = pd.DataFrame(
        [
            {
                "_is_fno": True,
                "%_Change_Price": 1.0,
                "%_Change_OI": 1.0,
                "%_Change_Delivery": 150.0,
                "Dist_From_52W_High": 10.0,
            },
            {
                "_is_fno": True,
                "%_Change_Price": 1.0,
                "%_Change_OI": 1.0,
                "%_Change_Delivery": 150.0,
                "Dist_From_52W_High": float("nan"),
            },
        ]
    )
    out = screen_mod.label(df)
    assert out.loc[0, "Operator_Action"] == "Long Build-up"
    assert bool(out.loc[0, "High_Momentum_Watch"]) is True
    # NaN distance treated as not-near-high
    assert bool(out.loc[1, "High_Momentum_Watch"]) is False


# ── output.py ──────────────────────────────────────────────────────────


def _labeled_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SYMBOL": ["AAA", "BBB", "CCC"],
            "Operator_Action": ["Long Build-up", None, "Short Covering"],
            "High_Momentum_Watch": [True, False, False],
            "CLOSE_PRICE": [110.0, 48.0, 30.0],
            "AVG_PRICE": [105.0, 49.0, 29.0],
            "%_Change_Price": [1.0, -2.0, 0.5],
            "%_Change_OI": [5.0, float("nan"), -3.0],
            "%_Change_Delivery": [150.0, 80.0, 120.0],
            "Dist_From_52W_High": [5.0, 40.0, 10.0],
            "_52W_High": [120.0, 60.0, 35.0],
            "_52W_Low": [80.0, 40.0, 25.0],
            "DELIV_QTY": [600.0, 1000.0, 400.0],
            "DELIV_PER": [60.0, 50.0, 40.0],
            "5_Day_Avg_Delivery": [400.0, 1200.0, 350.0],
            "Current_OI": [1000.0, float("nan"), 800.0],
            "Next_OI": [500.0, float("nan"), 200.0],
            "Cumulative_OI": [1500.0, float("nan"), 1000.0],
            "PREV_CLOSE": [100.0, 50.0, 30.0],
        }
    )


def test_write_csv_full(tmp_path):
    out = output_mod.write_csv(
        _labeled_df(), date(2024, 6, 14), out_path=tmp_path / "x.csv"
    )
    df = pd.read_csv(out)
    assert list(df.columns) == output_mod.OUTPUT_COLUMNS
    # HMW row sorts first
    assert df.iloc[0]["SYMBOL"] == "AAA"


def test_write_csv_only_actions_and_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = output_mod.write_csv(_labeled_df(), date(2024, 6, 14), only_actions=True)
    assert out.name == "daily_operator_data_20240614.csv"
    df = pd.read_csv(out)
    # BBB (None action) filtered out
    assert "BBB" not in df["SYMBOL"].tolist()
    assert len(df) == 2


# ── cli.py ─────────────────────────────────────────────────────────────


def _patch_cli(monkeypatch, captured):
    df = _labeled_df()

    def fake_build(as_of, universe_mode="fo+cash"):
        captured["as_of"] = as_of
        captured["mode"] = universe_mode
        return df.copy(), date(2024, 6, 14)

    monkeypatch.setattr(op_cli, "build_dataset", fake_build)
    monkeypatch.setattr(op_cli, "label", lambda d: d)

    def fake_write(d, as_of, out_path=None, only_actions=False):
        captured["only_actions"] = only_actions
        return Path("written.csv")

    monkeypatch.setattr(op_cli, "write_csv", fake_write)


def test_register_adds_command():
    grp = type(cli)(name="root")
    op_cli.register(grp)
    assert "operator-scan" in grp.commands


def test_operator_scan_with_date(monkeypatch):
    captured = {}
    _patch_cli(monkeypatch, captured)
    res = CliRunner().invoke(
        cli,
        ["operator-scan", "--date", "2024-06-14", "--universe", "fo", "-v"],
    )
    assert res.exit_code == 0, res.output
    assert captured["as_of"] == date(2024, 6, 14)
    assert captured["mode"] == "fo"
    assert "Operator scan" in res.output
    assert "High_Momentum_Watch" in res.output
    # value_counts non-empty branch
    assert "Long Build-up" in res.output


def test_operator_scan_defaults_today_only_actions(monkeypatch):
    captured = {}
    _patch_cli(monkeypatch, captured)

    # Empty actions to exercise the `actions.empty` (skip loop) branch.
    df = _labeled_df().copy()
    df["Operator_Action"] = None
    df["High_Momentum_Watch"] = False

    def fake_build(as_of, universe_mode="fo+cash"):
        captured["as_of"] = as_of
        return df.copy(), date(2024, 6, 14)

    monkeypatch.setattr(op_cli, "build_dataset", fake_build)

    res = CliRunner().invoke(cli, ["operator-scan", "--only-actions"])
    assert res.exit_code == 0, res.output
    assert captured["only_actions"] is True
    # Defaults to today when --date omitted.
    assert captured["as_of"] == date.today()
