"""Entity linking, turnover snapshots and price back-adjustment from NSE bhavcopy."""

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import india_pit_bhavcopy
from scripts.india_pit_bhavcopy import (
    ActionTerms,
    BhavcopyPriceFetcher,
    CorporateAction,
    adjust_prices,
    download_bhavcopy,
    entity_sessions,
    link_entities,
    parse_action_subject,
    read_bhavcopy_zip,
    remap_snapshots,
    turnover_snapshots,
)
from scripts.run_india_momentum_ablation import (
    circuit_hits,
    dry_powder,
    fixed_mix,
    prepare_bars,
)
from screener.strategies.spec import PrepareCtx


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Bonus 1:1", ActionTerms(split=2.0)),
        ("Bonus - 3:2", ActionTerms(split=2.5)),
        (
            "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share",
            ActionTerms(split=10.0),
        ),
        (
            "Face Valus Splt (Sub-Divisn) - Frm Rs 5 Per Shr To Re 1 Per Shr",
            ActionTerms(split=5.0),
        ),
        (
            "Interim Dividend - Rs 5 Per Share / Special Dividend - Rs 2.50 Per Share",
            ActionTerms(dividend=7.5),
        ),
        (
            "Bonus 1:1 And Face Value Split Rs.10/- To Rs.5/- Per Share",
            ActionTerms(split=4.0),
        ),
        ("Rights 1:5 @ Premium Rs 190/-", ActionTerms(rights=(1.0, 5.0, 190.0))),
        (
            "Rights 5: 116 At Premium Rs 244/- Per Share",
            ActionTerms(rights=(5.0, 116.0, 244.0)),
        ),
        (
            "Right Issue 1:15 @ Premium Of Rs 333 Per Share",
            ActionTerms(rights=(1.0, 15.0, 333.0)),
        ),
        ("Rights - 2:3@ Par Rs 10 Per Share", ActionTerms(rights=(2.0, 3.0, 0.0))),
        ("Scheme Of Arrangement", ActionTerms(demerger=True)),
        ("Bonus 1:1 On Preference Shares", ActionTerms()),
        ("Interest Payment On Debentures - Rs 50", ActionTerms()),
    ],
)
def test_parse_action_subject(subject, expected):
    assert parse_action_subject(subject) == expected


@pytest.mark.parametrize(
    "subject",
    [
        "Rights 2:7",
        "Dividend Re1/- Per Share + Rights Issue 5:6",
        "Rights - 4:25 Fully Paid Up Shares @ Premium Rs 500/- Per Share"
        " / 2:25 Partly Paid Up Shares @ Premium Rs 605/- Per Share",
    ],
)
def test_rights_without_one_stated_price_have_unknown_premium(subject):
    rights = parse_action_subject(subject).rights
    assert rights is not None and np.isnan(rights[2])


def _zip(name: str, frame: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, frame.to_csv(index=False))
    buffer.seek(0)
    return buffer


class _Archive:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, headers, timeout):
        self.calls += 1
        status, content = self.responses.pop(0)
        response = type("Response", (), {})()
        response.status_code = status
        response.content = content
        response.raise_for_status = lambda: None
        return response


def test_recent_404_is_retried_but_a_settled_one_is_final(tmp_path, monkeypatch):
    monkeypatch.setattr(
        india_pit_bhavcopy,
        "bhavcopy_target",
        lambda day: ("url", tmp_path / f"{day}.zip"),
    )
    today = date.today()
    archive = _Archive((404, b""), (200, _zip("bhav.csv", pd.DataFrame()).getvalue()))
    # Not yet published today: the next run asks again and gets the file.
    assert download_bhavcopy(today, archive) == "holiday"
    assert download_bhavcopy(today, archive) == "downloaded"
    assert download_bhavcopy(today, archive) == "cached"
    assert archive.calls == 2

    old = date(2020, 1, 1)
    (tmp_path / f"{old}.zip.404").touch()  # written long after the session
    assert download_bhavcopy(old, _Archive()) == "cached"


def test_read_bhavcopy_zip_handles_both_formats_and_two_digit_years(tmp_path):
    old = pd.DataFrame(
        {
            "SYMBOL": ["ABC "],
            "SERIES": ["EQ"],
            "OPEN": [1],
            "HIGH": [2],
            "LOW": [0.5],
            "CLOSE": [1.5],
            "TOTTRDQTY": [10],
            "TOTTRDVAL": [15],
            "TIMESTAMP": ["13-Jul-20"],
            "ISIN": ["INE000A01010"],
        }
    )
    udiff = pd.DataFrame(
        {
            "TradDt": ["2024-07-08"],
            "TckrSymb": ["ABC"],
            "SctySrs": ["EQ"],
            "ISIN": ["INE000A01010"],
            "OpnPric": [1],
            "HghPric": [2],
            "LwPric": [0.5],
            "ClsPric": [1.5],
            "TtlTradgVol": [10],
            "TtlTrfVal": [15],
        }
    )
    for name, frame, day in (
        ("old.zip", old, "2020-07-13"),
        ("udiff.zip", udiff, "2024-07-08"),
    ):
        path = tmp_path / name
        path.write_bytes(_zip("bhav.csv", frame).getvalue())
        out = read_bhavcopy_zip(path)
        assert out.loc[0, "date"] == pd.Timestamp(day)
        assert out.loc[0, "symbol"] == "ABC"
        assert out.loc[0, "value"] == 15


def test_link_entities_joins_isin_and_renames_under_latest_symbol():
    listings = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2021-01-01",
                    "2022-01-01",
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-01",
                ]
            ),
            "symbol": ["OLDCO", "MIDCO", "NEWCO", "NEWCO", "OTHER", "REUSED"],
            "isin": ["INE1", "INE1", "INE2", "INE2", "INE3", "INE4"],
        }
    )
    renamed = pd.Timestamp("2022-01-01")
    entity_of = link_entities(
        listings,
        [
            ("MIDCO", "NEWCO", renamed),
            ("GONE", "NEWCO", renamed),
            # REUSED first trades after its rename date: a new, unrelated listing.
            ("REUSED", "OTHER", renamed),
        ],
    )
    assert entity_of == {
        "OLDCO": "NEWCO",
        "MIDCO": "NEWCO",
        "NEWCO": "NEWCO",
        "OTHER": "OTHER",
        "REUSED": "REUSED",
    }
    snapshots = pd.DataFrame(
        {
            "effective_date": ["2020-02-01"] * 3,
            "symbol": ["OLDCO.NS", "NEWCO.NS", "UNSEEN.NS"],
        }
    )
    assert remap_snapshots(snapshots, entity_of)["symbol"].tolist() == [
        "NEWCO.NS",
        "UNSEEN.NS",
    ]


def test_entity_sessions_keeps_busiest_listing_per_day():
    bhav = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 2),
            "symbol": ["A", "A2"],
            "value": [5.0, 9.0],
        }
    )
    out = entity_sessions(bhav, {"A": "A", "A2": "A"})
    assert out["symbol"].tolist() == ["A2"]


def _sessions(
    values: dict[str, list[float]], start: str = "2020-01-01"
) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(next(iter(values.values()))))
    frames = [
        pd.DataFrame({"date": days, "entity": name, "value": series})
        for name, series in values.items()
    ]
    return pd.concat(frames).dropna()


def test_turnover_snapshots_rank_on_prior_sessions_only():
    count = 40
    days = pd.bdate_range("2020-01-01", periods=count)
    first_feb = int(np.flatnonzero(days == pd.Timestamp("2020-02-03"))[0])
    big = [1.0] * count
    # B outranks A only from the snapshot day on; the snapshot must not see it.
    late = [0.5] * count
    late[first_feb:] = [100.0] * (count - first_feb)
    out = turnover_snapshots(_sessions({"A": big, "B": late}), top=1, window=10)
    feb = out[out["effective_date"] == pd.Timestamp("2020-02-03").date()]
    assert feb["symbol"].tolist() == ["A.NS"]
    assert out["effective_date"].min() == pd.Timestamp("2020-02-03").date()


def test_turnover_snapshots_require_traded_share_and_enough_names():
    count = 40
    sparse = [500.0 if i % 3 == 0 else np.nan for i in range(count)]
    values = {"A": [1.0] * count, "B": [2.0] * count, "S": sparse}
    out = turnover_snapshots(_sessions(values), top=2, window=10)
    assert set(out["symbol"]) == {"A.NS", "B.NS"}  # S trades on too few days
    assert turnover_snapshots(_sessions(values), top=3, window=10).empty


def _bars(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    closes_ = np.asarray(closes, dtype=float)
    opens_ = np.asarray(opens if opens is not None else closes, dtype=float)
    return pd.DataFrame(
        {
            "open": opens_,
            "high": np.maximum(opens_, closes_),
            "low": np.minimum(opens_, closes_),
            "close": closes_,
            "volume": 100.0,
            "value": closes_ * 100,
        },
        index=pd.bdate_range("2021-01-04", periods=len(closes_)),
    )


def _action(
    sessions: pd.DataFrame, position: int, terms: ActionTerms, face: float = 0.0
):
    return CorporateAction(sessions.index[position], terms, face)


def test_split_back_adjusts_prices_and_volume():
    sessions = _bars([100, 102, 51, 52])
    out, log = adjust_prices(
        sessions, [_action(sessions, 2, ActionTerms(split=2.0))], {}
    )
    np.testing.assert_allclose(out["close"], [50, 51, 51, 52])
    np.testing.assert_allclose(out["volume"], [200, 200, 100, 100])
    np.testing.assert_allclose(out["adj_close"], out["close"])
    np.testing.assert_allclose(out["dollar_volume"], sessions["value"])
    assert [entry[1] for entry in log] == ["split"]


def test_split_takes_the_source_ratio_closest_to_the_gap():
    sessions = _bars([100, 100, 20, 20])
    out, _ = adjust_prices(
        sessions,
        [_action(sessions, 2, ActionTerms(split=2.0))],
        {sessions.index[2]: 5.0},
    )
    np.testing.assert_allclose(out["close"], [20, 20, 20, 20])


def test_split_without_a_matching_gap_is_skipped():
    sessions = _bars([100, 100, 99, 99])
    out, log = adjust_prices(
        sessions, [_action(sessions, 2, ActionTerms(split=2.0))], {}
    )
    np.testing.assert_allclose(out["close"], sessions["close"])
    assert [entry[1] for entry in log] == ["split_skipped"]


def test_unlisted_clean_gap_is_inferred_but_a_crash_is_not():
    split, log = adjust_prices(_bars([300, 300, 100, 100]), [], {})
    np.testing.assert_allclose(split["close"], [100, 100, 100, 100])
    assert log[0][1:3] == ("split_inferred", 3.0)
    crash, log = adjust_prices(_bars([300, 300, 170, 170]), [], {})
    np.testing.assert_allclose(crash["close"], [300, 300, 170, 170])
    assert log[0][1] == "gap_unexplained"


def test_rights_gap_is_not_inferred_as_a_split():
    # A 1:1 rights issue at Rs 5 halves the price, like a clean 2:1 split.
    sessions = _bars([300, 300, 152.5, 152.5])
    rights = ActionTerms(rights=(1.0, 1.0, 0.0))
    out, log = adjust_prices(sessions, [_action(sessions, 2, rights, face=5.0)], {})
    np.testing.assert_allclose(out["close"], sessions["close"])
    np.testing.assert_allclose(out["adj_close"], [152.5, 152.5, 152.5, 152.5])
    assert [entry[1] for entry in log] == ["rights"]


def test_rights_with_unknown_price_skip_adjustment_and_split_inference():
    sessions = _bars([300, 300, 152.5, 152.5])
    rights = ActionTerms(rights=(1.0, 1.0, float("nan")))
    out, log = adjust_prices(sessions, [_action(sessions, 2, rights, face=5.0)], {})
    np.testing.assert_allclose(out["adj_close"], sessions["close"])
    assert [entry[1] for entry in log] == ["rights_skipped"]


def test_dividend_rights_and_demerger_adjust_only_adj_close():
    sessions = _bars(
        [100, 100, 90, 90, 90, 60, 60], opens=[100, 100, 90, 90, 90, 60, 60]
    )
    actions = [
        _action(sessions, 2, ActionTerms(dividend=10.0)),
        _action(sessions, 4, ActionTerms(rights=(1.0, 4.0, 35.0)), face=5.0),
        _action(sessions, 5, ActionTerms(demerger=True)),
    ]
    out, _ = adjust_prices(sessions, actions, {})
    np.testing.assert_allclose(out["close"], sessions["close"])
    dividend = 1 - 10 / 100
    rights = (4 * 90 + 1 * 40) / 5 / 90
    demerger = 60 / 90
    expected = sessions["close"].to_numpy() * np.array(
        [
            dividend * rights * demerger,
            dividend * rights * demerger,
            rights * demerger,
            rights * demerger,
            demerger,
            1,
            1,
        ]
    )
    np.testing.assert_allclose(out["adj_close"], expected)


def test_one_day_bad_print_is_dropped():
    sessions = _bars([100, 101, 5, 102, 103])
    out, log = adjust_prices(sessions, [], {})
    assert len(out) == 4 and 5 not in out["close"].tolist()
    assert log[0][1] == "bad_print_dropped"


class _Fallback:
    def __init__(self):
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append(list(tickers))
        return {ticker: pd.DataFrame({"close": [1.0]}) for ticker in tickers}


def test_price_fetcher_reads_frames_and_never_backfills_ns_from_fallback(tmp_path):
    frame = _bars([100, 50]).drop(columns="value")
    frame["adj_close"] = [80.0, 50.0]
    frame.to_parquet(tmp_path / "ABC.parquet")
    fallback = _Fallback()
    fetcher = BhavcopyPriceFetcher(fallback, directory=tmp_path)
    out = fetcher.fetch(
        ["ABC.NS", "GONE.NS", "^NSEI"], frame.index[0].date(), frame.index[-1].date()
    )
    np.testing.assert_allclose(out["ABC.NS"]["close"], [80, 50])
    assert out["GONE.NS"].empty
    assert fallback.calls == [["^NSEI"]]


def test_circuit_hits_counts_band_locks():
    close = pd.Series([100, 105, 105, 99.75, 100], dtype=float)
    bars = pd.DataFrame(
        {
            "close": close,
            "high": [100, 105, 106, 101, 101],
            "low": [99, 101, 104, 99.75, 99],
        }
    )
    np.testing.assert_allclose(circuit_hits(bars, window=5).iloc[-1], 2)


def test_prepare_bars_ranks_12_1_momentum_and_skips_last_month():
    days = pd.bdate_range("2020-01-01", periods=300)
    # FAST gets all its gain in the last month, so 12-1 momentum flips the rank.
    steady = np.linspace(100, 200, len(days))
    fast = np.r_[np.full(len(days) - 21, 100.0), np.linspace(100, 200, 21)]
    bars = {
        name: pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
            index=days,
        )
        for name, close in (("NSE:STEADY", steady), ("NSE:FAST", fast))
    }
    ctx = PrepareCtx(
        market="india",
        benchmark="^NSEI",
        bars_by_tv=bars,
        price_panel=bars,
        tv_symbols=list(bars),
        start=days[0].date(),
        end=days[-1].date(),
        fetcher=_Fallback(),
        warnings=[],
    )
    last = days[-1]
    plain = prepare_bars("momentum")(ctx)
    assert plain["NSE:FAST"].loc[last, "rank_score"] == pytest.approx(1.0)
    assert (
        plain["NSE:FAST"].loc[last, "rank_score"]
        > plain["NSE:STEADY"].loc[last, "rank_score"]
    )
    skip = prepare_bars("momentum", "mom_12_1")(ctx)
    assert skip["NSE:FAST"].loc[last, "mom_12_1"] == pytest.approx(0.0)
    assert skip["NSE:STEADY"].loc[last, "mom_12_1"] == pytest.approx(
        steady[-22] / steady[-253] - 1
    )
    assert (
        skip["NSE:STEADY"].loc[last, "rank_score"]
        > skip["NSE:FAST"].loc[last, "rank_score"]
    )


def test_fixed_mix_rebalances_monthly():
    days = pd.to_datetime(["2021-01-28", "2021-01-29", "2021-02-01"])
    returns = pd.DataFrame({"a": [0.0, 1.0, 0.0], "b": [0.0, 0.0, 1.0]}, index=days)
    out = fixed_mix(returns, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(out, [1.0, 1.5, 1.5 * 1.5])


def test_dry_powder_moves_sleeve_in_steps_and_resets_on_a_new_high():
    days = pd.bdate_range("2021-01-04", periods=4)
    momentum = pd.Series([1.0, 0.85, 0.7, 1.1], index=days)
    returns = pd.DataFrame({"mom": momentum.pct_change().fillna(0.0), "gold": 0.0})
    out = dry_powder(returns, momentum, {"gold": 1.0}, core=0.6, triggers=(-0.1, -0.2))
    day1 = 0.6 * 0.85 + 0.4  # -15%: one of two steps moves in at this close
    day2 = day1 * (1 + 0.8 * (0.7 / 0.85 - 1))  # -30%: whole sleeve moves in
    day3 = day2 * (1.1 / 0.7)  # new high: back to 60/40 at this close
    np.testing.assert_allclose(out, [1.0, day1, day2, day3])
