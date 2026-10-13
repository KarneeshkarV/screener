"""Regression tests for provider scales, bar completion, and causal fundamentals."""

from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from screener.backtester import data, fundamentals, price_cache, price_frames, sessions
from screener.providers import FakeProvider


def bars(index, close=100.0):
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=index,
    )


def yahoo_download_frame(index, close=100.0):
    frame = bars(index, close).rename(columns=str.title)
    frame["Adj Close"] = close
    frame["Stock Splits"] = 0.0
    return frame


def test_yahoo_splits_only_preserves_provider_split_adjusted_prices(
    tmp_path, monkeypatch
):
    index = pd.bdate_range("2024-10-24", "2024-10-29")
    raw = yahoo_download_frame(index, [50.0, 50.0, 50.25, 50.5])
    raw.loc["2024-10-28", "Stock Splits"] = 2.0
    monkeypatch.setattr(data, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        data._optional, "load", lambda _: SimpleNamespace(download=lambda *a, **k: raw)
    )
    fetcher = data.YFinancePriceFetcher(cache_dir=tmp_path, auto_adjust=False)
    for _ in range(2):
        fetched = fetcher.fetch(["RELIANCE.NS"], index[0].date(), index[-1].date())
        adjusted = price_frames.apply_splits_only_adjustment(fetched)["RELIANCE.NS"]
        assert adjusted.close.tolist() == raw.Close.tolist()
        assert adjusted.volume.tolist() == raw.Volume.tolist()
        assert adjusted.close.pct_change().loc["2024-10-28"] == pytest.approx(0.005)


@pytest.mark.parametrize("direction", ["extend", "backfill"])
@pytest.mark.parametrize("provider", ["yahoo", "fmp"])
def test_cache_range_update_matches_cold_download(
    tmp_path, monkeypatch, direction, provider
):
    scale = [100.0]
    requests = []

    def download(ticker, **kwargs):
        requests.append((pd.Timestamp(kwargs["start"]), pd.Timestamp(kwargs["end"])))
        return yahoo_download_frame(
            pd.bdate_range(kwargs["start"], kwargs["end"], inclusive="left"), scale[0]
        )

    class Client:
        def get(self, path, params):
            index = pd.bdate_range(params["from"], params["to"])
            requests.append((index.min(), index.max()))
            frame = bars(index, scale[0])
            return {
                "historical": [
                    {"date": str(stamp), **row}
                    for stamp, row in frame.to_dict("index").items()
                ]
            }

    monkeypatch.setattr(data, "_configure_yfinance", lambda: None)
    monkeypatch.setattr(
        data._optional, "load", lambda _: SimpleNamespace(download=download)
    )

    def fetcher(directory):
        if provider == "yahoo":
            return data.YFinancePriceFetcher(cache_dir=directory)
        result = data.FMPPriceFetcher(api_key="offline", cache_dir=directory)
        result.client = Client()
        return result

    warm_fetcher = fetcher(tmp_path)
    initial = (
        (date(2020, 1, 1), date(2020, 1, 31))
        if direction == "extend"
        else (date(2020, 1, 15), date(2020, 2, 10))
    )
    warm_fetcher.fetch(["AAA"], *initial)
    scale[0] = 50.0
    warm = warm_fetcher.fetch(["AAA"], date(2020, 1, 1), date(2020, 2, 10))["AAA"]
    cold = fetcher(tmp_path / "cold").fetch(
        ["AAA"], date(2020, 1, 1), date(2020, 2, 10)
    )["AAA"]
    pd.testing.assert_frame_equal(warm, cold)
    assert warm.close.pct_change().dropna().eq(0).all()


@pytest.mark.parametrize(
    "interval,stamps",
    [
        ("5m", ["2025-06-20 14:10", "2025-06-20 14:15"]),
        ("1h", ["2025-06-20 13:00", "2025-06-20 14:00"]),
    ],
)
@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_intraday_cutoff_uses_bar_completion_utc(interval, stamps, timezone):
    index = pd.DatetimeIndex(stamps)
    if timezone:
        index = index.tz_localize("UTC").tz_convert(timezone)
    frame = bars(index)
    now = datetime(2025, 6, 20, 14, 17, tzinfo=UTC)
    kept = sessions.drop_incomplete_sessions(frame, "AAA", interval=interval, now=now)
    pd.testing.assert_frame_equal(kept, frame.iloc[:1])
    completed_at = (
        index[-1].tz_localize("UTC")
        if index.tz is None
        else index[-1].tz_convert("UTC")
    )
    completed_at += pd.Timedelta(minutes=5 if interval == "5m" else 60)
    pd.testing.assert_frame_equal(
        sessions.drop_incomplete_sessions(
            frame, "AAA", interval=interval, now=completed_at.to_pydatetime()
        ),
        frame,
    )


def test_fmp_fetcher_does_not_store_unfinished_intraday_bar(tmp_path, monkeypatch):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 6, 20, 14, 17, tzinfo=UTC).astimezone(tz)

    class Client:
        def get(self, path, params):
            return [
                {
                    "date": stamp,
                    "open": 100,
                    "high": 100,
                    "low": 100,
                    "close": 100,
                    "volume": 1000,
                }
                for stamp in ["2025-06-20 10:10", "2025-06-20 10:15"]
            ]

    monkeypatch.setattr(sessions, "datetime", FrozenDatetime)
    fetcher = data.FMPPriceFetcher(api_key="offline", cache_dir=tmp_path, interval="5m")
    fetcher.client = Client()
    fetched = fetcher.fetch(["AAA"], date(2025, 6, 20), date(2025, 6, 20))["AAA"]
    assert fetched.index.tolist() == [pd.Timestamp("2025-06-20 14:10")]
    cached = price_cache.load_cached_frame(
        data._fmp_cache_key("AAA", True, "5m"), tmp_path, "5m"
    )
    pd.testing.assert_frame_equal(cached, fetched)


@pytest.mark.parametrize("symbol", ["RELIANCE.NS", "^NSEI", "^NSEBANK", "^BSESN"])
def test_india_benchmark_uses_india_completion_clock(symbol):
    frame = bars(pd.DatetimeIndex(["2025-06-19", "2025-06-20"]))
    now = datetime(2025, 6, 20, 12, 30, tzinfo=UTC)
    pd.testing.assert_frame_equal(
        sessions.drop_incomplete_sessions(frame, symbol, now=now), frame
    )


def test_fmp_fields_follow_each_section_publication_date():
    payload = {
        "income": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-05", "revenue": 100}
        ],
        "balance": [
            {
                "date": "2023-12-31",
                "acceptedDate": "2024-02-20",
                "totalDebtToEquity": 0.4,
            }
        ],
        "ratios": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-25", "returnOnEquity": 0.18}
        ],
        "key_metrics": [
            {"date": "2023-12-31", "acceptedDate": "2024-03-01", "peRatioTTM": 19}
        ],
        "enterprise_values": [
            {
                "date": "2023-12-31",
                "acceptedDate": "2024-03-05",
                "marketCapitalization": 1000000,
            }
        ],
    }
    fields = ("debt_to_equity", "roe_ttm", "pe_ttm", "market_cap")
    frame = fundamentals._normalize_fmp_payload(payload, fields=fields, lag_days=1)
    expected = {
        "debt_to_equity": ("2024-02-21", 0.4),
        "roe_ttm": ("2024-02-26", 18),
        "pe_ttm": ("2024-03-02", 19),
        "market_cap": ("2024-03-06", 1000000),
    }
    for field, (stamp, value) in expected.items():
        assert frame[field].first_valid_index() == pd.Timestamp(stamp)
        assert frame.loc[stamp, field] == value


def test_fmp_dated_section_does_not_require_income():
    payload = {
        "ratios": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-20", "debtEquityRatio": 0.0}
        ]
    }
    frame = fundamentals._normalize_fmp_payload(
        payload, fields=("debt_to_equity",), lag_days=1
    )
    assert frame.loc["2024-02-21", "debt_to_equity"] == 0.0


def test_fmp_income_growth_waits_for_late_source_publication():
    payload = {
        "income": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-20", "revenue": 100},
            {"date": "2024-12-31", "acceptedDate": "2025-02-05", "revenue": 150},
            {"date": "2024-09-30", "acceptedDate": "2025-02-25", "revenue": 130},
            {"date": "2024-06-30", "acceptedDate": "2024-08-05", "revenue": 120},
        ]
    }
    frame = fundamentals._normalize_fmp_payload(
        payload, fields=("revenue_growth_yoy", "revenue_up_3q"), lag_days=1
    )
    assert frame.revenue_growth_yoy.first_valid_index() == pd.Timestamp("2025-02-06")
    assert frame.loc["2025-02-06", "revenue_growth_yoy"] == 50.0
    assert frame.revenue_up_3q.first_valid_index() == pd.Timestamp("2025-02-26")
    assert frame.loc["2025-02-26", "revenue_up_3q"] == 1.0


def test_fmp_undated_sections_are_not_published_with_income():
    payload = {
        "income": [
            {"date": "2023-12-31", "acceptedDate": "2024-02-05", "revenue": 100}
        ],
        "ratios": [{"date": "2023-12-31", "returnOnEquity": 0.18}],
    }
    frame = fundamentals._normalize_fmp_payload(
        payload, fields=("roe_ttm",), lag_days=1
    )
    assert frame.roe_ttm.dropna().empty


@pytest.mark.parametrize("provider", ["fmp", "openscreener", "yfinance"])
def test_fundamental_fetch_retains_latest_prior_observation_per_field(
    monkeypatch, provider
):
    index = pd.DatetimeIndex(["2024-01-01", "2024-02-06", "2024-03-02", "2024-04-01"])
    frame = pd.DataFrame(
        {"roe_ttm": [18.0, None, None, 25.0], "debt_to_equity": [None, 0.4, 0.3, None]},
        index=index,
    )
    monkeypatch.setattr(
        fundamentals, "_normalize_fmp_payload", lambda *a, **k: frame.copy()
    )
    monkeypatch.setattr(
        fundamentals, "_normalize_openscreener_payload", lambda *a, **k: frame.copy()
    )
    for name in [
        "_FMP_PROVIDER",
        "_OPENSCREENER_PROVIDER",
        "_YFINANCE_FUNDAMENTALS_PROVIDER",
    ]:
        monkeypatch.setattr(fundamentals, name, FakeProvider())
    monkeypatch.setattr(fundamentals, "_fetch_fmp_sections", lambda *a, **k: {})
    monkeypatch.setattr(
        fundamentals, "_fetch_openscreener_quarterly", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        fundamentals, "_fetch_yfinance_quarterly_revenue", lambda *a, **k: {}
    )
    types = {
        "fmp": fundamentals.FMPFundamentalFetcher,
        "openscreener": fundamentals.OpenScreenerFundamentalFetcher,
        "yfinance": fundamentals.YFinanceFundamentalFetcher,
    }
    kwargs = {"api_key": "offline"} if provider == "fmp" else {}
    fetcher = types[provider](fields=tuple(frame.columns), **kwargs)
    result = fetcher.fetch(["AAA"], date(2024, 3, 1), date(2024, 3, 4))["AAA"]
    enriched = fundamentals.merge_fundamentals_into_bars(
        {"AAA": bars(pd.date_range("2024-03-01", "2024-03-04", freq="h"))},
        {"AAA": result},
        {"AAA": "AAA"},
        filing_lag_days=1,
    )["AAA"]
    assert enriched.roe_ttm.eq(18.0).all()
    assert enriched.loc["2024-03-01", "debt_to_equity"].eq(0.4).all()
    assert enriched.loc["2024-03-02", "debt_to_equity"].eq(0.3).all()
    assert result.index.max() <= pd.Timestamp("2024-03-04")
