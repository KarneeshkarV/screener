from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from screener.options.greeks import black_scholes_price
from screener.options.metrics import compute_chain_metrics
from screener.options.nse_bhavcopy import (
    load_bhavcopy_chains,
    normalize_bhavcopy_options,
)
from screener.unusual_volume.nse_client import (
    fo_bhavcopy_cache_path,
    normalize_legacy_fo_bhavcopy,
    read_fo_bhavcopy_raw,
)


def _legacy_frame() -> pd.DataFrame:
    """A tiny legacy-shaped FO bhavcopy (with the trailing junk column)."""
    return pd.DataFrame(
        {
            "INSTRUMENT": ["OPTSTK", "OPTSTK", "FUTSTK", "OPTIDX"],
            "SYMBOL": ["RELIANCE", "RELIANCE", "RELIANCE", "NIFTY"],
            "EXPIRY_DT": ["25-Jan-2023", "25-Jan-2023", "25-Jan-2023", "25-Jan-2023"],
            "STRIKE_PR": [2500.0, 2500.0, 0.0, 18000.0],
            "OPTION_TYP": ["CE", "PE", "XX", "CE"],
            "OPEN": [10.0, 8.0, 2500.0, 100.0],
            "HIGH": [12.0, 9.0, 2550.0, 120.0],
            "LOW": [9.0, 7.0, 2480.0, 90.0],
            "CLOSE": [11.0, 8.5, 2530.0, 110.0],
            "SETTLE_PR": [11.0, 8.5, 2530.0, 110.0],
            "CONTRACTS": [1000, 800, 500, 2000],
            "VAL_INLAKH": [11.0, 6.8, 12.6, 22.0],
            "OPEN_INT": [50000, 40000, 30000, 60000],
            "CHG_IN_OI": [5000, -2000, 1000, 3000],
            "TIMESTAMP": ["02-JAN-2023"] * 4,
            "Unnamed: 15": [float("nan")] * 4,
        }
    )


def test_normalize_legacy_maps_columns_instruments_and_dates() -> None:
    out = normalize_legacy_fo_bhavcopy(_legacy_frame())
    # UDiff column names present; legacy names and junk gone.
    for col in ("TckrSymb", "XpryDt", "StrkPric", "OptnTp", "ClsPric", "OpnIntrst"):
        assert col in out.columns
    assert "INSTRUMENT" not in out.columns
    assert not any(str(c).startswith("Unnamed") for c in out.columns)
    # Instrument codes mapped to UDiff.
    assert list(out["FinInstrmTp"]) == ["STO", "STO", "STF", "IDO"]
    # Dates parsed to datetime64.
    assert pd.api.types.is_datetime64_any_dtype(out["TradDt"])
    assert out["TradDt"].iloc[0] == pd.Timestamp(2023, 1, 2)
    assert out["XpryDt"].iloc[0] == pd.Timestamp(2023, 1, 25)


def test_read_fo_bhavcopy_raw_normalizes_cached_legacy(tmp_path: Path) -> None:
    d = date(2023, 1, 2)
    cache_path = fo_bhavcopy_cache_path(d, tmp_path)
    _legacy_frame().to_csv(cache_path, index=False)
    # Cached legacy CSV must be normalized even without a fresh download.
    df = read_fo_bhavcopy_raw(
        d,
        cache_root=tmp_path,
        archive_url_template="unused-{yyyymmdd}",
    )
    assert "FinInstrmTp" in df.columns
    assert set(df["FinInstrmTp"]) == {"STO", "STF", "IDO"}


def test_normalize_bhavcopy_options_builds_chain_from_legacy() -> None:
    frame = normalize_legacy_fo_bhavcopy(_legacy_frame())
    chains = normalize_bhavcopy_options(frame, as_of=date(2023, 1, 2))
    # STO (RELIANCE) and IDO (NIFTY) both yield option chains; FUTSTK is dropped.
    assert set(chains) == {"RELIANCE", "NIFTY"}
    chain = chains["RELIANCE"]
    assert chain.market == "india"
    # Legacy frames have no UndrlygPric, so spot is unresolved here (commit 2).
    assert chain.spot is None
    rights = {contract.right for contract in chain.contracts}
    assert rights == {"call", "put"}


def test_spot_prices_fill_spot_when_underlying_missing() -> None:
    frame = normalize_legacy_fo_bhavcopy(_legacy_frame())
    chains = normalize_bhavcopy_options(
        frame, as_of=date(2023, 1, 2), spot_prices={"RELIANCE": 2600.0}
    )
    assert chains["RELIANCE"].spot == 2600.0
    # NIFTY has no cash close (index) -> stays None.
    assert chains["NIFTY"].spot is None


def test_load_bhavcopy_chains_legacy_uses_injected_cash_closes() -> None:
    legacy = normalize_legacy_fo_bhavcopy(_legacy_frame())

    def fetcher(_d: date) -> pd.DataFrame:
        return legacy

    chains = load_bhavcopy_chains(
        date(2023, 1, 2),
        fetcher=fetcher,
        cash_fetcher=lambda _d: {"RELIANCE": 2600.0},
    )
    assert chains["RELIANCE"].spot == 2600.0

    # Missing cash mapping leaves spot unresolved.
    chains_none = load_bhavcopy_chains(
        date(2023, 1, 2), fetcher=fetcher, cash_fetcher=lambda _d: {}
    )
    assert chains_none["RELIANCE"].spot is None


def test_load_bhavcopy_chains_udiff_unaffected_by_cash_fetcher() -> None:
    fixture = Path(__file__).parent / "fixtures" / "nse_fo_bhavcopy_options_sample.csv"
    udiff = pd.read_csv(fixture)
    calls: list[date] = []

    def cash_fetcher(d: date) -> dict[str, float]:
        calls.append(d)
        return {"RELIANCE": 9999.0}

    # 2026-07-08 is post-UDiff: spot comes from UndrlygPric, cash_fetcher unused.
    chains = load_bhavcopy_chains(
        date(2026, 7, 8), fetcher=lambda _d: udiff, cash_fetcher=cash_fetcher
    )
    assert chains["RELIANCE"].spot == 1275.9
    assert calls == []


_RATE = 0.065
_TRAD_DT = "2024-01-01"
_XPRY_DT = "2024-01-31"
_TIME_YEARS = 30 / 365.25


def _iv_frame(vol: float = 0.30, spot: float = 100.0) -> pd.DataFrame:
    """A synthetic UDiff frame whose ATM call/put are priced with ``vol``."""
    rows = []
    for right in ("CE", "PE"):
        price = black_scholes_price(
            spot,
            100.0,
            _TIME_YEARS,
            _RATE,
            vol,
            "call" if right == "CE" else "put",
        )
        rows.append(
            {
                "TradDt": _TRAD_DT,
                "FinInstrmTp": "STO",
                "TckrSymb": "ABC",
                "XpryDt": _XPRY_DT,
                "StrkPric": 100.0,
                "OptnTp": right,
                "FinInstrmNm": f"ABC24JAN100{right}",
                "ClsPric": price,
                "LastPric": price,
                "PrvsClsgPric": price,
                "UndrlygPric": spot,
                "SttlmPric": price,
                "OpnIntrst": 5000,
                "ChngInOpnIntrst": 100,
                "TtlTradgVol": 200,
                "NewBrdLotQty": 500,
            }
        )
    # A dead row (no OI, no volume) that must be skipped by IV derivation.
    rows.append(
        {
            "TradDt": _TRAD_DT,
            "FinInstrmTp": "STO",
            "TckrSymb": "ABC",
            "XpryDt": _XPRY_DT,
            "StrkPric": 130.0,
            "OptnTp": "CE",
            "FinInstrmNm": "ABC24JAN130CE",
            "ClsPric": 0.5,
            "LastPric": 0.5,
            "PrvsClsgPric": 0.5,
            "UndrlygPric": spot,
            "SttlmPric": 0.5,
            "OpnIntrst": 0,
            "ChngInOpnIntrst": 0,
            "TtlTradgVol": 0,
            "NewBrdLotQty": 500,
        }
    )
    return pd.DataFrame(rows)


def test_derived_iv_round_trips_known_vol() -> None:
    chains = normalize_bhavcopy_options(_iv_frame(vol=0.30), as_of=date(2024, 1, 1))
    contracts = {c.strike: c for c in chains["ABC"].contracts if c.right == "call"}
    atm = contracts[100.0]
    assert atm.iv == pytest.approx(0.30, abs=1e-3)
    # Greeks populated alongside a found IV.
    assert atm.delta is not None and 0 < atm.delta < 1
    assert atm.gamma is not None and atm.vega is not None
    # Dead row skipped -> no IV.
    assert contracts[130.0].iv is None
    assert contracts[130.0].delta is None


def test_derive_iv_disabled_leaves_iv_none() -> None:
    chains = normalize_bhavcopy_options(
        _iv_frame(), as_of=date(2024, 1, 1), derive_iv=False
    )
    assert all(c.iv is None for c in chains["ABC"].contracts)


def test_metrics_use_bhavcopy_derived_iv() -> None:
    chains = normalize_bhavcopy_options(_iv_frame(vol=0.30), as_of=date(2024, 1, 1))
    metrics = compute_chain_metrics(chains["ABC"])
    assert metrics.atm_iv == pytest.approx(0.30, abs=1e-3)
    assert metrics.median_iv is not None


def test_risk_free_rate_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCREENER_INDIA_RISK_FREE_RATE", "0.10")
    # Price the frame at rate 0.065 but invert at 0.10 -> IV shifts off 0.30.
    chains = normalize_bhavcopy_options(_iv_frame(vol=0.30), as_of=date(2024, 1, 1))
    atm = next(
        c for c in chains["ABC"].contracts if c.right == "call" and c.strike == 100.0
    )
    assert atm.iv is not None
    assert abs(atm.iv - 0.30) > 1e-3
