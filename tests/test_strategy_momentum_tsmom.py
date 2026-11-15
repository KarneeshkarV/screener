"""Unit tests for 12-1 momentum confirmed by time-series momentum."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from screener.strategies.plugins.momentum_12_1 import momentum_12_1_score
from screener.strategies.plugins.momentum_12_1_tsmom import (
    ENTRY_TSMOM,
    trailing_return,
)
from screener.strategies.spec import PrepareCtx, discover_plugins, registry
from tests.conftest import StubPriceFetcher

_N = 320
_INDEX = pd.bdate_range("2022-01-03", periods=_N)


def _bars(close: np.ndarray) -> pd.DataFrame:
    series = pd.Series(close, index=_INDEX)
    return pd.DataFrame(
        {
            "open": series,
            "high": series * 1.001,
            "low": series * 0.999,
            "close": series,
            "volume": 500_000.0,
        }
    )


def _prepare(bars_by_tv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    discover_plugins()
    spec = registry.get("momentum_12_1_tsmom")
    ctx = PrepareCtx(
        market="us",
        benchmark="SPY",
        bars_by_tv=bars_by_tv,
        price_panel=bars_by_tv,
        tv_symbols=list(bars_by_tv),
        start=date(2023, 1, 1),
        end=date(2023, 3, 1),
        fetcher=StubPriceFetcher(bars_by_tv),
        warnings=[],
    )
    return spec.prepare_bars(ctx)


def test_strategy_registered() -> None:
    discover_plugins()
    spec = registry.get("momentum_12_1_tsmom")
    assert spec.entry == ENTRY_TSMOM
    assert "mom_xs_rank <= 20" in spec.entry
    assert "tsmom_12 > 0" in spec.entry
    assert spec.required_lookback is not None
    assert spec.required_lookback() == 253


def test_trailing_return_does_not_skip_last_month() -> None:
    close = pd.Series(np.linspace(100.0, 200.0, _N), index=_INDEX)
    ret = trailing_return(close, None, 252)
    assert ret.iloc[:252].isna().all()
    t = 300
    assert ret.iloc[t] == close.iloc[t] / close.iloc[t - 252] - 1.0


def test_rolled_over_winner_keeps_12_1_but_fails_short_tsmom() -> None:
    # Rallies for 11 months, then gives back part of it in the last month: the
    # 12-1 window still shows a big gain, the 1-month trailing return does not.
    up = 100.0 * 1.003 ** np.arange(_N)
    rolled = up.copy()
    rolled[-21:] = np.linspace(up[-22], up[-22] * 0.8, 21)
    steady = 100.0 * 1.001 ** np.arange(_N)

    out = _prepare({"ROLL": _bars(rolled), "STEADY": _bars(steady)})
    roll, std = out["ROLL"].iloc[-1], out["STEADY"].iloc[-1]

    assert roll["mom_12_1"] == momentum_12_1_score(out["ROLL"]["close"]).iloc[-1]
    assert roll["mom_xs_rank"] == 1
    assert std["mom_xs_rank"] == 2
    assert roll["rank_score"] == roll["mom_12_1"]
    assert roll["tsmom_1"] < 0 < std["tsmom_1"]
    assert roll["tsmom_12"] > 0
    assert roll["tsmom_12_vol"] > 0
