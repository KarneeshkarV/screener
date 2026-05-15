"""Deterministic 5-ticker OHLCV panel for backtester correctness audits."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from screener.backtester.pine import evaluate, parse


def _eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _ohlc_from_close(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    open_ = close.shift(1).fillna(float(close.iloc[0]) - 0.5)
    pair = pd.concat([open_, close], axis=1)
    high = pair.max(axis=1) + 0.5
    low = pair.min(axis=1) - 0.5
    return open_, high, low


def _finalize_frame(
    idx: pd.DatetimeIndex,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    *,
    entry_day0: float = 0.0,
    dividend: pd.Series | None = None,
) -> pd.DataFrame:
    ed0 = pd.Series(0.0, index=idx, dtype=float)
    if entry_day0 != 0.0:
        ed0.iloc[0] = float(entry_day0)
    if dividend is None:
        div = pd.Series(0.0, index=idx, dtype=float)
    else:
        div = dividend.reindex(idx).fillna(0.0).astype(float)
    frame = pd.DataFrame(
        {
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
            "entry_day0": ed0,
            "dividend": div,
        },
        index=idx,
    )
    for i in range(len(frame)):
        o = float(frame.iat[i, 0])
        h = float(frame.iat[i, 1])
        l_ = float(frame.iat[i, 2])
        c = float(frame.iat[i, 3])
        lo = min(o, c, l_)
        hi = max(o, c, h)
        frame.iat[i, 2] = lo
        frame.iat[i, 1] = hi
    return frame


def _validate_panel(panel: dict[str, pd.DataFrame]) -> None:
    idx = panel["TICKER_A"].index
    assert len(idx) == 120 and idx[0] == pd.Timestamp("2023-01-02")

    a = panel["TICKER_A"]
    assert float(a.iloc[40]["open"]) < float(a.iloc[39]["close"]) * 0.92
    assert float(a.iloc[40]["volume"]) == 500_000.0
    assert float(a["dividend"].iloc[50]) == 0.25
    assert abs(float(a["dividend"].sum()) - 0.25) < 1e-9
    cross_a = evaluate(parse("crossover(close, sma(close, 20))"), a)
    assert bool(cross_a.iloc[:30].any())

    b = panel["TICKER_B"]
    assert float(b.iloc[35]["open"]) > float(b.iloc[34]["close"]) * 1.12
    rsi_b = evaluate(parse("rsi(close, 14)"), b)
    assert bool((rsi_b.iloc[:25] < 30).any())

    c = panel["TICKER_C"]
    assert _eq(float(c.iloc[29]["close"]), float(c.iloc[0]["close"]) * 1.25)
    assert _eq(float(c.iloc[39]["close"]), float(c.iloc[29]["close"]) * 0.85)
    assert float(c["entry_day0"].iloc[0]) == 1.0

    d = panel["TICKER_D"]
    assert bool(d["close"].between(0.50, 2.50).all())
    assert float((d["close"] * d["volume"]).mean()) < 10_000.0
    rsi_d = evaluate(parse("rsi(close, 14)"), d)
    assert not bool((rsi_d < 30).fillna(False).any())
    assert not bool((rsi_d > 70).fillna(False).any())
    cross_d = evaluate(parse("crossover(close, sma(close, 20))"), d)
    assert not bool(cross_d.fillna(False).any())

    e = panel["TICKER_E"]
    rsi_e = evaluate(parse("rsi(close, 14)"), e)
    below = rsi_e < 30
    above = rsi_e > 70
    episodes = 0
    in_cold = False
    saw_hot = True
    for i in range(len(rsi_e)):
        if bool(below.iloc[i]) and not in_cold and saw_hot:
            episodes += 1
            in_cold = True
        if bool(above.iloc[i]):
            saw_hot = True
            in_cold = False
        elif not bool(below.iloc[i]):
            in_cold = False
    assert episodes >= 3

    for _name, df in panel.items():
        need = {"open", "high", "low", "close", "volume", "entry_day0", "dividend"}
        assert need <= set(df.columns)
        for i in range(len(df)):
            o = float(df.iloc[i]["open"])
            h = float(df.iloc[i]["high"])
            l_ = float(df.iloc[i]["low"])
            cl = float(df.iloc[i]["close"])
            assert l_ <= min(o, cl) + 1e-9
            assert h >= max(o, cl) - 1e-9


def make_audit_panel() -> dict[str, pd.DataFrame]:
    """Return five 120-bar OHLCV frames (business daily from 2023-01-02)."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2023-01-02", periods=120)
    n = len(idx)

    # TICKER_C
    c_close = np.empty(n, dtype=float)
    for i in range(n):
        if i <= 29:
            c_close[i] = 100.0 * (1.0 + 0.25 * (i / 29.0))
        elif i <= 39:
            t = (i - 29) / 10.0
            c_close[i] = 125.0 * (1.0 - 0.15 * t)
        else:
            c_close[i] = 106.25 + 0.02 * (i - 40)
    c_close_s = pd.Series(c_close, index=idx)
    o_c, h_c, l_c = _ohlc_from_close(c_close_s)
    vol_c = pd.Series(
        [300_000.0 + 100_000.0 * math.sin(2 * math.pi * i / 20.0) for i in range(n)],
        index=idx,
        dtype=float,
    )
    ticker_c = _finalize_frame(idx, o_c, h_c, l_c, c_close_s, vol_c, entry_day0=1.0)

    # TICKER_A: long flat base, one-bar thrust through SMA(20), then trend + gap
    a_close = np.empty(n, dtype=float)
    for i in range(n):
        if i <= 21:
            a_close[i] = 48.0
        elif i == 22:
            a_close[i] = 47.0
        elif i == 23:
            a_close[i] = 55.0
        elif i < 40:
            a_close[i] = a_close[i - 1] + 2.8 + 0.08 * (i - 24)
        elif i == 40:
            a_close[i] = a_close[39] * 0.88
        else:
            a_close[i] = a_close[i - 1] + float(rng.normal(0.05, 0.2))
    a_close_s = pd.Series(a_close, index=idx)
    o_a, h_a, l_a = _ohlc_from_close(a_close_s)
    prev_c = float(a_close_s.iloc[39])
    o_a.iloc[40] = prev_c * 0.86
    a_close_s.iloc[40] = float(o_a.iloc[40]) - 0.4
    l_a.iloc[40] = min(float(l_a.iloc[40]), float(o_a.iloc[40]) - 3.0, prev_c * 0.70)
    h_a.iloc[40] = max(float(o_a.iloc[40]), float(a_close_s.iloc[40])) + 0.6
    div_a = pd.Series(0.0, index=idx)
    div_a.iloc[50] = 0.25
    ticker_a = _finalize_frame(
        idx, o_a, h_a, l_a, a_close_s, pd.Series(500_000.0, index=idx), dividend=div_a
    )

    # TICKER_B: deep early dip for RSI<30, then gap-up bar 35
    b_close = np.empty(n, dtype=float)
    for i in range(n):
        if i == 0:
            b_close[i] = 45.0
        elif i < 12:
            b_close[i] = b_close[i - 1] - 1.35
        elif i < 35:
            b_close[i] = b_close[i - 1] + 0.35 + 0.02 * math.sin(i)
        elif i == 35:
            b_close[i] = b_close[34] * 1.08
        else:
            b_close[i] = b_close[i - 1] + float(rng.normal(0.0, 0.05))
    b_close_s = pd.Series(b_close, index=idx)
    o_b, h_b, l_b = _ohlc_from_close(b_close_s)
    prev_cb = float(b_close_s.iloc[34])
    o_b.iloc[35] = prev_cb * 1.14
    b_close_s.iloc[35] = float(o_b.iloc[35]) + 0.3
    h_b.iloc[35] = max(float(o_b.iloc[35]), float(b_close_s.iloc[35])) + 0.5
    l_b.iloc[35] = min(float(o_b.iloc[35]), float(b_close_s.iloc[35])) - 0.2
    ticker_b = _finalize_frame(
        idx, o_b, h_b, l_b, b_close_s, pd.Series(400_000.0, index=idx)
    )

    # TICKER_D: perfectly flat close → RSI NaN (no extremes), still fails liquidity filters
    d_close = pd.Series(1.0, index=idx, dtype=float)
    o_d, h_d, l_d = _ohlc_from_close(d_close)
    ticker_d = _finalize_frame(
        idx, o_d, h_d, l_d, d_close, pd.Series(3000.0, index=idx)
    )

    # TICKER_E
    e_close = np.empty(n, dtype=float)
    e_close[0] = 100.0
    segments = [
        (1, 18, -1.15),
        (19, 28, 2.2),
        (29, 48, -1.05),
        (49, 58, 2.35),
        (59, 78, -1.08),
        (79, 88, 2.4),
        (89, 108, -1.05),
        (109, 119, 1.2),
    ]
    for lo, hi, step in segments:
        for i in range(lo, min(hi + 1, n)):
            e_close[i] = e_close[i - 1] + step + float(rng.normal(0, 0.08))
    e_close_s = pd.Series(e_close, index=idx)
    o_e, h_e, l_e = _ohlc_from_close(e_close_s)
    ticker_e = _finalize_frame(
        idx, o_e, h_e, l_e, e_close_s, pd.Series(250_000.0, index=idx)
    )

    audit = pd.Series(0.0, index=idx)
    audit.iloc[60] = 1.0
    for t in (ticker_a, ticker_b, ticker_c):
        t["audit_signal"] = audit.astype(float)
    ticker_d["audit_signal"] = pd.Series(0.0, index=idx)
    ticker_e["audit_signal"] = pd.Series(0.0, index=idx)

    panel = {
        "TICKER_A": ticker_a,
        "TICKER_B": ticker_b,
        "TICKER_C": ticker_c,
        "TICKER_D": ticker_d,
        "TICKER_E": ticker_e,
    }
    _validate_panel(panel)
    return panel
