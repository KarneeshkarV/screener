"""Price-volume & flow strategies built on pure OHLCV volume.

Three evidence-grounded volume/flow signals, all computable from OHLCV alone
(no delivery/fundamental data required):

``volume_surge``      — Gervais-Kaniel-Mingelgrin (2001) "high volume return
                        premium": a stock whose daily volume is unusually high
                        relative to its own recent norm earns positive excess
                        returns over the following ~20 trading days. Entry on a
                        >= 2x volume surge vs the prior 50-day mean with a
                        bullish close and mild 20-day uptrend; short 21-day
                        hold captures the documented one-month horizon. This is
                        a *volume* event, not a price-range breakout — the
                        failed repo breakout family (donchian/turtle/chandelier)
                        triggers on price extremes with no volume requirement
                        and holds 100-500 days.
``obv_flow_trend``    — Granville (1963) On-Balance Volume as a flow
                        confirmation of price trend: enter when OBV is above
                        its 60-day flow average (accumulation) AND price is
                        above its 60-day SMA (uptrend); exit when OBV breaks
                        below its flow average (distribution), i.e. flow and
                        price disagree. Hsu & Kuan (2005, J. Financial
                        Econometrics) found OBV rules significantly profitable
                        on S&P 500 under data-snooping checks; Blume, Easley &
                        O'Hara (1994, J. Finance) give the theory — volume
                        signals the *reliability* of a price move, so requiring
                        flow agreement filters momentum entries that lack
                        institutional support (late/exhausted trends).
``cmf_flow_factor``   — cross-sectional Chaikin Money Flow factor: CMF(20)
                        measures 20-day buying vs selling pressure as
                        sum(money-flow-volume)/sum(volume). Names are ranked
                        each day by their CMF percentile (the nifty_momentum
                        rank pattern); entry gates on positive CMF plus price
                        above the 60-day SMA. Volume-weighted flow selection is
                        the cross-sectional cousin of the volume-informed
                        trading literature (Easley & O'Hara 1987/1992, J.
                        Finance) and of Chordia & Swaminathan (2000, J. Finance)
                        — high-flow names lead price adjustment.

All indicators are causal (bar ``t`` uses bars ``<= t``). No delivery data is
needed; the strategies run identically on US and India.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

_VOL_BASELINE_50 = 50  # volume-surge baseline (prior bars, current excluded)
_TREND_SMA = 20  # volume-surge mild uptrend filter
_OBV_FAST = 20  # OBV fast flow average
_OBV_SLOW = 60  # OBV slow flow average
_CMF_WINDOW = 20  # Chaikin money flow window
_CMF_TREND_SMA = 60  # CMF factor trend gate


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative signed volume (Granville OBV); no-change days add zero."""
    direction = pd.Series(
        np.sign(close.astype(float).diff().fillna(0.0)),
        index=close.index,
        dtype=float,
    )
    return (direction * volume.astype(float)).cumsum()


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = _CMF_WINDOW,
) -> pd.Series:
    """Chaikin Money Flow over ``window`` bars, in [-1, 1].

    MFM = ((close - low) - (high - close)) / (high - low); bars where
    high == low (halted/limit days) contribute no money-flow volume.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    volume = volume.astype(float)
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm * volume
    return (
        mfv.rolling(window, min_periods=window).sum()
        / volume.rolling(window, min_periods=window).sum()
    )


def _prepare_volume_surge(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        volume = frame["volume"].astype(float)
        # Baseline excludes the current bar (volume[t] vs the prior 50-bar mean).
        baseline_50 = (
            volume.shift(1)
            .rolling(_VOL_BASELINE_50, min_periods=_VOL_BASELINE_50)
            .mean()
        )
        frame["vol_ratio_50"] = volume / baseline_50
        out[tv] = frame
    return out


def _prepare_obv_trend(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        frame["obv"] = on_balance_volume(frame["close"], frame["volume"])
        frame["obv_ma20"] = (
            frame["obv"].rolling(_OBV_FAST, min_periods=_OBV_FAST).mean()
        )
        frame["obv_ma60"] = (
            frame["obv"].rolling(_OBV_SLOW, min_periods=_OBV_SLOW).mean()
        )
        out[tv] = frame
    return out


def _prepare_cmf_factor(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    cmf: dict[str, pd.Series] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        cmf[tv] = chaikin_money_flow(
            bars["high"], bars["low"], bars["close"], bars["volume"]
        )
    if not cmf:
        return ctx.bars_by_tv
    # Per-day cross-sectional percentile of CMF (same rank pattern as
    # nifty_momentum): strongest buying pressure ranks first.
    cmf_panel = pd.DataFrame(cmf)
    cmf_pct = cmf_panel.rank(axis=1, pct=True)

    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        frame["cmf_20"] = cmf[tv].reindex(frame.index)
        frame["cmf_pct"] = cmf_pct[tv].reindex(frame.index)
        frame["rank_score"] = frame["cmf_pct"]
        out[tv] = frame
    return out


def _lookback_volume_surge() -> int:
    # vol baseline needs 50 prior bars + current; sma(close, 20) needs 20.
    return _VOL_BASELINE_50


def _lookback_obv_trend() -> int:
    # obv_ma60 and sma(close, 60) both need 60 bars.
    return _OBV_SLOW


def _lookback_cmf() -> int:
    # sma(close, 60) needs 60 bars; CMF needs 20.
    return _CMF_TREND_SMA


register_expression_strategy(
    "volume_surge",
    entry=(
        f"vol_ratio_50 >= 2.0 and close > open and close > sma(close, {_TREND_SMA})"
    ),
    exit=None,
    prepare_bars=_prepare_volume_surge,
    required_lookback=_lookback_volume_surge,
)

register_expression_strategy(
    "obv_flow_trend",
    entry=f"obv_ma20 > obv_ma60 and close > sma(close, {_OBV_SLOW})",
    exit="crossunder(obv_ma20, obv_ma60)",
    prepare_bars=_prepare_obv_trend,
    required_lookback=_lookback_obv_trend,
)

register_expression_strategy(
    "cmf_flow_factor",
    entry=f"cmf_20 > 0 and close > sma(close, {_CMF_TREND_SMA})",
    exit=None,
    prepare_bars=_prepare_cmf_factor,
    required_lookback=_lookback_cmf,
)
