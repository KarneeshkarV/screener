"""Delivery-based accumulation with price/flow confirmation (India-first).

NSE publishes daily cash-segment delivery data (bhavcopy ``DELIV_PER``: the
share of traded quantity that was settled by delivery rather than squared off).
Rising delivery percentage on a stock is the standard Indian institutional
proxy for accumulation — delivery buyers take and hold shares, whereas
intraday squared-off volume is speculative churn. The repo already treats high
delivery as an "institutional footprint" (``screener/unusual_volume/delivery.py``
uses delivery_rvol >= 2 as a conviction signal, and ``rs_breakout`` requires
rising delivery as a confirmation). Practitioner evidence for Indian equities
repeatedly links rising delivery % to positive short/medium-horizon returns
(e.g. NSE/stock-broker research notes on delivery-based accumulation; see the
FAMILY_BRIEF for the repo-internal treatment).

Rule:
    entry: close > anchored VWAP          (institutional-favoured territory)
           and OBV > OBV-20-SMA            (flow accumulation, pure OHLCV)
           and (delivery unavailable OR delivery_pct above its prior 20-day SMA)
    exit : crossunder(close, VWAP)         (price breaks the accumulation zone)
    --hold caps the maximum holding period (quarterly rotation ~63 days).

Delivery is OPTIONAL: when the delivery panel cannot be loaded (US market,
fetch failure, or a symbol missing from the bhavcopy), ``delivery_active`` is
False and the rule degrades to the pure-OHLCV flow leg (VWAP + OBV), so the
strategy still runs on any market. Delivery data is used as published at the
end of day ``t`` (same convention as ``rs_breakout``), so entry at bar ``t``
close is causal.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from screener.strategies.plugins.volume_flow import on_balance_volume
from screener.strategies.spec import PrepareCtx, register_expression_strategy

_OBV_FAST = 20
_DELIVERY_SMA_WINDOW = 20
_DELIVERY_MIN_PERIODS = 10


def _prepare_delivery_accumulation(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    from screener.rs_breakout import india_symbol
    from screener.unusual_volume.delivery import load_delivery_panel

    panel = pd.DataFrame()
    if ctx.market == "india":
        history_days = max(
            (pd.Timestamp(ctx.end) - pd.Timestamp(ctx.start)).days + 30, 60
        )
        try:
            panel = load_delivery_panel(
                [india_symbol(symbol) for symbol in ctx.tv_symbols],
                ctx.end,
                history_days=history_days,
            )
        except (
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            ctx.warnings.append(
                f"delivery panel unavailable for delivery_accumulation: {exc}"
            )

    delivery_by_sym: dict[str, pd.DataFrame] = {}
    if panel is not None and not panel.empty:
        panel = panel.copy()
        panel["SYMBOL"] = panel["SYMBOL"].astype(str).str.upper()
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
        for sym, grp in panel.groupby("SYMBOL"):
            grp = (
                grp.dropna(subset=["date"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
            )
            if grp.empty:
                continue
            pct = pd.to_numeric(grp["DELIV_PER"], errors="coerce")
            pct.index = pd.DatetimeIndex(grp["date"])
            # Rising delivery = today's DELIV_PER above its prior 20-day norm
            # (the current bar is excluded from the baseline, like volume rvol).
            sma = (
                pct.shift(1)
                .rolling(_DELIVERY_SMA_WINDOW, min_periods=_DELIVERY_MIN_PERIODS)
                .mean()
            )
            delivery_by_sym[sym] = pd.DataFrame(
                {"delivery_pct": pct, "delivery_sma_20": sma}
            )

    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        volume = frame["volume"].astype(float)
        cum_pv = (typical * volume).cumsum()
        cum_v = volume.cumsum()
        frame["vwap"] = cum_pv / cum_v.where(cum_v > 0)
        frame["obv"] = on_balance_volume(frame["close"], volume)
        frame["obv_ma20"] = (
            frame["obv"].rolling(_OBV_FAST, min_periods=_OBV_FAST).mean()
        )

        # Delivery columns are always present (NaN-filled when inactive) so the
        # Pine expression evaluates uniformly across the universe.
        frame["delivery_active"] = False
        frame["delivery_pct"] = np.nan
        frame["delivery_sma_20"] = np.nan
        if delivery_by_sym:
            rows = delivery_by_sym.get(india_symbol(tv))
            if rows is not None and not rows.empty:
                frame["delivery_active"] = True
                frame["delivery_pct"] = (
                    rows["delivery_pct"].reindex(frame.index).to_numpy(dtype=float)
                )
                frame["delivery_sma_20"] = (
                    rows["delivery_sma_20"].reindex(frame.index).to_numpy(dtype=float)
                )
        out[tv] = frame
    return out


def _lookback() -> int:
    # OBV fast average needs 20 bars; delivery SMA needs 20 prior delivery days.
    return max(_OBV_FAST, _DELIVERY_SMA_WINDOW)


register_expression_strategy(
    "delivery_accumulation",
    entry=(
        "close > vwap and obv > obv_ma20 "
        "and (not delivery_active or delivery_pct > delivery_sma_20)"
    ),
    exit="crossunder(close, vwap)",
    prepare_bars=_prepare_delivery_accumulation,
    required_lookback=_lookback,
)
