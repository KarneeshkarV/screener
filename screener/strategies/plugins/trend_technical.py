"""Advanced trend, time-series momentum and technical-regime strategies.

Four pure-price strategies spanning the "trend timing / time-series momentum"
literature. None of them needs fundamentals, so they run on any universe
(US SP500 or India Nifty500) with no ``--fundamental-field`` and no FMP
dependency; selection among eligible names uses the ``rank_score`` column each
``prepare_bars`` hook emits, so the rolling backtester fills its ``--top``
slots with the strongest names under the strategy's own cross-sectional
ranking.

Methodology sources:

- ``tsmom_12_1``        — Moskowitz, Ooi & Pedersen (2012), "Time Series
                          Momentum", Journal of Financial Economics 104(2):
                          228-250. The 12-month *own* return (skipping the most
                          recent month) predicts the next period's return, and
                          the SIGN of the 12-month return is the strongest
                          predictor; persistence lasts roughly a year. MOP's
                          big Sharpe gain comes from scaling positions by
                          inverse ex-ante volatility — the repo's
                          ``--sizing inverse_vol --sizing-risk-pct 0.1``
                          implements exactly that leg, so this strategy is
                          designed to be run with it. The exit mirrors MOP's
                          signal-reversal rule: leave when the 12-1 own return
                          turns negative (the 126-day ``--hold`` is the fallback
                          for slots that never see a sign flip).
- ``kama_trend``        — Kaufman (1995), "Trading Systems and Methods" (3rd
                          ed., Wiley): the Kaufman Adaptive Moving Average. The
                          efficiency ratio ER = |net change| / sum(|changes|)
                          over n bars separates trending from choppy regimes;
                          KAMA is an EMA whose smoothing constant is ER-weighted
                          (fast in trends, slow in chop). This strategy gates
                          entries on a high ER (strong trend) plus price above
                          KAMA, ranks names by ER (smoothest persistent trend
                          first) and exits on a close below KAMA (trend break).
- ``hurst_trend_quality`` — Hurst (1951), "Long-Term Storage Capacity of
                          Reservoirs", Trans. ASCE 116; Mandelbrot & Van Ness
                          (1968), "Fractional Brownian Motions, Fractional
                          Noises and Applications", SIAM Review 10(4); Lo &
                          MacKinlay (1988), "Stock Market Prices Do Not Follow
                          Random Walks: Evidence from a Simple Specification
                          Test", Review of Financial Studies 1(1). H > 0.5 means
                          a persistently trending (long-memory) price path,
                          H < 0.5 means mean reversion. The Hurst exponent is
                          estimated here with the variance-ratio proxy
                          (VR(q) = Var(q-day returns) / (q * Var(1-day)),
                          H = 0.5 + log(VR) / (2 log q)) — Lo-MacKinlay's
                          specification statistic — so only persistently
                          trending names (H > 0.55) with a positive 12-month
                          trend are bought; the book exits when persistence
                          breaks (H < 0.5).
- ``ma_timing_200``     — Zakamulin (2017), "Market Timing with Moving
                          Averages: The Anatomy and Performance of Technical
                          Rules" (Palgrave Macmillan): the meta-analytic
                          evidence that long-term MA timing (the 200-day rule)
                          beats short-term rules, and that "momentum-stop
                          hybrids" (trend timing + momentum selection +
                          mechanical stops) are the best performers; and Odean
                          (1998), "Are Investors Reluctant to Realize Their
                          Losses?", Journal of Finance 53(5):1775-1798 — the
                          disposition effect: investors sell winners too early
                          and hold losers too long. The exit below is the
                          mechanical loser cut: a crossunder of the 200-day MA
                          closes the position by rule, not by hope.

All signals are causal (bar ``t`` uses only data ``<= t``). Recommended holds:
126 trading days for the 6-12 month momentum family, 63 for the faster KAMA
book, 250 for the 200-day MA timing book. Wilder's ADX trend-strength gate
(ADX(14) > 25, +DI > -DI) is already registered as ``adx_trend``; this module
instead covers the time-series-momentum, long-memory and adaptive-MA branches
of the trend family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Shared trading-day windows ─────────────────────────────────────────
_LOOKBACK_12M = 252  # ~12 months
_SKIP_1M = 21  # ~1 month (short-term-reversal skip)
_LOOKBACK_6M = 126  # ~6 months
_SMA_LONG = 200  # long-term trend MA


def _momentum(close: pd.Series, lookback: int, skip: int = _SKIP_1M) -> pd.Series:
    """Causal own-price momentum: close[t-skip] / close[t-lookback] - 1."""
    close = close.astype(float)
    return close.shift(skip) / close.shift(lookback) - 1.0


# ── 1. Time-series momentum (Moskowitz, Ooi & Pedersen 2012) ────────────

TSMOM_ENTRY = "mom_12_1 > 0"
TSMOM_EXIT = "mom_12_1 < 0"


def _prepare_tsmom(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the 12-1 own return and rank names by it (MOP predictor)."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        mom = _momentum(frame["close"], _LOOKBACK_12M)
        frame["mom_12_1"] = mom
        frame["rank_score"] = mom
        out[tv] = frame
    return out


def _lookback_tsmom() -> int:
    # The oldest leg of the 12-1 ratio needs 252 prior closes.
    return _LOOKBACK_12M


# ── 2. Kaufman Adaptive Moving Average (Kaufman 1995) ───────────────────

_KAMA_N = 10
_KAMA_FAST = 2
_KAMA_SLOW = 30
_ER_MIN = 0.35

KAMA_ENTRY = f"er_10 > {_ER_MIN} and close > kama_10"
KAMA_EXIT = "close < kama_10"


def _efficiency_ratio(close: pd.Series, n: int = _KAMA_N) -> pd.Series:
    """Kaufman efficiency ratio over n bars: |net change| / sum(|changes|).

    Near 1 on a straight-line trend, near 0 in pure chop. Causal.
    """
    close = close.astype(float)
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n, min_periods=n).sum()
    return (change / volatility).clip(lower=0.0, upper=1.0)


def _kama(
    close: pd.Series,
    n: int = _KAMA_N,
    fast: int = _KAMA_FAST,
    slow: int = _KAMA_SLOW,
) -> np.ndarray:
    """Kaufman Adaptive Moving Average, seeded with an n-bar SMA.

    alpha = (ER * (sc_fast - sc_slow) + sc_slow)^2 with sc = 2/(period+1):
    ER near 1 speeds the average toward ``fast``, ER near 0 slows it toward
    ``slow``. Causal (value at ``t`` uses data <= ``t``).
    """
    close_arr = close.astype(float).to_numpy(dtype=float)
    er = _efficiency_ratio(close, n).to_numpy(dtype=float)
    sc_fast = 2.0 / (fast + 1.0)
    sc_slow = 2.0 / (slow + 1.0)
    alpha = np.power(er * (sc_fast - sc_slow) + sc_slow, 2.0)
    kama = np.full(len(close_arr), np.nan, dtype=np.float64)
    if len(close_arr) < n:
        return kama
    kama[n - 1] = np.nanmean(close_arr[:n])
    for i in range(n, len(close_arr)):
        kama[i] = kama[i - 1] + alpha[i] * (close_arr[i] - kama[i - 1])
    return kama


def _prepare_kama(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the efficiency ratio + KAMA and rank by trend smoothness."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["er_10"] = _efficiency_ratio(close)
        frame["kama_10"] = _kama(close)
        frame["rank_score"] = frame["er_10"]
        out[tv] = frame
    return out


def _lookback_kama() -> int:
    # Efficiency ratio needs n+1 closes; the KAMA seed needs n. 20 covers both
    # with margin for the backtester's warmup arithmetic.
    return 20


# ── 3. Hurst trend-quality filter (Hurst 1951 / Lo-MacKinlay 1988) ──────

_HURST_WINDOW = 126  # trailing bars over which the variance ratio is estimated
_VR_Q = 5  # q-day overlapping return used by the variance ratio
_HURST_ENTRY_MIN = 0.55
_HURST_EXIT_MIN = 0.50

HURST_ENTRY = f"hurst_126 > {_HURST_ENTRY_MIN} and mom_12_1 > 0"
HURST_EXIT = f"hurst_126 < {_HURST_EXIT_MIN}"


def _hurst_variance_ratio(
    close: pd.Series,
    window: int = _HURST_WINDOW,
    q: int = _VR_Q,
) -> pd.Series:
    """Hurst exponent via the Lo-MacKinlay variance-ratio proxy.

    For a path with Hurst H, Var(q-day returns) ~ q^(2H) * Var(1-day returns),
    so H = 0.5 + log(VR(q)) / (2 log q) with VR(q) = Var_q / (q * Var_1).
    Rolling (overlapping) sample variances over the trailing ``window`` bars;
    NaN where history is short or the 1-day variance is zero (flat prices).
    H > 0.5 -> persistent/trending; H < 0.5 -> mean-reverting. Causal.
    """
    close = close.astype(float)
    ret1 = close.pct_change()
    retq = close / close.shift(q) - 1.0
    var1 = ret1.rolling(window, min_periods=window).var()
    varq = retq.rolling(window, min_periods=window).var()
    # Compute on the numpy mirror so mypy keeps the return typed (np.log on a
    # Series is ``Any`` under pandas-stubs); NaN rows stay NaN through the log.
    vr_arr = varq.to_numpy(dtype=float) / (
        q * var1.where(var1 > 0).to_numpy(dtype=float)
    )
    hurst_arr = 0.5 + np.log(vr_arr) / (2.0 * np.log(q))
    return pd.Series(np.clip(hurst_arr, 0.0, 1.0), index=var1.index)


def _prepare_hurst(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach the Hurst estimate + 12-1 momentum; rank by quality x strength."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        mom = _momentum(close, _LOOKBACK_12M)
        frame["mom_12_1"] = mom
        frame["hurst_126"] = _hurst_variance_ratio(close)
        # Trend quality times trend strength: persistent AND strong uptrends
        # rank first (both legs are positive under the entry gate).
        frame["rank_score"] = frame["hurst_126"] * mom
        out[tv] = frame
    return out


def _lookback_hurst() -> int:
    # mom_12_1 needs 252 prior closes; the VR window (126 returns + q) is shorter.
    return _LOOKBACK_12M


# ── 4. 200-day MA timing with momentum-stop (Zakamulin 2017; Odean 1998) ─

MA_TIMING_ENTRY = f"close > sma(close, {_SMA_LONG}) and mom_6_1 > 0"
MA_TIMING_EXIT = f"crossunder(close, sma(close, {_SMA_LONG}))"


def _prepare_ma_timing(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Attach 6-1 momentum and rank by it within the long-term uptrend."""
    out: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            out[tv] = bars
            continue
        frame = bars.copy()
        mom = _momentum(frame["close"], _LOOKBACK_6M)
        frame["mom_6_1"] = mom
        frame["rank_score"] = mom
        out[tv] = frame
    return out


def _lookback_ma_timing() -> int:
    # sma(close, 200) in the entry is the long pole; mom_6_1 (126) is shorter.
    return _SMA_LONG


# ── Registrations ───────────────────────────────────────────────────────

register_expression_strategy(
    "tsmom_12_1",
    entry=TSMOM_ENTRY,
    exit=TSMOM_EXIT,
    prepare_bars=_prepare_tsmom,
    required_lookback=_lookback_tsmom,
)

register_expression_strategy(
    "kama_trend",
    entry=KAMA_ENTRY,
    exit=KAMA_EXIT,
    prepare_bars=_prepare_kama,
    required_lookback=_lookback_kama,
)

register_expression_strategy(
    "hurst_trend_quality",
    entry=HURST_ENTRY,
    exit=HURST_EXIT,
    prepare_bars=_prepare_hurst,
    required_lookback=_lookback_hurst,
)

register_expression_strategy(
    "ma_timing_200",
    entry=MA_TIMING_ENTRY,
    exit=MA_TIMING_EXIT,
    prepare_bars=_prepare_ma_timing,
    required_lookback=_lookback_ma_timing,
)
