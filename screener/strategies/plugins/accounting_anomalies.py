"""Accounting anomalies & fundamental quality: accruals, F-Score, gross profitability.

Methodology sources (the canonical accounting-anomaly / quality literature):

- ``sloan_low_accruals`` — Sloan (1996) "Do Stock Prices Fully Reflect
  Information in Accruals and Cash Flows About Future Earnings?", *The
  Accounting Review* 71(3), 289-315. Earnings made up largely of accruals are
  less persistent than earnings backed by cash flow, yet investors over-extrapolate
  them, so low-accrual firms subsequently outperform high-accrual firms (the
  accrual anomaly; Xie 2001 shows abnormal accruals drive it). We hold profitable
  names (pe_ttm > 0, roe_ttm > 0) with positive operating cash flow and low
  Sloan accruals (accruals <= 2% of average assets; negative is strongest).
  Fundamentals merge *after* the prepare hook, so the accrual gate lives in the
  entry expression and selection among the passing names uses dollar volume.
- ``piotroski_value`` — Piotroski (2000) "Value Investing: The Use of Historical
  Financial Statement Information to Separate Winners from Losers", *Journal of
  Accounting Research* 38 (Supplement), 1-41. Apply the 9-signal F-Score inside
  the high book-to-market (cheap) universe: cheap names with F >= 7 (the
  paper's high-F group) are high-quality value, while cheap names with low F are
  value traps. Gate: pe_ttm in (0, 20], pb_ttm in (0, 3] and piotroski_fscore
  >= 7 (NaN F-Score fails the gate, so names with missing statement data are
  excluded - fail-closed).
- ``gross_profitability`` — Novy-Marx (2013) "The Other Side of Value: The Gross
  Profitability Premium", *Journal of Financial Economics* 108(1), 1-28. Gross
  profit / assets is the profitability measure that best predicts returns, and
  profitability is complementary to momentum ("profitable firms, and firms with
  high momentum, earn particularly large excess returns"). We gate on
  gross_profit_to_assets >= 0.25 with positive earnings and low leverage, and
  rank the passing names by trailing 6-month return percentile (price-only, so
  it can live in ``rank_score`` via prepare_bars) - harvesting momentum *within*
  the profitable universe, as the paper prescribes.
- ``conservative_investment`` — Fama & French (2015) "A Five-Factor Asset
  Pricing Model", *Journal of Financial Economics* 116(1), 1-22: the investment
  factor (CMA). Low asset-growth (conservative) firms earn higher returns than
  high-growth (aggressive) firms (also Cooper, Gulen & Schill 2008, asset
  growth anomaly). We require asset_growth <= 8% yoy inside a value screen, and
  add the Altman (1968) *Journal of Finance* 23(4), 589-609 Z-score >= 1.8
  (above the distress zone) plus a 200-day trend gate so deep-value names that
  are about to fail - or already falling - are skipped (value-trap avoidance).

All four are market-agnostic expressions (FMP serves both US and India .NS
symbols); run with ``--fundamentals-provider fmp`` and the per-strategy
``--fundamental-field`` list (see module docstring of ``value_garp`` for the
merge/lag conventions). NaN fundamentals fail every comparison (the evaluator
fills comparison results with False), so missing statement data excludes the
name - the intended fail-closed behaviour.
"""

from __future__ import annotations

import pandas as pd

from screener.strategies.spec import PrepareCtx, register_expression_strategy

# ── Shared, deliberately moderate gates (Nifty500/SP500-friendly) ──────────
_PE_MAX = 20.0
_PB_MAX = 3.0
_DEBT_MAX = 2.0
_TREND_SMA = 200
_MOM_WINDOW = 126  # ~6 months, Novy-Marx momentum complementarity leg

# Sloan (1996): low-accrual, cash-backed earnings. accruals <= 2% of avg assets
# (fraction, e.g. 0.014); negative accruals are strongest but 0.02 keeps a
# usable cross-section. operating_cash_flow is raw currency, so > 0 means the
# cash leg is genuinely positive; NaN (missing statement) fails closed.
SLOAN_ENTRY = (
    "pe_ttm > 0 and roe_ttm > 0 and operating_cash_flow > 0 and accruals <= 0.02"
)

# Piotroski (2000): F-Score applied INSIDE the cheap universe. pe/pb positive
# (real earnings/book), moderate multiples, F >= 7 (high-F group of the paper).
PIOTROSKI_ENTRY = (
    f"pe_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm > 0 and pb_ttm <= {_PB_MAX} "
    "and piotroski_fscore >= 7"
)

# Novy-Marx (2013): gross profit / assets >= 25% of assets, positive earnings,
# low leverage. Selection ranks the profitable names by 6-month momentum
# (prepare_bars below) - profitability + momentum are complementary.
GROSS_PROFIT_ENTRY = (
    f"pe_ttm > 0 and debt_to_equity <= {_DEBT_MAX} and gross_profit_to_assets >= 0.25"
)

# Fama-French (2015) CMA + Altman (1968): conservative investment (asset_growth
# <= 8% yoy) inside a value screen, with the bankruptcy Z-score above the
# distress zone (z_score >= 1.8) and a 200-day trend gate as extra
# value-trap insurance.
CONSERVATIVE_ENTRY = (
    f"pe_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm > 0 and pb_ttm <= {_PB_MAX} "
    f"and asset_growth <= 8 and z_score >= 1.8 and close > sma(close, {_TREND_SMA})"
)


def _prepare_gross_profit_momentum(ctx: PrepareCtx) -> dict[str, pd.DataFrame]:
    """Rank gross-profitable names by trailing 6-month return (price-only).

    Fundamentals merge after this hook, so ``rank_score`` can only use price
    data: among the names passing the gross-profitability gate in the entry
    expression, the rolling backtester fills its ``--top`` slots with the
    strongest trailing-6-month performers (causal: bar ``t`` uses closes
    <= ``t``). Cross-sectional percentile per day over the full prepared
    universe, same convention as ``nifty_momentum``/``mom_lowvol_combo``.
    """
    ratios: dict[str, pd.DataFrame] = {}
    for tv, bars in ctx.bars_by_tv.items():
        if bars is None or bars.empty:
            continue
        frame = bars.copy()
        close = frame["close"].astype(float)
        frame["mom_126"] = close / close.shift(_MOM_WINDOW) - 1.0
        ratios[tv] = frame

    if not ratios:
        return ctx.bars_by_tv

    mom = pd.DataFrame({tv: frame["mom_126"] for tv, frame in ratios.items()})
    mom_pct = mom.rank(axis=1, pct=True)

    out: dict[str, pd.DataFrame] = {}
    for tv, frame in ratios.items():
        frame["rank_score"] = mom_pct[tv].reindex(frame.index)
        out[tv] = frame
    return out


def _lookback_basic() -> int:
    # No rolling windows; just enough history for the merge + entry eval.
    return 20


def _lookback_momentum() -> int:
    # 6-month momentum rank leg needs 126 prior closes.
    return _MOM_WINDOW


def _lookback_trend() -> int:
    # ``close > sma(close, 200)`` needs 200 trailing closes.
    return _TREND_SMA


#: Relaxed F-Score floor for thin-coverage windows (India 1y/2y trade counts).
PIOTROSKI_ENTRY_F6 = (
    f"pe_ttm > 0 and pe_ttm <= {_PE_MAX} and pb_ttm > 0 and pb_ttm <= {_PB_MAX} "
    "and piotroski_fscore >= 6"
)

register_expression_strategy(
    "sloan_low_accruals",
    entry=SLOAN_ENTRY,
    exit=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "piotroski_value_f6",
    entry=PIOTROSKI_ENTRY_F6,
    exit=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "piotroski_value",
    entry=PIOTROSKI_ENTRY,
    exit=None,
    required_lookback=_lookback_basic,
)

register_expression_strategy(
    "gross_profitability",
    entry=GROSS_PROFIT_ENTRY,
    exit=None,
    prepare_bars=_prepare_gross_profit_momentum,
    required_lookback=_lookback_momentum,
)

register_expression_strategy(
    "conservative_investment",
    entry=CONSERVATIVE_ENTRY,
    exit=None,
    required_lookback=_lookback_trend,
)
