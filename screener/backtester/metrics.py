"""Backtest performance metrics.

All metrics derive from the portfolio equity curve and the aligned benchmark.
Alpha/beta use a simple OLS fit via ``numpy.polyfit`` — no sklearn.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd

from screener.backtester.models import BacktestConfig, BacktestResult, Trade
from screener.format import fmt_money, fmt_pct, is_missing
from screener.regime import classify_regimes

TRADING_DAYS_PER_YEAR = 252
_EULER_MASCHERONI = 0.5772156649015329

# Regular-session bars per trading day for each supported interval (US markets:
# a 6.5h session). Used to scale the 252-trading-day annualization factor so
# intraday equity curves annualize over the right number of periods. Daily is
# 1 bar/day, reproducing the legacy 252 factor exactly. India sessions are
# shorter (~6.25h), so India intraday annualization is slightly off; no behavior
# change intended.
_BARS_PER_SESSION = {"1d": 1, "1h": 7, "30m": 13, "15m": 26, "5m": 78, "1m": 390}


MetricKind = Literal["pct", "money", "ratio", "count", "int"]


@dataclass(frozen=True)
class ResultViewRow:
    """One display-ready metric from a backtest result."""

    key: str
    label: str
    kind: MetricKind
    value: Any
    formatted: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe representation used by browser renderers."""
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "value": self.value,
            "formatted": self.formatted,
        }


@dataclass(frozen=True)
class ResultView(Sequence[ResultViewRow]):
    """Ordered display model shared by every backtest result renderer."""

    rows: tuple[ResultViewRow, ...]

    def __len__(self) -> int:
        return len(self.rows)

    @overload
    def __getitem__(self, index: int) -> ResultViewRow: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ResultViewRow, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ResultViewRow | tuple[ResultViewRow, ...]:
        return self.rows[index]


# The only ordered metric schema. Unknown metrics are deliberately appended by
# ``result_view`` so adding a computed metric needs no renderer change.
_RESULT_VIEW_ORDER: tuple[tuple[str, str, MetricKind], ...] = (
    ("starting_equity", "Starting Capital", "money"),
    ("final_equity", "Final Equity", "money"),
    ("total_return", "Total Return", "pct"),
    ("invested_return", "Invested Return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("vol_annual", "Volatility (ann.)", "pct"),
    ("sharpe", "Sharpe", "ratio"),
    ("sortino", "Sortino", "ratio"),
    ("calmar", "Calmar", "ratio"),
    ("psr", "Probabilistic Sharpe", "pct"),
    ("dsr", "Deflated Sharpe", "pct"),
    ("max_drawdown", "Max Drawdown", "pct"),
    ("hit_rate", "Hit Rate", "pct"),
    ("alpha_annual", "Alpha (ann.)", "pct"),
    ("beta", "Beta", "ratio"),
    ("exposure", "Avg Exposure", "pct"),
    ("benchmark_return", "Benchmark Return", "pct"),
    ("trade_count", "Trades", "count"),
    ("unique_tickers", "Unique Tickers", "count"),
    ("median_trade_return", "Median Trade Return", "pct"),
    ("avg_trade_return", "Avg Trade Return", "pct"),
    ("best_trade_return", "Best Trade", "pct"),
    ("worst_trade_return", "Worst Trade", "pct"),
    ("profit_factor", "Profit Factor", "ratio"),
    ("expectancy", "Expectancy", "pct"),
    ("winning_trades", "Winning Trades", "count"),
    ("losing_trades", "Losing Trades", "count"),
)
_RESULT_VIEW_SPECS = {key: (label, kind) for key, label, kind in _RESULT_VIEW_ORDER}

SIZING_COMPARISON_COLUMNS = ("Fixed slots", "Reinvested slots")


def sizing_comparison_rows(
    fixed: Mapping[str, Any],
    reinvested: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Return ``(label, *cells)`` per metric, one cell per sizing rule.

    Every metric either run computed appears, ordered and labelled by
    ``result_view``, so a compared run never shows a narrower picture than a
    single-rule run does and a metric added to the schema reaches both columns
    with no renderer edit. A metric only one rule produces still gets a row,
    with ``-`` in the column that lacks it.

    Each row is as wide as ``SIZING_COMPARISON_COLUMNS`` plus its label, so a
    renderer can lay the body out from the row itself rather than naming the
    two columns positionally.
    """
    columns = (fixed, reinvested)
    merged: dict[str, Any] = {}
    for metrics in columns:
        merged.update(metrics)
    return tuple(
        (
            row.label,
            *(
                format_result_value(metrics.get(row.key), row.kind)
                for metrics in columns
            ),
        )
        for row in result_view(merged)
    )


def format_result_value(value: Any, kind: MetricKind) -> str:
    """Format a result metric once, independently of its output surface."""
    if is_missing(value):
        return "-"
    if kind == "pct":
        return fmt_pct(float(value) * 100)
    if kind == "money":
        return fmt_money(value)
    if kind == "count":
        return f"{int(value):,}"
    if kind == "int":
        return str(int(value))
    if isinstance(value, (float, int)):
        return f"{float(value):+.3f}"
    return str(value)


def _default_metric_spec(key: str, value: Any) -> tuple[str, MetricKind]:
    """Give newly-computed metrics a useful view without renderer edits."""
    label = key.replace("_", " ").title()
    if key == "risk_adjusted_return":
        return label, "ratio"
    if (
        key.endswith("_return")
        or key.endswith("_rate")
        or key.endswith("_pct")
        or "_pct_" in key
        or key in {"cagr", "vol_annual", "max_drawdown", "exposure"}
    ):
        return label, "pct"
    if key.endswith("_equity") or key == "total_fees" or key.startswith("fee_"):
        return label, "money"
    if key.endswith("_count") or key.endswith("_trades") or key.endswith("_tickers"):
        return label, "count"
    if isinstance(value, int) and not isinstance(value, bool):
        return label, "int"
    return label, "ratio"


def result_view(metrics: Mapping[str, Any]) -> ResultView:
    """Return every metric in a stable order with labels, kinds, and formatting."""
    ordered_keys = [key for key, _, _ in _RESULT_VIEW_ORDER if key in metrics]
    ordered_keys.extend(key for key in metrics if key not in _RESULT_VIEW_SPECS)
    rows = []
    for key in ordered_keys:
        value = metrics[key]
        label, kind = _RESULT_VIEW_SPECS.get(key, _default_metric_spec(key, value))
        rows.append(
            ResultViewRow(key, label, kind, value, format_result_value(value, kind))
        )
    return ResultView(tuple(rows))


def result_view_columns(metric_sets: Iterable[Mapping[str, Any]]) -> ResultView:
    """Return the ordered union of metrics for column-oriented result tables."""
    keys: dict[str, None] = {}
    for metrics in metric_sets:
        keys.update(dict.fromkeys(metrics))
    return result_view({key: 0 for key in keys})


def periods_per_year_for_interval(interval: str) -> int:
    """Return the annualization period count for ``interval``.

    ``1d`` -> 252 (unchanged); intraday intervals scale 252 by the number of
    regular-session bars per day so Sharpe/vol/CAGR annualize consistently.
    """
    return TRADING_DAYS_PER_YEAR * _BARS_PER_SESSION.get(interval, 1)


def _daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def _cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    # Annualize over the elapsed return periods (N-1 for an N-point curve), not
    # the point count: an N-bar equity curve spans N-1 return periods, so the
    # horizon is (N-1)/periods_per_year years. Using len(equity)/... overstated
    # the horizon by one bar and understated CAGR; matches empyrical's convention.
    years = max((len(equity) - 1) / periods_per_year, 1e-9)
    return float((end / start) ** (1.0 / years) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) if not dd.empty else 0.0


def equity_curve_sharpe(
    daily: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if daily.empty or daily.std(ddof=0) == 0:
        return 0.0
    excess = daily - rf / periods_per_year
    return float(excess.mean() / excess.std(ddof=0) * np.sqrt(periods_per_year))


def _vol_annual(
    daily: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    if daily.empty:
        return 0.0
    return float(daily.std(ddof=0) * np.sqrt(periods_per_year))


def _alpha_beta(
    daily: pd.Series,
    bench_daily: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    if daily.empty or bench_daily.empty:
        return 0.0, 0.0
    aligned = pd.concat([daily, bench_daily], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0, 0.0
    x = aligned.iloc[:, 1].to_numpy()
    y = aligned.iloc[:, 0].to_numpy()
    if x.std() == 0:
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    # annualize alpha geometrically (intercept is per-period): (1+a)^ppy - 1.
    # Matches empyrical/quantstats; arithmetic intercept*ppy ignored compounding.
    if intercept == -1.0:
        annualized_alpha = -1.0
    elif intercept < -1.0:
        annualized_alpha = float("nan")
    else:
        log_growth = periods_per_year * np.log1p(intercept)
        max_log = np.log(np.finfo(float).max)
        annualized_alpha = (
            float("inf") if log_growth > max_log else float(np.expm1(log_growth))
        )
    return annualized_alpha, float(slope)


def _exposure(
    equity_index: pd.DatetimeIndex, trades: Iterable[Trade], slot_count: int
) -> float:
    trades = list(trades)
    if not trades or len(equity_index) == 0:
        return 0.0
    # Convert inclusive trade intervals into +1/-1 events, then scan once.
    # ``left`` for entries and ``right`` for exits exactly preserve the prior
    # ``entry <= session <= exit`` mask semantics, including off-calendar dates.
    changes = np.zeros(len(equity_index) + 1, dtype=np.int64)
    entries = pd.DatetimeIndex([pd.Timestamp(t.entry_date) for t in trades])
    exits = pd.DatetimeIndex([pd.Timestamp(t.exit_date) for t in trades])
    starts = equity_index.searchsorted(entries, side="left")
    stops = equity_index.searchsorted(exits, side="right")
    valid = (starts < len(equity_index)) & (stops > 0)
    np.add.at(changes, starts[valid], 1)
    np.add.at(changes, np.minimum(stops[valid], len(equity_index)), -1)
    open_count = np.cumsum(changes[:-1])
    return float(open_count.mean() / max(slot_count, 1))


def _sortino(
    daily: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if daily.empty:
        return 0.0
    excess = daily - rf / periods_per_year
    # Canonical target-downside-deviation: RMS of min(excess, 0) over ALL N
    # periods (target = per-period rf). Matches empyrical.sortino_ratio.
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside_dev == 0:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def _calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    mdd = _max_drawdown(equity)
    if mdd >= 0:
        return 0.0
    return float(_cagr(equity, periods_per_year) / abs(mdd))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Standard-normal inverse CDF via bisection on erf. Good to ~1e-12."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _psr(
    daily: pd.Series,
    sr_benchmark_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Probabilistic Sharpe Ratio (López de Prado, 2012).

    Probability that the *true* annualized Sharpe exceeds ``sr_benchmark_annual``,
    corrected for sample size and non-normality (skew, excess kurtosis). Returns
    a value in [0, 1].
    """
    if daily.empty or len(daily) < 30:
        return 0.0
    T = len(daily)
    sr_per = equity_curve_sharpe(daily, periods_per_year=periods_per_year) / math.sqrt(
        periods_per_year
    )
    sr_bench_per = sr_benchmark_annual / math.sqrt(periods_per_year)
    skew = float(cast(Any, daily.skew())) if daily.std(ddof=0) else 0.0
    kurt_excess = float(cast(Any, daily.kurt())) if daily.std(ddof=0) else 0.0
    denom_sq = 1.0 - skew * sr_per + (kurt_excess / 4.0) * sr_per * sr_per
    denom = math.sqrt(max(denom_sq, 1e-12))
    z = (sr_per - sr_bench_per) * math.sqrt(max(T - 1, 1)) / denom
    return _phi(z)


def _dsr(
    daily: pd.Series,
    n_trials: int = 1,
    sr_trial_std_annual: float = 0.5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Deflated Sharpe Ratio (López de Prado, 2014).

    Like PSR, but the benchmark Sharpe is the *expected maximum* across
    ``n_trials`` independent strategies under the null — i.e. the bar a random
    strategy would clear just from multiple-testing luck. ``sr_trial_std_annual``
    is the cross-trial std of annualized Sharpes (0.5 is a reasonable default
    for equity strategies); pass the measured value if you have it.
    """
    if n_trials <= 1:
        return _psr(daily, 0.0, periods_per_year=periods_per_year)
    sr0_annual = sr_trial_std_annual * (
        (1.0 - _EULER_MASCHERONI) * _phi_inv(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * _phi_inv(1.0 - 1.0 / (n_trials * math.e))
    )
    return _psr(
        daily, sr_benchmark_annual=sr0_annual, periods_per_year=periods_per_year
    )


def _invested_return(trades: Iterable[Trade]) -> float:
    """Capital-deployed-only total return.

    Ignores idle cash in the denominator: sums realized PnL across all closed
    trades and divides by total capital that actually touched the market
    (sum of entry_cost). Exposes the dead-cash gap even when the engine's
    reinvestment path is on — a low ratio of equity_return to invested_return
    indicates a large share of capital sat idle.
    """
    trades = list(trades)
    total_cost = sum(float(t.entry_cost) for t in trades)
    total_pnl = sum(float(t.pnl) for t in trades)
    if total_cost <= 0:
        return 0.0
    return total_pnl / total_cost


def positive_pnl_rate(trades: Iterable[Trade]) -> float:
    """Return the fractional share of accounting trades with positive cash PnL."""
    trades = list(trades)
    return (
        float(sum(1 for trade in trades if trade.pnl > 0) / len(trades))
        if trades
        else 0.0
    )


def _trade_return_stats(trades: Iterable[Trade]) -> dict[str, float | int]:
    """Return realized trade-level diagnostics."""
    trades = list(trades)
    if not trades:
        return {
            "median_trade_return": 0.0,
            "avg_trade_return": 0.0,
            "best_trade_return": 0.0,
            "worst_trade_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
    returns = [float(t.return_pct) for t in trades]
    profits = [float(t.pnl) for t in trades if t.pnl > 0]
    losses = [abs(float(t.pnl)) for t in trades if t.pnl < 0]
    gross_profit = sum(profits)
    gross_loss = sum(losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    return {
        "median_trade_return": float(np.median(returns)),
        "avg_trade_return": float(np.mean(returns)),
        "best_trade_return": float(max(returns)),
        "worst_trade_return": float(min(returns)),
        "profit_factor": float(profit_factor),
        "expectancy": float(np.mean(returns)),
        "winning_trades": sum(1 for t in trades if t.pnl > 0),
        "losing_trades": sum(1 for t in trades if t.pnl < 0),
    }


def compute_cost_metrics(
    fees_paid: dict[str, float],
    initial_capital: float,
    net_pnl: float,
) -> dict:
    """Fee-attribution metrics from a portfolio's accumulated fee breakdown.

    Returns ``total_fees`` (currency), one ``fee_<component>`` key per charged
    component (e.g. ``fee_brokerage``, ``fee_stt``, ``fee_sec_fee``,
    ``fee_taf``), plus ``fees_pct_capital`` (total fees / initial capital) and
    ``fees_pct_net_pnl`` (total fees / realized net PnL, 0 when flat). All
    ratios are fractions; the caller formats them as percentages.
    """
    total = float(sum(float(v) for v in fees_paid.values()))
    out: dict[str, float] = {"total_fees": total}
    for name, amount in fees_paid.items():
        out[f"fee_{name}"] = float(amount)
    out["fees_pct_capital"] = total / initial_capital if initial_capital else 0.0
    out["fees_pct_net_pnl"] = total / net_pnl if net_pnl else 0.0
    return out


def compute_regime_metrics(benchmark: pd.Series, trades: list[Trade]) -> dict:
    """Per-regime trade stats keyed by each trade's entry-date regime.

    The regime is classified from the (point-in-time) benchmark close curve;
    each trade is bucketed by the benchmark regime in force on its entry date
    (most recent benchmark date at or before entry). Returns keys of the form
    ``regime_<label>_trades`` / ``regime_<label>_win_rate`` /
    ``regime_<label>_avg_return`` for each label with at least one trade.
    """
    if benchmark is None or benchmark.empty or not trades:
        return {}
    regimes = classify_regimes(benchmark)
    idx = regimes.index
    by_label: dict[str, list[float]] = {}
    for t in trades:
        pos = int(idx.searchsorted(pd.Timestamp(t.entry_date), side="right")) - 1
        label = str(regimes.iloc[pos]) if pos >= 0 else "unknown"
        by_label.setdefault(label, []).append(float(t.return_pct))
    out: dict = {}
    for label, rets in by_label.items():
        out[f"regime_{label}_trades"] = len(rets)
        out[f"regime_{label}_win_rate"] = sum(1 for r in rets if r > 0) / len(rets)
        out[f"regime_{label}_avg_return"] = sum(rets) / len(rets)
    return out


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: list[Trade],
    slot_count: int,
    n_trials: int = 1,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    daily = _daily_returns(equity)
    bench_daily = (
        _daily_returns(benchmark) if not benchmark.empty else pd.Series(dtype=float)
    )
    starting_equity = float(equity.iloc[0]) if not equity.empty else 0.0
    final_equity = float(equity.iloc[-1]) if not equity.empty else 0.0
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if len(equity) >= 2 and equity.iloc[0] > 0
        else 0.0
    )
    alpha, beta = _alpha_beta(daily, bench_daily, periods_per_year)
    hit_rate = positive_pnl_rate(trades)
    bench_return = (
        float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        if len(benchmark) >= 2 and benchmark.iloc[0] > 0
        else 0.0
    )
    metrics = {
        "starting_equity": starting_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": _cagr(equity, periods_per_year),
        "vol_annual": _vol_annual(daily, periods_per_year),
        "sharpe": equity_curve_sharpe(daily, periods_per_year=periods_per_year),
        "sortino": _sortino(daily, periods_per_year=periods_per_year),
        "calmar": _calmar(equity, periods_per_year),
        "psr": _psr(daily, sr_benchmark_annual=0.0, periods_per_year=periods_per_year),
        "dsr": _dsr(daily, n_trials=n_trials, periods_per_year=periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": hit_rate,
        "alpha_annual": alpha,
        "beta": beta,
        "exposure": _exposure(cast(pd.DatetimeIndex, equity.index), trades, slot_count),
        "benchmark_return": bench_return,
        "trade_count": len(trades),
        "invested_return": _invested_return(trades),
    }
    metrics.update(_trade_return_stats(trades))
    return metrics


def no_trades_result(
    cfg: BacktestConfig,
    *,
    calendar: pd.Index,
    benchmark: pd.Series,
    warnings: list[str],
    selection: pd.DataFrame | None = None,
) -> BacktestResult:
    """Build the flat, no-trade result every engine falls back to.

    Reached whenever a run never opens a position: no candidate survived
    selection, or the rolling window contained no bars at all. Equity is
    initial capital held flat across ``calendar`` and the benchmark is
    forward-filled onto the same index, so the result still renders and
    compares like any other.
    """
    equity = pd.Series(cfg.initial_capital, index=calendar, dtype=float)
    benchmark_aligned = benchmark.reindex(calendar, method="ffill").dropna()
    metrics = compute_metrics(
        equity,
        benchmark_aligned,
        [],
        max(cfg.top, 1),
        periods_per_year=periods_per_year_for_interval(cfg.interval),
    )
    metrics["unique_tickers"] = 0
    return BacktestResult(
        config=cfg,
        trades=[],
        equity_curve=equity,
        benchmark_curve=benchmark_aligned,
        metrics=metrics,
        warnings=warnings,
        selection=pd.DataFrame() if selection is None else selection,
    )
