"""Standalone validation of screener.backtester against ground-truth math.

Run: python validate_backtest.py

Compares engine-reported metrics against independent pandas/numpy calculations
on the same raw price data, for buy-and-hold on AAPL/MSFT/SPY/QQQ and for the
golden-cross strategy on AAPL.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from screener.backtester import BacktestConfig, run_backtest
from screener.backtester.data import YFinancePriceFetcher
from screener.backtester.strategies import STRATEGIES


TICKERS = ("AAPL", "MSFT", "SPY", "QQQ")
START = date(2015, 1, 1)
END = date(2024, 12, 31)
BENCHMARK = "SPY"
INITIAL_CAPITAL = 100_000.0

RETURN_TOL = 0.005   # absolute, on total_return / cagr / max_dd / vol
SHARPE_TOL = 0.01    # relative (1%), on sharpe

TRADING_DAYS = 252


@dataclass
class CappedFetcher:
    """Wrap a fetcher to clamp the requested end date.

    The engine internally re-fetches with end ≈ as_of + hold*2 + 30 days,
    which for hold=10_000 reaches ~2070 — yfinance returns through today,
    so the engine ends up running through today instead of the requested END.
    Clamping forces the engine's bar calendar to stop at ``end_cap``.
    """
    inner: Any
    end_cap: date

    def fetch(self, tickers: Iterable[str], start: date, end: date):
        capped = min(end, self.end_cap)
        return self.inner.fetch(tickers, start, capped)


# --------------------------------------------------------------------------- #
# buy-and-hold                                                                 #
# --------------------------------------------------------------------------- #

def run_engine_buy_and_hold(ticker: str, fetcher: YFinancePriceFetcher) -> dict[str, Any]:
    cfg = BacktestConfig(
        market="us",
        as_of=START,
        hold=10_000,
        top=1,
        entry_expr="1",
        exit_expr=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=INITIAL_CAPITAL,
        benchmark=BENCHMARK,
        tickers=(ticker,),
        min_price=None,
        min_avg_dollar_volume=None,
    )
    result = run_backtest(cfg, fetcher)
    return {
        "metrics": result.metrics,
        "equity": result.equity_curve,
        "trades": result.trades,
        "warnings": result.warnings,
    }


def compute_truth_metrics(close: pd.Series) -> dict[str, float]:
    """Replicate screener.backtester.metrics formulas on a price (or equity) series."""
    daily = close.pct_change().dropna()
    total_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    years = max(len(close) / TRADING_DAYS, 1e-9)
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)
    peak = close.cummax()
    max_dd = float(((close - peak) / peak).min())
    std = float(daily.std(ddof=0))
    sharpe = 0.0 if std == 0 else float(daily.mean() / std * np.sqrt(TRADING_DAYS))
    vol_annual = float(std * np.sqrt(TRADING_DAYS))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "vol_annual": vol_annual,
    }


def _flag(metric: str, abs_diff: float, pct_diff: float) -> str:
    if metric == "sharpe":
        return "*" if abs(pct_diff) > SHARPE_TOL * 100 else ""
    return "*" if abs(abs_diff) > RETURN_TOL else ""


def compare_buy_and_hold(
    ticker: str, engine_out: dict, fetcher: YFinancePriceFetcher
) -> list[tuple]:
    """Return list of comparison rows: (ticker, metric, engine, t_aligned, t_naive, dabs, dpct, flag)."""
    bars = fetcher.fetch([ticker], START, END)[ticker]
    close = bars["close"].copy()
    close.index = pd.to_datetime(close.index)

    naive = compute_truth_metrics(close)

    eq = engine_out["equity"]
    aligned_close = close.loc[eq.index[0] : eq.index[-1]]
    aligned = compute_truth_metrics(aligned_close)

    em = engine_out["metrics"]
    rows = []
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "vol_annual"):
        e = float(em[k])
        ta = aligned[k]
        tn = naive[k]
        abs_diff = e - ta
        pct_diff = (abs_diff / ta * 100.0) if ta != 0 else float("nan")
        rows.append((ticker, k, e, ta, tn, abs_diff, pct_diff, _flag(k, abs_diff, pct_diff)))
    return rows


# --------------------------------------------------------------------------- #
# golden cross                                                                 #
# --------------------------------------------------------------------------- #

def run_engine_golden_cross(
    ticker: str, fetcher: Any, as_of: date
) -> dict[str, Any]:
    gc = STRATEGIES["golden_cross"]
    cfg = BacktestConfig(
        market="us",
        as_of=as_of,
        hold=10_000,
        top=1,
        entry_expr=gc.entry,
        exit_expr=gc.exit,
        stop_loss=None,
        take_profit=None,
        trailing_stop=None,
        slippage_bps=0.0,
        commission_bps=0.0,
        initial_capital=INITIAL_CAPITAL,
        benchmark=BENCHMARK,
        tickers=(ticker,),
        min_price=None,
        min_avg_dollar_volume=None,
        allow_reentry=True,
        max_reentries=20,
    )
    result = run_backtest(cfg, fetcher)
    return {
        "trades": result.trades,
        "metrics": result.metrics,
        "equity": result.equity_curve,
        "warnings": result.warnings,
    }


def hand_rolled_golden_cross(ticker: str, fetcher: Any) -> dict[str, Any]:
    """Reimplement golden-cross with pandas to mirror engine semantics:
      * entry at NEXT bar's open after sma50 crosses above sma200
      * exit at SAME bar's close on sma50-cross-below (engine.py:517-518)
      * still-open at end → exit at last bar's close (eod)
      * fixed-budget sizing: each trade uses slot_capital=initial_capital/top,
        so realized $PnL = budget * pct_return (NOT compounded) — matches
        portfolio.py:82 ``budget = min(slot_capital, cash)`` for top=1.

    Computes two total-return numbers:
      ``total_return_pnl`` = sum(pnl)/initial_capital
        — answers "return on initial capital"
      ``total_return_engine_match`` = sum(pnl) / (shares0 * close_at_first_entry)
        — mirrors engine's ``equity[-1]/equity[0] - 1`` where
          ``equity[0]`` is the first-entry-day-close MTM of the position.
    """
    bars = fetcher.fetch([ticker], START, END)[ticker].copy()
    bars.index = pd.to_datetime(bars.index)
    close = bars["close"]
    open_ = bars["open"]

    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    cross_up = (sma50 > sma200) & (sma50.shift(1) <= sma200.shift(1))
    cross_dn = (sma50 < sma200) & (sma50.shift(1) >= sma200.shift(1))

    trades: list[dict] = []
    in_pos = False
    entry_date: pd.Timestamp | None = None
    entry_price: float = 0.0
    first_entry_idx: int | None = None
    first_entry_shares: float = 0.0
    n = len(bars)
    for i in range(n):
        if not in_pos and bool(cross_up.iloc[i]) and i + 1 < n:
            entry_date = bars.index[i + 1]
            entry_price = float(open_.iloc[i + 1])
            in_pos = True
            if first_entry_idx is None:
                first_entry_idx = i + 1
                first_entry_shares = INITIAL_CAPITAL / entry_price
        elif in_pos and bool(cross_dn.iloc[i]):
            exit_date = bars.index[i]
            exit_price = float(close.iloc[i])
            trades.append({
                "entry_date": entry_date.date(),
                "entry_price": entry_price,
                "exit_date": exit_date.date(),
                "exit_price": exit_price,
                "ret": exit_price / entry_price - 1.0,
                "exit_reason": "exit_expr",
            })
            in_pos = False

    if in_pos and entry_date is not None:
        exit_date = bars.index[-1]
        exit_price = float(close.iloc[-1])
        trades.append({
            "entry_date": entry_date.date(),
            "entry_price": entry_price,
            "exit_date": exit_date.date(),
            "exit_price": exit_price,
            "ret": exit_price / entry_price - 1.0,
            "exit_reason": "eod",
        })

    total_pnl = sum(INITIAL_CAPITAL * t["ret"] for t in trades)
    total_return_pnl = total_pnl / INITIAL_CAPITAL
    if first_entry_idx is not None:
        eq0_match = first_entry_shares * float(close.iloc[first_entry_idx])
        final_value = INITIAL_CAPITAL + total_pnl
        total_return_engine_match = final_value / eq0_match - 1.0
    else:
        total_return_engine_match = total_return_pnl
    return {
        "trades": trades,
        "total_return": total_return_pnl,
        "total_return_engine_match": total_return_engine_match,
    }


# --------------------------------------------------------------------------- #
# formatting / printing                                                        #
# --------------------------------------------------------------------------- #

def print_section(title: str) -> None:
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def format_buy_and_hold_table(rows: list[tuple]) -> str:
    header = (
        f"{'Ticker':<7}{'Metric':<16}{'Engine':>12}{'Truth(aligned)':>18}"
        f"{'Truth(naive)':>16}{'Δabs':>12}{'Δpct%':>10}  Flag"
    )
    lines = [header, "-" * len(header)]
    for ticker, metric, e, ta, tn, dabs, dpct, flag in rows:
        lines.append(
            f"{ticker:<7}{metric:<16}{e:>12.4f}{ta:>18.4f}{tn:>16.4f}"
            f"{dabs:>12.4f}{dpct:>10.2f}  {flag}"
        )
    return "\n".join(lines)


def format_golden_cross_report(engine_out: dict, hand_out: dict) -> tuple[str, dict]:
    e_trades = engine_out["trades"]
    h_trades = hand_out["trades"]
    lines: list[str] = []
    lines.append(f"Engine trades: {len(e_trades)} | Hand-rolled trades: {len(h_trades)}")
    lines.append("")
    header = (
        f"{'#':>3} | {'Engine entry':>12} {'exit':>12} {'ret%':>8} {'reason':>10} | "
        f"{'Hand entry':>12} {'exit':>12} {'ret%':>8} | Match"
    )
    lines.append(header)
    lines.append("-" * len(header))

    n = max(len(e_trades), len(h_trades))
    matches = 0
    for i in range(n):
        et = e_trades[i] if i < len(e_trades) else None
        ht = h_trades[i] if i < len(h_trades) else None
        if et is not None:
            e_entry = et.entry_date.isoformat()
            e_exit = et.exit_date.isoformat()
            e_ret = (et.exit_price / et.entry_price - 1.0) * 100.0
            e_reason = et.exit_reason
        else:
            e_entry = e_exit = "-"
            e_ret = float("nan")
            e_reason = "-"
        if ht is not None:
            h_entry = ht["entry_date"].isoformat()
            h_exit = ht["exit_date"].isoformat()
            h_ret = ht["ret"] * 100.0
        else:
            h_entry = h_exit = "-"
            h_ret = float("nan")

        ok = (
            et is not None
            and ht is not None
            and abs((et.entry_date - ht["entry_date"]).days) <= 1
            and abs((et.exit_date - ht["exit_date"]).days) <= 1
            and abs(e_ret - h_ret) < 0.5
        )
        match = "OK" if ok else "MISS"
        if ok:
            matches += 1
        lines.append(
            f"{i+1:>3} | {e_entry:>12} {e_exit:>12} {e_ret:>+8.2f} {e_reason:>10} | "
            f"{h_entry:>12} {h_exit:>12} {h_ret:>+8.2f} | {match}"
        )

    lines.append("")
    e_total = engine_out["metrics"]["total_return"] * 100.0
    h_pnl = hand_out["total_return"] * 100.0
    h_match = hand_out["total_return_engine_match"] * 100.0
    delta_match = e_total - h_match
    lines.append(
        f"Total return  engine                   = {e_total:+.4f}%"
    )
    lines.append(
        f"              hand (sum pnl / capital) = {h_pnl:+.4f}%"
    )
    lines.append(
        f"              hand (engine-match: sum pnl / eq[0])"
        f" = {h_match:+.4f}%   Δ vs engine = {delta_match:+.4f}%"
    )

    summary = {
        "engine_n": len(e_trades),
        "hand_n": len(h_trades),
        "matches": matches,
        "engine_total_return_pct": e_total,
        "hand_pnl_pct": h_pnl,
        "hand_engine_match_pct": h_match,
        "delta_pct": delta_match,
    }
    return "\n".join(lines), summary


def print_summary(bh_rows: list[tuple], gc_summary: dict) -> None:
    print_section("SUMMARY")

    by_ticker: dict[str, list[tuple]] = {}
    for row in bh_rows:
        by_ticker.setdefault(row[0], []).append(row)

    print("Buy-and-hold (vs aligned ground truth):")
    flagged_unexpected: list[str] = []
    for ticker, rows in by_ticker.items():
        flags = [(r[1], r[5], r[6]) for r in rows if r[7]]
        if not flags:
            print(f"  {ticker:<5}: PASS — all 5 metrics within tolerance")
        else:
            descs = [f"{m}(Δ={da:+.4f})" for m, da, dp in flags]
            print(f"  {ticker:<5}: FLAG — {', '.join(descs)}")
            for m, _, _ in flags:
                flagged_unexpected.append(f"{ticker}/{m}")

    print()
    print("Golden cross (AAPL):")
    n_e = gc_summary["engine_n"]
    n_h = gc_summary["hand_n"]
    matches = gc_summary["matches"]
    delta = gc_summary["delta_pct"]
    if matches == n_h == n_e and abs(delta) < 0.5:
        verdict = "PASS"
    else:
        verdict = "FLAG"
        flagged_unexpected.append(f"golden_cross(Δ={delta:+.2f}%)")
    print(
        f"  AAPL : {verdict} — engine={n_e} hand={n_h} per-trade-matched={matches} "
        f"total-return Δ vs engine-match={delta:+.4f}%"
    )

    print()
    print("Hypothesized divergence sources:")
    print("  - Entry T+1 lag (expected): engine fills at next-bar open after as_of,")
    print("    naive truth starts at first-day close. Visible in 'Truth(naive)' column;")
    print("    aligned truth strips this out.")
    print("  - Entry-open vs first-close micro-gap (expected, typically <0.05%):")
    print("    even with window alignment, engine begins from open[as_of+1] while")
    print("    aligned truth starts from close[as_of].")
    print("  - End-of-window force close (expected): engine.py:886-910 closes any")
    print("    open position at last close with sell-side slippage (zero here).")
    print("  - Engine total_return uses equity_curve[0] as denominator, not")
    print("    initial_capital. equity_curve[0] is MTM at entry-day CLOSE, not")
    print("    initial_capital, so a strategy whose first entry-day shows positive")
    print("    intraday gain (open→close) understates total_return. The hand-rolled")
    print("    'engine-match' total_return mirrors this by dividing by")
    print("    shares*close[first_entry_day] instead of initial_capital.")
    print("  - Fractional shares (NOT a source): portfolio.open uses fractional")
    print("    shares, so no integer-rounding cash drift.")
    print("  - Slippage / commission (NOT a source): both set to 0 in this script.")
    print("  - Dividend handling (NOT a source): YFinancePriceFetcher auto_adjust=True")
    print("    folds dividends into OHLC for both engine and ground-truth fetches.")
    print("  - Sharpe ddof (NOT a source): metrics.py and ground-truth both ddof=0.")
    print("  - CAGR year denominator (NOT a source): both use len/252, not calendar years.")

    print()
    if flagged_unexpected:
        print(f"UNEXPECTED FLAGS: {', '.join(flagged_unexpected)}")
        print("These exceed tolerance vs ALIGNED truth — investigate engine math.")
    else:
        print("All flagged divergences (if any) are vs naive truth and are EXPECTED.")
        print("Aligned-truth comparisons all within tolerance: engine math validated.")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    print("Validating screener.backtester against independent ground truth")
    print(f"Tickers: {', '.join(TICKERS)} | Window: {START} -> {END}")

    raw = YFinancePriceFetcher()
    fetcher = CappedFetcher(raw, END)

    bh_rows: list[tuple] = []
    print_section("BUY-AND-HOLD VALIDATION")
    for ticker in TICKERS:
        print(f"  running engine buy-and-hold: {ticker} ...", flush=True)
        engine_out = run_engine_buy_and_hold(ticker, fetcher)
        if engine_out["warnings"]:
            print(f"    WARNINGS: {engine_out['warnings']}")
        if not engine_out["trades"]:
            print(f"    ERROR: engine produced no trades for {ticker} — entry_expr='1' may be broken")
        rows = compare_buy_and_hold(ticker, engine_out, fetcher)
        bh_rows.extend(rows)

    print()
    print(format_buy_and_hold_table(bh_rows))

    print_section("GOLDEN CROSS SANITY CHECK — AAPL")
    print("  running hand-rolled golden cross (AAPL) ...", flush=True)
    gc_hand = hand_rolled_golden_cross("AAPL", fetcher)
    if not gc_hand["trades"]:
        print("    ERROR: hand-rolled produced no crossovers in window")
        return 1
    # Engine select_candidates only fires if entry_expr is True at as_of, so
    # set as_of to the bar BEFORE the first hand-rolled entry (the signal day).
    first_entry = gc_hand["trades"][0]["entry_date"]
    bars = fetcher.fetch(["AAPL"], START, END)["AAPL"]
    bars_idx = pd.to_datetime(bars.index)
    entry_pos = int(np.where(bars_idx == pd.Timestamp(first_entry))[0][0])
    signal_day = bars_idx[entry_pos - 1].date()
    print(
        f"  running engine golden cross (AAPL, as_of={signal_day}, "
        f"allow_reentry=True) ...",
        flush=True,
    )
    gc_engine = run_engine_golden_cross("AAPL", fetcher, signal_day)
    if gc_engine["warnings"]:
        print(f"    WARNINGS: {gc_engine['warnings']}")

    report, gc_summary = format_golden_cross_report(gc_engine, gc_hand)
    print()
    print(report)

    print_summary(bh_rows, gc_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
