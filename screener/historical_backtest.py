"""Historical backtest: screen as of a past date, test forward.

No lookahead bias — criteria are evaluated using only OHLCV data available on
or before ``--as-of``.  Because TradingView Screener has no as-of-date API,
criteria are computed locally from downloaded price history.

Only OHLCV-computable criteria are supported (ema, breakout, ema_breakout).
Fundamentals-based criteria (value, quality, etc.) are rejected at the CLI.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from screener import history, metrics
from screener.backtest import BENCHMARKS, _fetch_benchmark, _forward_test_from_matrix
from screener.fundamentals import fetch_fundamentals, fundamental_snapshot
from screener.historical_criteria import (
    FUND_CRITERIA,
    HIST_CRITERIA,
    compute_scores,
)
from screener.prices import fetch_ohlcv
from screener.universes import load_universe


@dataclass
class HistoricalBacktestResult:
    market: str
    criteria: str
    as_of_date: date
    entry_date: date
    exit_date: date
    hold_days: int
    top_n: int
    universe_label: str
    universe_size: int
    matches_total: int
    benchmark: str
    tickers: list[str]           # selected tickers (ranked)
    ticker_scores: list[float]   # setup_score for each selected ticker
    failed: list[str]            # tickers with no price data
    skipped: list[str]           # tickers with insufficient history
    dropped: list[str]           # tickers dropped after entry (no forward data)
    basket_curve: pd.Series
    basket_returns: pd.Series
    benchmark_curve: pd.Series
    benchmark_returns: pd.Series
    per_ticker: pd.DataFrame
    summary: dict


def run_historical_backtest(
    market: str,
    criteria_name: str,
    as_of: date,
    hold_days: int = 252,
    top: int = 20,
    universe_path: Optional[str] = None,
    benchmark_override: Optional[str] = None,
    refresh: bool = False,
) -> HistoricalBacktestResult:
    """Screen a universe as of *as_of*, then backtest matches forward.

    *criteria_name* can be a single name (``"ema"``) or a ``"+"``-separated
    combination (``"ema+breakout"``).  When combined, a ticker must pass
    **all** constituent filters to be included.

    Raises ``click.ClickException`` for unsupported criteria.
    Raises ``RuntimeError`` if no matches or no price data are found.
    """
    # ── validate criteria ────────────────────────────────────────────────────
    criteria_parts = [c.strip() for c in criteria_name.split("+")]
    for c in criteria_parts:
        if c not in HIST_CRITERIA:
            raise click.ClickException(
                f"Unknown criterion '{c}'. "
                f"Historical mode supports: {sorted(HIST_CRITERIA)}."
            )

    need_fundamentals = any(c in FUND_CRITERIA for c in criteria_parts)

    eval_fns = [HIST_CRITERIA[c] for c in criteria_parts]

    # ── load universe ────────────────────────────────────────────────────────
    if universe_path:
        universe_label = Path(universe_path).name
    else:
        universe_label = f"default_{market}"

    all_tickers = load_universe(market, universe_path)
    universe_size = len(all_tickers)
    click.echo(
        f"Universe: {universe_label} ({universe_size} tickers) | "
        f"as-of: {as_of} | criteria: {criteria_name} | hold: {hold_days}d | top: {top}",
        err=True,
    )

    # ── date ranges ──────────────────────────────────────────────────────────
    as_of_ts = pd.Timestamp(as_of).normalize()
    # need 2+ years lookback so EMA200 is well-seeded before as_of
    fetch_start = (as_of_ts - pd.DateOffset(years=2)).date()
    # forward: hold_days trading days + a buffer for non-trading days (~1.5x)
    fetch_end = (as_of_ts + pd.DateOffset(days=int(hold_days * 1.5) + 30)).date()

    # ── fetch OHLCV (+ fundamentals if needed) ───────────────────────────────
    ohlcv_data: dict[str, pd.DataFrame] = {}
    fund_data: dict[str, Optional[dict]] = {}
    failed: list[str] = []

    def _fetch(ticker: str):
        ohlcv = fetch_ohlcv(ticker, fetch_start, fetch_end, market, refresh=refresh)
        fund  = fetch_fundamentals(ticker, market, refresh=refresh) if need_fundamentals else None
        return ticker, ohlcv, fund

    label = "Fetching price history + fundamentals…" if need_fundamentals else "Fetching price history…"
    click.echo(label, err=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch, t): t for t in all_tickers}
        done = 0
        for fut in as_completed(futures):
            ticker, df, fund = fut.result()
            done += 1
            if df is None or df.empty:
                failed.append(ticker)
            else:
                ohlcv_data[ticker] = df
                if need_fundamentals:
                    fund_data[ticker] = fund
            if done % 50 == 0 or done == len(all_tickers):
                click.echo(f"  {done}/{len(all_tickers)} fetched…", err=True)

    if not ohlcv_data:
        raise RuntimeError("No price data fetched for any ticker in the universe.")

    # ── screen as of as_of_ts ────────────────────────────────────────────────
    click.echo("Screening…", err=True)
    matches: list[tuple[str, dict]] = []   # (ticker, eval_result)
    skipped: list[str] = []

    for ticker, df in ohlcv_data.items():
        # Compute point-in-time fundamentals snapshot when needed
        fund_snap: Optional[dict] = None
        if need_fundamentals:
            raw_fund = fund_data.get(ticker)
            if raw_fund is not None:
                close_on_asof = None
                df_dates = pd.to_datetime(df["date"]).dt.normalize()
                on_or_before = df[df_dates <= as_of_ts]
                if not on_or_before.empty:
                    close_on_asof = float(on_or_before["close"].iloc[-1])
                if close_on_asof:
                    fund_snap = fundamental_snapshot(raw_fund, as_of_ts, close_on_asof)

        # Evaluate all criteria; ticker must pass every one.
        all_pass = True
        last_result = None
        for eval_fn in eval_fns:
            result = eval_fn(df, as_of_ts, fund_snap)
            if result is None:
                all_pass = False
                break
            last_result = result
            if not result["passes"]:
                all_pass = False
                break
        if last_result is None:
            skipped.append(ticker)
        elif all_pass:
            matches.append((ticker, last_result))

    matches_total = len(matches)
    click.echo(
        f"Matches: {matches_total} / {len(ohlcv_data)} screened "
        f"({len(skipped)} skipped — insufficient history, {len(failed)} no data)",
        err=True,
    )

    if matches_total == 0:
        raise RuntimeError(
            f"No tickers matched '{criteria_name}' as of {as_of}. "
            "Try a broader universe or a different criteria / date."
        )

    # ── rank and select top-N ────────────────────────────────────────────────
    score_list = compute_scores([r["score_inputs"] for _, r in matches])
    ranked = sorted(
        zip([t for t, _ in matches], score_list),
        key=lambda x: x[1],
        reverse=True,
    )
    selected_tickers = [t for t, _ in ranked[:top]]
    selected_scores = [s for _, s in ranked[:top]]

    click.echo(
        f"Selected top {len(selected_tickers)}: "
        + ", ".join(selected_tickers[:10])
        + ("…" if len(selected_tickers) > 10 else ""),
        err=True,
    )

    # ── determine entry date: first trading day strictly after as_of ─────────
    entry_date: Optional[date] = None
    for ticker in selected_tickers:
        df = ohlcv_data[ticker]
        df_dates = pd.to_datetime(df["date"]).dt.normalize()
        after = df_dates[df_dates > as_of_ts]
        if not after.empty:
            candidate = after.iloc[0].date()
            if entry_date is None or candidate < entry_date:
                entry_date = candidate

    if entry_date is None:
        raise RuntimeError(
            f"No trading days found after {as_of} for any selected ticker. "
            "Extend the fetch window or choose an earlier --as-of date."
        )

    # ── build wide forward price matrix ─────────────────────────────────────
    series: dict[str, pd.Series] = {}
    for ticker in selected_tickers:
        df = ohlcv_data[ticker]
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        forward = df[df["_date"].dt.date >= entry_date]
        if forward.empty:
            continue
        s = pd.Series(
            forward["adj_close"].values,
            index=forward["_date"].values,
            name=ticker,
        )
        series[ticker] = s

    if not series:
        raise RuntimeError("No forward price data for any selected ticker.")

    wide = pd.DataFrame(series).sort_index()

    if len(wide) < 2:
        raise RuntimeError(
            f"Only {len(wide)} trading day(s) of forward data after {entry_date}. "
            "Use a more recent --as-of date or a smaller --hold value."
        )

    # ── forward test ─────────────────────────────────────────────────────────
    benchmark_sym = benchmark_override or BENCHMARKS.get(market, "AMEX:SPY")
    ft = _forward_test_from_matrix(wide, hold_days, benchmark_sym, market, refresh)

    dropped = sorted(set(ft["dropped_extra"]))
    final_tickers = ft["valid_tickers"]
    final_scores = [
        selected_scores[selected_tickers.index(t)]
        for t in final_tickers
        if t in selected_tickers
    ]

    # Attach score to per_ticker df
    score_map = dict(zip(selected_tickers, selected_scores))
    per_ticker = ft["per_ticker"].copy()
    per_ticker.insert(1, "score", per_ticker["ticker"].map(score_map))
    per_ticker.insert(2, "rank", range(1, len(per_ticker) + 1))

    # ── exit date ─────────────────────────────────────────────────────────────
    exit_date: date
    if not per_ticker.empty:
        exit_idx = int(per_ticker["trading_days"].max())
        if len(wide) > exit_idx:
            exit_ts = wide.index[exit_idx]
            exit_date = exit_ts.date() if hasattr(exit_ts, "date") else exit_ts
        else:
            exit_ts = wide.index[-1]
            exit_date = exit_ts.date() if hasattr(exit_ts, "date") else exit_ts
    else:
        exit_date = entry_date + timedelta(days=hold_days)

    # ── save to DB ────────────────────────────────────────────────────────────
    history.save_historical_backtest(
        market=market,
        criteria=criteria_name,
        as_of_date=as_of.isoformat(),
        entry_date=entry_date.isoformat(),
        exit_date=exit_date.isoformat(),
        hold_days=hold_days,
        top_n=top,
        universe_label=universe_label,
        universe_size=universe_size,
        matches_total=matches_total,
        benchmark=benchmark_sym,
        basket_summary=ft["summary"]["basket"],
        bench_summary=ft["summary"]["benchmark"],
        per_ticker=per_ticker,
    )

    return HistoricalBacktestResult(
        market=market,
        criteria=criteria_name,
        as_of_date=as_of,
        entry_date=entry_date,
        exit_date=exit_date,
        hold_days=hold_days,
        top_n=top,
        universe_label=universe_label,
        universe_size=universe_size,
        matches_total=matches_total,
        benchmark=benchmark_sym,
        tickers=final_tickers,
        ticker_scores=final_scores,
        failed=sorted(failed),
        skipped=sorted(skipped),
        dropped=dropped,
        basket_curve=ft["basket_curve"],
        basket_returns=ft["basket_returns"],
        benchmark_curve=ft["benchmark_curve"],
        benchmark_returns=ft["benchmark_returns"],
        per_ticker=per_ticker,
        summary=ft["summary"],
    )
