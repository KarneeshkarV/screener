from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from screener import history, metrics
from screener.historical_exits import ExitPolicy, resolve_exit, resolve_trades
from screener.prices import fetch_adj_close_matrix, fetch_ohlcv


BENCHMARKS = {
    "us": "AMEX:SPY",
    "india": "NSE:NIFTY",
}


@dataclass
class BacktestResult:
    run_id: int
    run_ts: str
    market: str
    criteria: str
    scope: str
    hold_days: int
    start_date: date
    end_date: date
    benchmark: str
    tickers: list[str]
    dropped: list[str]
    basket_curve: pd.Series
    basket_returns: pd.Series
    benchmark_curve: pd.Series
    benchmark_returns: pd.Series
    per_ticker: pd.DataFrame
    summary: dict


def _parse_run_ts(run_ts: str) -> datetime:
    try:
        return datetime.fromisoformat(run_ts)
    except ValueError:
        return datetime.fromisoformat(run_ts.replace("Z", "+00:00"))


def _resolve_scope(run_ts_str: str, scope: str) -> tuple[date, date]:
    today = date.today()
    if scope == "next":
        run_dt = _parse_run_ts(run_ts_str)
        start = run_dt.date()
        end = today
        if start > end:
            start = end
        return start, end
    # all: last ~5 years up to today
    end = today
    start = end - timedelta(days=365 * 5)
    return start, end


def _fetch_benchmark(
    benchmark_sym: str,
    start_d: date,
    end_d: date,
    market: str,
    basket_index: pd.Index,
    refresh: bool,
) -> tuple[pd.Series, pd.Series]:
    """Fetch benchmark prices and align to *basket_index*.

    Returns ``(benchmark_curve, benchmark_returns)``.  Falls back to a flat
    curve if the benchmark cannot be fetched.
    """
    bench_df = fetch_ohlcv(benchmark_sym, start_d, end_d, market, refresh=refresh)
    if bench_df is None or bench_df.empty:
        bench_df = fetch_ohlcv(benchmark_sym, start_d, end_d, "us", refresh=refresh)

    if bench_df is not None and not bench_df.empty:
        bench_series = pd.Series(
            bench_df["adj_close"].values,
            index=pd.to_datetime(bench_df["date"]).values,
        ).sort_index()
        bench_series = bench_series.reindex(basket_index).ffill().bfill()
        benchmark_returns = bench_series.pct_change().fillna(0.0)
        benchmark_curve = (1 + benchmark_returns).cumprod()
    else:
        benchmark_returns = pd.Series(0.0, index=basket_index)
        benchmark_curve = pd.Series(1.0, index=basket_index)

    return benchmark_curve, benchmark_returns


def _forward_test_from_matrix(
    wide: pd.DataFrame,
    hold_days: int,
    benchmark_sym: str,
    market: str,
    refresh: bool,
    exit_policy: Optional[ExitPolicy] = None,
    ohlcv_full: Optional[dict[str, pd.DataFrame]] = None,
    re_entry: bool = False,
    entry_eval_fns: Optional[dict[str, Callable[[int], bool]]] = None,
) -> dict:
    """Core forward-test logic: given a wide adj-close matrix, compute basket
    and benchmark curves, per-ticker returns, and summary metrics.

    *wide* must already be sorted by date.  Tickers with no price on the first
    bar are dropped.

    When *exit_policy* is provided and non-noop, per-ticker exits are resolved
    against *ohlcv_full* (full OHLCV history including pre-entry warmup bars
    so EMAs/RSI/MACD can be computed).  Otherwise the legacy uniform
    ``hold_days`` cutoff applies to every ticker.

    Returns a dict with keys: valid_tickers, dropped, basket_curve,
    basket_returns, benchmark_curve, benchmark_returns, per_ticker, summary.
    """
    wide = wide.sort_index()
    wide = wide.ffill(limit=1)

    first_row = wide.iloc[0]
    valid_tickers = [c for c in wide.columns if pd.notna(first_row[c])]
    dropped_extra = [c for c in wide.columns if c not in valid_tickers]
    wide = wide[valid_tickers]

    start_d = wide.index[0].date() if hasattr(wide.index[0], "date") else wide.index[0]
    end_d = wide.index[-1].date() if hasattr(wide.index[-1], "date") else wide.index[-1]

    use_per_ticker_exits = (
        exit_policy is not None
        and not exit_policy.is_noop()
        and ohlcv_full is not None
    )

    # Per-ticker exit resolution.  If not using per-ticker exits, every ticker
    # gets the same time-stop at hold_days so the legacy path falls out as a
    # special case of the new code.
    exit_info: dict[str, tuple[int, str]] = {}
    trades_info: dict[str, list[tuple[int, int, str]]] = {}
    entry_ts = wide.index[0]
    for t in valid_tickers:
        forward_len = int(wide[t].notna().sum())
        if use_per_ticker_exits and t in ohlcv_full:
            full_df = ohlcv_full[t].copy()
            full_df["_date"] = pd.to_datetime(full_df["date"]).dt.normalize()
            full_df = full_df.sort_values("_date").reset_index(drop=True)
            entry_rows = full_df.index[full_df["_date"] == entry_ts]
            if len(entry_rows) == 0:
                cap = min(forward_len - 1, hold_days) if hold_days > 0 else forward_len - 1
                exit_info[t] = (cap, "time")
                trades_info[t] = [(0, cap, "time")]
                continue
            policy = exit_policy if exit_policy.time_stop_bars is not None else ExitPolicy(
                stop_loss=exit_policy.stop_loss,
                take_profit=exit_policy.take_profit,
                trailing_stop=exit_policy.trailing_stop,
                signals=exit_policy.signals,
                time_stop_bars=hold_days if hold_days > 0 else None,
            )
            if re_entry and entry_eval_fns is not None and t in entry_eval_fns:
                ticker_trades = resolve_trades(
                    full_df, int(entry_rows[0]), forward_len, policy,
                    entry_eval_fn=entry_eval_fns[t], cooldown_bars=1,
                )
                trades_info[t] = ticker_trades
                first = ticker_trades[0]
                exit_info[t] = (first[1], first[2])
            else:
                exit_info[t] = resolve_exit(
                    full_df, int(entry_rows[0]), forward_len, policy
                )
                trades_info[t] = [(0, exit_info[t][0], exit_info[t][1])]
        else:
            cap = min(forward_len - 1, hold_days) if hold_days > 0 else forward_len - 1
            exit_info[t] = (max(cap, 0), "time")
            trades_info[t] = [(0, max(cap, 0), "time")]

    # Build per-ticker return mask: held bars are True, cash bars False.
    # For single-entry mode this is (entry, exit_bar]; for re-entry mode the
    # mask is the union of each round-trip's (entry_bar, exit_bar] window.
    ticker_returns = wide.pct_change()
    mask = pd.DataFrame(False, index=ticker_returns.index, columns=ticker_returns.columns)
    for t in valid_tickers:
        col_idx = mask.columns.get_loc(t)
        for entry_rel, exit_rel, _ in trades_info[t]:
            if exit_rel > entry_rel:
                mask.iloc[entry_rel + 1 : exit_rel + 1, col_idx] = True
    ticker_returns = ticker_returns.where(mask, other=0.0)

    basket_returns = ticker_returns.mean(axis=1, skipna=True).fillna(0.0)
    basket_curve = (1 + basket_returns).cumprod()

    benchmark_curve, benchmark_returns = _fetch_benchmark(
        benchmark_sym, start_d, end_d, market, basket_curve.index, refresh
    )

    per_ticker_rows = []
    for t in valid_tickers:
        col = wide[t].dropna()
        if col.empty:
            continue
        trades = trades_info.get(t) or [(0, exit_info[t][0], exit_info[t][1])]
        max_idx = len(col) - 1
        first_entry_rel, _, _ = trades[0]
        last_exit_rel, last_reason = trades[-1][1], trades[-1][2]
        first_entry_idx = min(max_idx, first_entry_rel)
        last_exit_idx = min(max_idx, last_exit_rel)
        entry = float(col.iloc[first_entry_idx])
        exit_px = float(col.iloc[last_exit_idx])

        if len(trades) > 1:
            # Compound trade-level returns so the per-ticker figure matches
            # what the basket mask produces.
            compounded = 1.0
            held_days = 0
            for entry_rel, exit_rel, _ in trades:
                if exit_rel <= entry_rel:
                    continue
                e_px = float(col.iloc[min(max_idx, entry_rel)])
                x_px = float(col.iloc[min(max_idx, exit_rel)])
                if e_px > 0:
                    compounded *= x_px / e_px
                held_days += exit_rel - entry_rel
            ret = compounded - 1.0
            reason_label = last_reason if len(trades) == 1 else "multi"
            trading_days = int(held_days)
        else:
            reason_label = last_reason
            trading_days = int(last_exit_idx - first_entry_idx)
            ret = exit_px / entry - 1.0 if entry > 0 else float("nan")

        per_ticker_rows.append({
            "ticker": t,
            "entry_close": entry,
            "exit_close": exit_px,
            "return_pct": ret,
            "trading_days": trading_days,
            "exit_reason": reason_label,
            "trades": int(len(trades)),
        })
    per_ticker = pd.DataFrame(per_ticker_rows).sort_values(
        "return_pct", ascending=False
    ).reset_index(drop=True)

    summary = metrics.summarise(
        basket_curve=basket_curve,
        basket_returns=basket_returns,
        benchmark_curve=benchmark_curve,
        benchmark_returns=benchmark_returns,
        per_ticker_returns=per_ticker["return_pct"] if not per_ticker.empty else pd.Series(dtype=float),
    )

    return {
        "valid_tickers": valid_tickers,
        "dropped_extra": dropped_extra,
        "basket_curve": basket_curve,
        "basket_returns": basket_returns,
        "benchmark_curve": benchmark_curve,
        "benchmark_returns": benchmark_returns,
        "per_ticker": per_ticker,
        "summary": summary,
    }


def run_backtest(
    market: Optional[str],
    criteria: Optional[str],
    scope: str = "next",
    hold_days: int = 0,
    top: Optional[int] = None,
    benchmark_override: Optional[str] = None,
    refresh: bool = False,
) -> BacktestResult:
    run = history.load_last_run(market=market, criteria=criteria)
    if run is None:
        raise RuntimeError(
            "No saved screener run found. Run `screen` first to seed history."
        )

    rows = run["rows"]
    if top:
        rows = rows.head(top)
    tickers = [t for t in rows["ticker"].tolist() if t]
    if not tickers:
        raise RuntimeError("Last run has no tickers.")

    run_market = run["market"]
    start_d, end_d = _resolve_scope(run["run_ts"], scope)

    wide, failed = fetch_adj_close_matrix(tickers, start_d, end_d, run_market, refresh=refresh)
    if wide.empty:
        raise RuntimeError(
            f"No price data fetched for any of {len(tickers)} tickers "
            f"(window {start_d} → {end_d})."
        )
    if len(wide) < 2:
        raise RuntimeError(
            f"Only {len(wide)} trading day(s) in window {start_d} → {end_d}. "
            "Need ≥2 for returns. Use --scope all for a longer history."
        )

    benchmark_sym = benchmark_override or BENCHMARKS.get(run_market, "AMEX:SPY")
    ft = _forward_test_from_matrix(wide, hold_days, benchmark_sym, run_market, refresh)

    dropped = sorted(set(failed + ft["dropped_extra"]))

    result = BacktestResult(
        run_id=run["id"],
        run_ts=run["run_ts"],
        market=run_market,
        criteria=run["criteria"],
        scope=scope,
        hold_days=hold_days,
        start_date=start_d,
        end_date=end_d,
        benchmark=benchmark_sym,
        tickers=ft["valid_tickers"],
        dropped=dropped,
        basket_curve=ft["basket_curve"],
        basket_returns=ft["basket_returns"],
        benchmark_curve=ft["benchmark_curve"],
        benchmark_returns=ft["benchmark_returns"],
        per_ticker=ft["per_ticker"],
        summary=ft["summary"],
    )

    history.save_backtest(
        run_id=result.run_id,
        scope=scope,
        start_date=start_d.isoformat(),
        end_date=end_d.isoformat(),
        hold_days=hold_days,
        benchmark=benchmark_sym,
        summary=result.summary["basket"],
        per_ticker=ft["per_ticker"],
    )
    return result
