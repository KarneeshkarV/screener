import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Symbol",
    "description": "Name",
    "close": "Close",
    "change": "Chg%",
    "volume": "Volume",
    "market_cap_basic": "Mkt Cap",
    "setup_score": "Score",
    "EMA5": "EMA5",
    "EMA20": "EMA20",
    "EMA100": "EMA100",
    "EMA200": "EMA200",
    "price_earnings_ttm": "P/E",
    "return_on_equity": "ROE%",
    "dividend_yield_recent": "Div%",
    "debt_to_equity": "D/E",
    "RSI": "RSI",
    "P/E": "P/E",
    "ROCE%": "ROCE%",
    "ROE%": "ROE%",
}

RIGHT_ALIGN = {
    "close",
    "change",
    "volume",
    "market_cap_basic",
    "setup_score",
    "EMA5",
    "EMA20",
    "EMA100",
    "EMA200",
    "price_earnings_ttm",
    "return_on_equity",
    "dividend_yield_recent",
    "debt_to_equity",
    "RSI",
    "P/E",
    "ROCE%",
    "ROE%",
}


def _format_value(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"

    if col == "change":
        return f"{val:+.2f}%"
    if col == "volume":
        if val >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{val:,.0f}"
    if col == "market_cap_basic":
        if val >= 1e12:
            return f"{val / 1e12:.2f}T"
        if val >= 1e9:
            return f"{val / 1e9:.2f}B"
        if val >= 1e6:
            return f"{val / 1e6:.1f}M"
        return f"{val:,.0f}"
    if col in ("close", "EMA5", "EMA20", "EMA100", "EMA200"):
        return f"{val:.2f}"
    if col in (
        "setup_score",
        "price_earnings_ttm",
        "return_on_equity",
        "dividend_yield_recent",
        "debt_to_equity",
        "RSI",
        "P/E",
        "ROCE%",
        "ROE%",
    ):
        return f"{val:.2f}"

    return str(val)


def print_results(
    df: pd.DataFrame,
    total: int,
    market: str,
    criteria_name: str,
    added: list[str] | None = None,
    removed: list[str] | None = None,
    first_run: bool = False,
) -> None:
    console.print(
        f"\n[bold]{criteria_name.upper()}[/bold] screen on "
        f"[cyan]{market.upper()}[/cyan] — "
        f"{total} matches, showing {len(df)}\n"
    )

    skip = {"ticker"}
    if len(df.columns) > 8:
        skip.add("description")
    display_cols = [c for c in df.columns if c not in skip]

    table = Table(show_header=True, header_style="bold", show_lines=False)

    for col_name in display_cols:
        label = COLUMN_LABELS.get(col_name, col_name)
        justify = "right" if col_name in RIGHT_ALIGN else "left"
        if col_name == "name":
            table.add_column(label, justify=justify, min_width=8, no_wrap=True)
        elif col_name == "description":
            table.add_column(label, justify=justify, min_width=12, max_width=20)
        else:
            table.add_column(label, justify=justify, no_wrap=True)

    for _, row in df.iterrows():
        cells = [_format_value(col_name, row[col_name]) for col_name in display_cols]
        table.add_row(*cells)

    console.print(table)
    _print_diff(market, criteria_name, added or [], removed or [], first_run)


def _print_diff(
    market: str,
    criteria_name: str,
    added: list[str],
    removed: list[str],
    first_run: bool,
) -> None:
    if first_run:
        console.print(
            f"[dim]No prior run for {market} / {criteria_name} — saved as baseline.[/dim]"
        )
        return

    if not added and not removed:
        console.print("[dim]No changes since last run.[/dim]")
        return

    console.print("\n[bold]Diff vs previous run[/bold]")
    if added:
        console.print(
            f"  [green]+ {', '.join(added)}[/green]  "
            f"[dim]({len(added)} new)[/dim]"
        )
    if removed:
        console.print(
            f"  [red]- {', '.join(removed)}[/red]  "
            f"[dim]({len(removed)} dropped)[/dim]"
        )


def print_csv(df: pd.DataFrame) -> None:
    print(df.to_csv(index=False))


def _fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    return f"{val * 100:+.2f}%"


def _fmt_num(val, digits: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    return f"{val:.{digits}f}"


_EXIT_SHORT = {
    "time":   ("time",  "dim"),
    "stop":   ("stop",  "red"),
    "target": ("tgt",   "green"),
    "trail":  ("trail", "yellow"),
}


def _fmt_exit_reason(reason) -> str:
    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return "-"
    reason = str(reason)
    if reason.startswith("signal:"):
        label = "sig:" + reason.split(":", 1)[1]
        return f"[cyan]{label}[/cyan]"
    short, style = _EXIT_SHORT.get(reason, (reason, ""))
    return f"[{style}]{short}[/{style}]" if style else short


def print_backtest(result) -> None:
    console.print(
        f"\n[bold]Backtest[/bold] — run #{result.run_id} "
        f"([cyan]{result.market.upper()}[/cyan] / {result.criteria}) "
        f"@ {result.run_ts}"
    )
    console.print(
        f"  scope=[yellow]{result.scope}[/yellow]  window "
        f"{result.start_date} → {result.end_date}  "
        f"hold={result.hold_days or 'full'}  "
        f"tickers={len(result.tickers)}  "
        f"benchmark={result.benchmark}"
    )
    if result.dropped:
        console.print(
            f"  [dim]dropped {len(result.dropped)}: "
            f"{', '.join(result.dropped[:8])}"
            + ("…" if len(result.dropped) > 8 else "")
            + "[/dim]"
        )

    summary = result.summary
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("", justify="left")
    for col in ("Total", "CAGR", "Vol", "Sharpe", "MaxDD", "Hit%", "Alpha", "Beta"):
        table.add_column(col, justify="right")

    for label, stats in (("Basket", summary["basket"]), ("Benchmark", summary["benchmark"])):
        table.add_row(
            label,
            _fmt_pct(stats["total_return"]),
            _fmt_pct(stats["cagr"]),
            _fmt_pct(stats["vol"]),
            _fmt_num(stats["sharpe"]),
            _fmt_pct(stats["max_drawdown"]),
            _fmt_pct(stats["hit_rate"]) if stats["hit_rate"] == stats["hit_rate"] else "-",
            _fmt_pct(stats["alpha"]) if stats["alpha"] is not None else "-",
            _fmt_num(stats["beta"]) if stats["beta"] is not None else "-",
        )
    console.print(table)

    if result.per_ticker.empty:
        return

    pt = Table(title="Per-ticker", show_header=True, header_style="bold")
    pt.add_column("Ticker")
    pt.add_column("Entry", justify="right")
    pt.add_column("Exit", justify="right")
    pt.add_column("Return", justify="right")
    pt.add_column("Days", justify="right")

    for _, row in result.per_ticker.iterrows():
        ret = row["return_pct"]
        style = "green" if pd.notna(ret) and ret > 0 else ("red" if pd.notna(ret) and ret < 0 else "")
        pt.add_row(
            row["ticker"],
            _fmt_num(row["entry_close"]),
            _fmt_num(row["exit_close"]),
            f"[{style}]{_fmt_pct(ret)}[/{style}]" if style else _fmt_pct(ret),
            str(int(row["trading_days"])),
        )
    console.print(pt)


def print_backtest_csv(result) -> None:
    print(result.per_ticker.to_csv(index=False))


def print_historical_backtest(result) -> None:
    console.print(
        f"\n[bold]Historical Backtest[/bold] — "
        f"[cyan]{result.market.upper()}[/cyan] / {result.criteria}"
    )
    console.print(
        f"  Screen as-of: [yellow]{result.as_of_date}[/yellow]  "
        f"Entry: {result.entry_date}  Exit: {result.exit_date}  "
        f"Hold: {result.hold_days} trading days"
    )
    console.print(
        f"  Universe: {result.universe_label} ({result.universe_size} tickers)  "
        f"Matches: {result.matches_total}  Selected: top {result.top_n}  "
        f"Benchmark: {result.benchmark}"
    )
    policy = getattr(result, "exit_policy", None)
    if policy is not None and not policy.is_noop():
        parts = []
        if policy.stop_loss is not None:
            parts.append(f"stop {policy.stop_loss:.0%}")
        if policy.take_profit is not None:
            parts.append(f"target {policy.take_profit:.0%}")
        if policy.trailing_stop is not None:
            parts.append(f"trail {policy.trailing_stop:.0%}")
        if policy.signals:
            parts.append("signals=" + ",".join(policy.signals))
        if getattr(result, "re_entry", False):
            parts.append("re-entry ON")
        console.print(f"  [magenta]Exits:[/magenta] " + " · ".join(parts))
    elif getattr(result, "re_entry", False):
        console.print("  [magenta]Exits:[/magenta] re-entry ON")
    if result.failed:
        console.print(
            f"  [dim]no data: {len(result.failed)} tickers[/dim]"
        )
    if result.skipped:
        console.print(
            f"  [dim]skipped (insufficient history): {len(result.skipped)} tickers[/dim]"
        )
    if result.dropped:
        console.print(
            f"  [dim]dropped (no forward data): "
            + ", ".join(result.dropped[:8])
            + ("…" if len(result.dropped) > 8 else "")
            + "[/dim]"
        )

    summary = result.summary
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("", justify="left")
    for col in ("Total", "CAGR", "Vol", "Sharpe", "MaxDD", "Hit%", "Alpha", "Beta"):
        table.add_column(col, justify="right")

    for label, stats in (("Basket", summary["basket"]), ("Benchmark", summary["benchmark"])):
        table.add_row(
            label,
            _fmt_pct(stats["total_return"]),
            _fmt_pct(stats["cagr"]),
            _fmt_pct(stats["vol"]),
            _fmt_num(stats["sharpe"]),
            _fmt_pct(stats["max_drawdown"]),
            _fmt_pct(stats["hit_rate"]) if stats["hit_rate"] == stats["hit_rate"] else "-",
            _fmt_pct(stats["alpha"]) if stats["alpha"] is not None else "-",
            _fmt_num(stats["beta"]) if stats["beta"] is not None else "-",
        )
    console.print(table)

    if result.per_ticker.empty:
        return

    show_trades = getattr(result, "re_entry", False)
    pt = Table(title="Selected tickers", show_header=True, header_style="bold")
    pt.add_column("Rank", justify="right")
    pt.add_column("Ticker")
    pt.add_column("Score", justify="right")
    pt.add_column("Entry", justify="right")
    pt.add_column("Exit", justify="right")
    pt.add_column("Return", justify="right")
    pt.add_column("Days", justify="right")
    if show_trades:
        pt.add_column("Trades", justify="right")
    pt.add_column("Why")

    for _, row in result.per_ticker.iterrows():
        ret = row.get("return_pct")
        style = (
            "green" if pd.notna(ret) and ret > 0
            else ("red" if pd.notna(ret) and ret < 0 else "")
        )
        cells = [
            str(int(row["rank"])),
            row["ticker"],
            _fmt_num(row.get("score"), 4),
            _fmt_num(row.get("entry_close")),
            _fmt_num(row.get("exit_close")),
            f"[{style}]{_fmt_pct(ret)}[/{style}]" if style else _fmt_pct(ret),
            str(int(row["trading_days"])),
        ]
        if show_trades:
            trade_count = row.get("trades")
            cells.append(str(int(trade_count)) if pd.notna(trade_count) else "-")
        cells.append(_fmt_exit_reason(row.get("exit_reason")))
        pt.add_row(*cells)
    console.print(pt)


def print_historical_backtest_csv(result) -> None:
    print(result.per_ticker.to_csv(index=False))
