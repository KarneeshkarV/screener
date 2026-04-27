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


_INSIDER_INDIA_COLUMNS = [
    "name",
    "close",
    "promoter_pct_prev",
    "promoter_pct_latest",
    "promoter_change",
    "latest_quarter",
    "fii_pct_latest",
    "dii_pct_latest",
]

_INSIDER_US_COLUMNS = [
    "name",
    "description",
    "close",
    "yf_net_shares_6m",
    "yf_net_pct_6m",
    "yf_total_held",
    "yf_buy_trans_6m",
    "yf_sell_trans_6m",
]

_INSIDER_LABELS = {
    "promoter_pct_latest": "Promoter%",
    "promoter_pct_prev": "PrevQ%",
    "promoter_change": "ΔPP",
    "latest_quarter": "Quarter",
    "fii_pct_latest": "FII%",
    "dii_pct_latest": "DII%",
    "yf_net_shares_6m": "YF Net Shares 6m",
    "yf_net_pct_6m": "YF Net%",
    "yf_total_held": "Insider Held",
    "yf_buy_trans_6m": "Buys",
    "yf_sell_trans_6m": "Sells",
}


def _format_insider(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if col in {"promoter_pct_latest", "promoter_pct_prev", "fii_pct_latest", "dii_pct_latest"}:
        return f"{float(val):.2f}%"
    if col == "promoter_change":
        return f"{float(val):+.2f}"
    if col == "yf_net_pct_6m":
        return f"{float(val) * 100:+.3f}%"
    if col == "yf_net_shares_6m":
        v = float(val)
        sign = "+" if v >= 0 else ""
        if abs(v) >= 1_000_000:
            return f"{sign}{v / 1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{sign}{v / 1_000:.1f}K"
        return f"{sign}{v:,.0f}"
    if col == "yf_total_held":
        v = float(val)
        if v >= 1_000_000:
            return f"{v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v / 1_000:.1f}K"
        return f"{v:,.0f}"
    if col in {"yf_buy_trans_6m", "yf_sell_trans_6m"}:
        return f"{int(val)}"
    return _format_value(col, val)


def print_insider_results(
    df: pd.DataFrame,
    market: str,
    universe_size: int,
    match_count: int,
) -> None:
    label = "Promoter buys (India)" if market == "india" else "Insider buys (US)"
    console.print(
        f"\n[bold]{label}[/bold] — {match_count} matches from "
        f"{universe_size} liquid tickers, showing {len(df)}\n"
    )

    columns = _INSIDER_INDIA_COLUMNS if market == "india" else _INSIDER_US_COLUMNS
    columns = [c for c in columns if c in df.columns]

    table = Table(show_header=True, header_style="bold", show_lines=False)
    for col_name in columns:
        label = _INSIDER_LABELS.get(col_name, COLUMN_LABELS.get(col_name, col_name))
        justify = "right" if col_name not in {"name", "description", "latest_quarter"} else "left"
        if col_name == "description":
            table.add_column(label, justify=justify, min_width=12, max_width=22)
        elif col_name == "name":
            table.add_column(label, justify=justify, min_width=10, no_wrap=True)
        else:
            table.add_column(label, justify=justify, no_wrap=True)

    for _, row in df.iterrows():
        cells = [_format_insider(c, row[c]) for c in columns]
        table.add_row(*cells)
    console.print(table)
