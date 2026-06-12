import pandas as pd
from rich.console import Console
from rich.table import Table

from screener.format import fmt_mcap, fmt_pct

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
    "garp_score": "GARP",
    "peg": "PEG",
    "sales": "Sales",
    "sales_growth_5y": "Sales 5Y%",
    "operating_profit_growth": "OP Growth%",
    "eps_growth_5y": "EPS 5Y%",
    "roe_5y": "ROE 5Y%",
    "roce_or_roic": "ROCE/ROIC",
    "quarterly_profit_growth": "Q Profit%",
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
    "garp_score",
    "peg",
    "sales",
    "sales_growth_5y",
    "operating_profit_growth",
    "eps_growth_5y",
    "roe_5y",
    "roce_or_roic",
    "quarterly_profit_growth",
}


def _format_value(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"

    if col == "change":
        return fmt_pct(val)
    if col == "volume":
        if val >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{val:,.0f}"
    if col == "market_cap_basic":
        return fmt_mcap(val)
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
        "garp_score",
        "peg",
        "sales_growth_5y",
        "operating_profit_growth",
        "eps_growth_5y",
        "roe_5y",
        "roce_or_roic",
        "quarterly_profit_growth",
    ):
        return f"{val:.2f}"
    if col == "sales":
        return fmt_mcap(val)

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
            f"  [green]+ {', '.join(added)}[/green]  [dim]({len(added)} new)[/dim]"
        )
    if removed:
        console.print(
            f"  [red]- {', '.join(removed)}[/red]  [dim]({len(removed)} dropped)[/dim]"
        )


def print_csv(df: pd.DataFrame) -> None:
    print(df.to_csv(index=False))


def print_garp_results(df: pd.DataFrame, market: str) -> None:
    console.print(
        f"\n[bold]GARP[/bold] screen on [cyan]{market.upper()}[/cyan] — "
        f"showing {len(df)}\n"
    )
    columns = [
        "name",
        "description",
        "garp_score",
        "peg",
        "sales",
        "sales_growth_5y",
        "operating_profit_growth",
        "eps_growth_5y",
        "roe_5y",
        "roce_or_roic",
        "quarterly_profit_growth",
    ]
    columns = [col for col in columns if col in df.columns]
    table = Table(show_header=True, header_style="bold", show_lines=False)
    for col_name in columns:
        label = COLUMN_LABELS.get(col_name, col_name)
        justify = "right" if col_name in RIGHT_ALIGN else "left"
        if col_name == "description":
            table.add_column(label, justify=justify, min_width=12, max_width=22)
        elif col_name == "name":
            table.add_column(label, justify=justify, min_width=8, no_wrap=True)
        else:
            table.add_column(label, justify=justify, no_wrap=True)
    for _, row in df.iterrows():
        table.add_row(*[_format_value(col, row[col]) for col in columns])
    console.print(table)


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
    "fmp_net_shares_6m",
    "fmp_buy_shares_6m",
    "fmp_sell_shares_6m",
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
    "fmp_net_shares_6m": "FMP Net 6m",
    "fmp_buy_shares_6m": "FMP Buy",
    "fmp_sell_shares_6m": "FMP Sell",
    "yf_net_shares_6m": "YF Net Shares 6m",
    "yf_net_pct_6m": "YF Net%",
    "yf_total_held": "Insider Held",
    "yf_buy_trans_6m": "Buys",
    "yf_sell_trans_6m": "Sells",
}


def _format_insider(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if col in {
        "promoter_pct_latest",
        "promoter_pct_prev",
        "fii_pct_latest",
        "dii_pct_latest",
    }:
        return f"{float(val):.2f}%"
    if col == "promoter_change":
        return f"{float(val):+.2f}"
    if col == "yf_net_pct_6m":
        return f"{float(val) * 100:+.3f}%"
    if col in {"yf_net_shares_6m", "fmp_net_shares_6m"}:
        v = float(val)
        sign = "+" if v >= 0 else ""
        if abs(v) >= 1_000_000:
            return f"{sign}{v / 1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{sign}{v / 1_000:.1f}K"
        return f"{sign}{v:,.0f}"
    if col in {"yf_total_held", "fmp_buy_shares_6m", "fmp_sell_shares_6m"}:
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
        justify = (
            "right"
            if col_name not in {"name", "description", "latest_quarter"}
            else "left"
        )
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


_INSTITUTIONAL_COLUMNS = [
    "symbol",
    "holders",
    "total_shares",
    "qoq_change_shares",
    "qoq_change_pct",
]

_INSTITUTIONAL_LABELS = {
    "symbol": "Symbol",
    "holders": "Holders",
    "total_shares": "Inst. Shares",
    "qoq_change_shares": "QoQ Chg",
    "qoq_change_pct": "QoQ Chg%",
}


def _format_institutional(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if col == "holders":
        return f"{int(val):,}"
    if col == "qoq_change_pct":
        return f"{float(val):+.2f}%"
    if col in {"total_shares", "qoq_change_shares"}:
        v = float(val)
        sign = "+" if col == "qoq_change_shares" and v >= 0 else ""
        if abs(v) >= 1_000_000_000:
            return f"{sign}{v / 1_000_000_000:.2f}B"
        if abs(v) >= 1_000_000:
            return f"{sign}{v / 1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{sign}{v / 1_000:.1f}K"
        return f"{sign}{v:,.0f}"
    return str(val)


def print_institutional_results(df: pd.DataFrame) -> None:
    console.print(
        f"\n[bold]Institutional ownership (US)[/bold] — {len(df)} tickers, "
        "ranked by QoQ share change\n"
    )
    columns = [c for c in _INSTITUTIONAL_COLUMNS if c in df.columns]
    table = Table(show_header=True, header_style="bold", show_lines=False)
    for col_name in columns:
        justify = "left" if col_name == "symbol" else "right"
        table.add_column(
            _INSTITUTIONAL_LABELS.get(col_name, col_name),
            justify=justify,
            no_wrap=True,
        )
    for _, row in df.iterrows():
        table.add_row(*[_format_institutional(c, row[c]) for c in columns])
    console.print(table)
