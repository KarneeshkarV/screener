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
    df: pd.DataFrame, total: int, market: str, criteria_name: str
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


def print_csv(df: pd.DataFrame) -> None:
    print(df.to_csv(index=False))
