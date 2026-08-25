from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rich.console import Console, JustifyMethod
from rich.table import Table

from screener.format import (
    fmt_float,
    fmt_mcap,
    fmt_mcap_for_market,
    fmt_pct,
    fmt_volume,
    is_missing,
)

console = Console()


@dataclass(frozen=True)
class ColumnSpec:
    """Per-column display spec: header label, alignment, and value formatter."""

    label: str
    align: JustifyMethod
    formatter: Callable[[Any], str]


def _default(val: Any) -> str:
    return str(val)


# Single source of truth fusing the former COLUMN_LABELS / RIGHT_ALIGN /
# _format_value per-column tables. ``change`` uses fmt_pct, ``volume`` routes
# through the shared compact volume formatter (B/M/K tiers, ``-`` for missing),
# ``market_cap_basic`` / ``sales`` use US million/billion by default; India
# screens swap in lakh/crore via ``fmt_mcap_for_market``. Every other numeric
# column is a fixed two-decimal float.
COLUMNS: dict[str, ColumnSpec] = {
    "ticker": ColumnSpec("Ticker", "left", _default),
    "name": ColumnSpec("Symbol", "left", _default),
    "description": ColumnSpec("Name", "left", _default),
    "close": ColumnSpec("Close", "right", fmt_float),
    "change": ColumnSpec("Chg%", "right", fmt_pct),
    "volume": ColumnSpec("Volume", "right", fmt_volume),
    "market_cap_basic": ColumnSpec("Mkt Cap", "right", fmt_mcap),
    "setup_score": ColumnSpec("Score", "right", fmt_float),
    "EMA5": ColumnSpec("EMA5", "right", fmt_float),
    "EMA20": ColumnSpec("EMA20", "right", fmt_float),
    "EMA100": ColumnSpec("EMA100", "right", fmt_float),
    "EMA200": ColumnSpec("EMA200", "right", fmt_float),
    "price_earnings_ttm": ColumnSpec("P/E", "right", fmt_float),
    "return_on_equity": ColumnSpec("ROE%", "right", fmt_float),
    "dividend_yield_recent": ColumnSpec("Div%", "right", fmt_float),
    "debt_to_equity": ColumnSpec("D/E", "right", fmt_float),
    "RSI": ColumnSpec("RSI", "right", fmt_float),
    "P/E": ColumnSpec("P/E", "right", fmt_float),
    "ROCE%": ColumnSpec("ROCE%", "right", fmt_float),
    "ROE%": ColumnSpec("ROE%", "right", fmt_float),
    "garp_score": ColumnSpec("GARP", "right", fmt_float),
    "peg": ColumnSpec("PEG", "right", fmt_float),
    "sales": ColumnSpec("Sales", "right", fmt_mcap),
    "sales_growth_5y": ColumnSpec("Sales 5Y%", "right", fmt_float),
    "operating_profit_growth": ColumnSpec("OP Growth%", "right", fmt_float),
    "eps_growth_5y": ColumnSpec("EPS 5Y%", "right", fmt_float),
    "roe_5y": ColumnSpec("ROE 5Y%", "right", fmt_float),
    "roce_or_roic": ColumnSpec("ROCE/ROIC", "right", fmt_float),
    "quarterly_profit_growth": ColumnSpec("Q Profit%", "right", fmt_float),
    "days_to_earnings": ColumnSpec("Days to Earn", "right", fmt_float),
}


def _aux_label(aux_column: str) -> str:
    """Turn an aux column name into a table header (``mom_12_1`` becomes ``Mom 12-1``).

    Alpha segments are title-cased. A run of numeric segments joins with ``-``.
    """
    words: list[str] = []
    digits: list[str] = []
    for part in aux_column.split("_"):
        if part.isdigit():
            digits.append(part)
            continue
        if digits:
            words.append("-".join(digits))
            digits.clear()
        words.append(part.capitalize())
    if digits:
        words.append("-".join(digits))
    return " ".join(words)


def _register_aux_columns() -> None:
    """Register a formatted display spec for every declared price-score aux column.

    Without this, a detail-only column such as ``mom_12_1`` falls through
    ``_format_value`` to raw ``str`` and prints left-aligned under its own
    column name. The import is lazy so ``screener.factors`` does not depend
    on this module.
    """
    from screener import factors

    for _, spec in factors.registry.items():
        aux = spec.aux_column
        if aux is not None and aux not in COLUMNS:
            COLUMNS[aux] = ColumnSpec(_aux_label(aux), "right", fmt_float)


_register_aux_columns()

#: Backward-compatible label map (e.g. ``commands.screen_report`` lookups).
COLUMN_LABELS = {col: spec.label for col, spec in COLUMNS.items()}

_MCAP_COLUMNS = frozenset({"market_cap_basic", "sales"})

#: Backward-compatible right-align set for the screen tables.
RIGHT_ALIGN = {col for col, spec in COLUMNS.items() if spec.align == "right"}


def _format_value(col: str, val, market: str = "") -> str:
    if col in _MCAP_COLUMNS:
        return fmt_mcap_for_market(val, market)
    spec = COLUMNS.get(col)
    if spec is not None:
        return spec.formatter(val)
    if is_missing(val):
        return "-"
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
        justify: JustifyMethod = "right" if col_name in RIGHT_ALIGN else "left"
        if col_name == "name":
            table.add_column(label, justify=justify, min_width=8, no_wrap=True)
        elif col_name == "description":
            table.add_column(label, justify=justify, min_width=12, max_width=20)
        else:
            table.add_column(label, justify=justify, no_wrap=True)

    for _, row in df.iterrows():
        cells = [
            _format_value(col_name, row[col_name], market) for col_name in display_cols
        ]
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
        justify: JustifyMethod = "right" if col_name in RIGHT_ALIGN else "left"
        if col_name == "description":
            table.add_column(label, justify=justify, min_width=12, max_width=22)
        elif col_name == "name":
            table.add_column(label, justify=justify, min_width=8, no_wrap=True)
        else:
            table.add_column(label, justify=justify, no_wrap=True)
    for _, row in df.iterrows():
        table.add_row(*[_format_value(col, row[col], market) for col in columns])
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
    if is_missing(val):
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
        justify: JustifyMethod = (
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
    if is_missing(val):
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
        justify: JustifyMethod = "left" if col_name == "symbol" else "right"
        table.add_column(
            _INSTITUTIONAL_LABELS.get(col_name, col_name),
            justify=justify,
            no_wrap=True,
        )
    for _, row in df.iterrows():
        table.add_row(*[_format_institutional(c, row[c]) for c in columns])
    console.print(table)
