"""Market condition checker via EMA penetration analysis."""

from __future__ import annotations

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from tradingview_screener import Query

from screener.cache import parse_ttl
from screener.markets import market_option
from screener.regime import classify_breadth
from screener.scanner import MARKETS, get_scanner_data_cached

# The breadth universe is the top N stocks by market cap, unfiltered. The EMA
# comparison is done here in pandas rather than as a query filter: filtering
# server-side would return the top N stocks *that already pass*, making the
# ratio 100% whenever the market has more than N such stocks.
_UNIVERSE_COLUMNS = ["name", "close", "EMA20", "EMA200", "market_cap_basic"]


def fetch_universe(
    market: str,
    universe_size: int,
    *,
    cache_ttl: float | None,
    refresh: bool,
) -> pd.DataFrame:
    """Fetch the top ``universe_size`` stocks by market cap, unfiltered."""
    query = (
        Query()
        .set_markets(MARKETS[market])
        .select(*_UNIVERSE_COLUMNS)
        .order_by("market_cap_basic", ascending=False)
        .limit(universe_size)
    )

    _, df = get_scanner_data_cached(
        query,
        key_parts=("market_condition", market, universe_size, _UNIVERSE_COLUMNS),
        columns=_UNIVERSE_COLUMNS,
        operation="market condition universe",
        cache_ttl=cache_ttl,
        refresh=refresh,
    )
    return df


def count_above(df: pd.DataFrame, ema_column: str) -> int:
    """Count rows whose close is above ``ema_column``; unknown values do not count."""
    close = pd.to_numeric(df["close"], errors="coerce")
    ema = pd.to_numeric(df[ema_column], errors="coerce")
    return int((close > ema).sum())


# Display text per regime label. The bands themselves live in
# ``screener.regime`` so the live command and the backtester's breadth gate
# classify identically.
_REGIME_DISPLAY: dict[str, tuple[str, str]] = {
    "strong_bull": ("STRONG BULL", "Broad participation across both horizons."),
    "bullish": ("BULLISH", "Majority of the universe is trending up."),
    "long_term_bull_pullback": (
        "LONG-TERM BULL, SHORT-TERM PULLBACK",
        "Primary uptrend intact; short-term breadth has washed out.",
    ),
    "recovery_attempt": (
        "RECOVERY ATTEMPT",
        "Short-term thrust off a weak base; unconfirmed until 200-day breadth follows.",
    ),
    "bearish": ("BEARISH", "Breadth is broken on both horizons."),
    "mixed": ("MIXED", "Breadth is between regimes; no clear signal."),
    "unknown": ("UNKNOWN", "Not enough data to classify."),
}


def classify_regime(ema_20_pct: float, ema_200_pct: float) -> tuple[str, str]:
    """Map the two breadth readings to a display name and its one-line reading."""
    return _REGIME_DISPLAY[classify_breadth(ema_20_pct, ema_200_pct)]


def _signal(pct: float) -> str:
    if pct > 60:
        return "STRONG"
    if pct < 40:
        return "WEAK"
    return "NEUTRAL"


def _print_condition_report(
    market: str,
    universe: int,
    ema_20_count: int,
    ema_20_pct: float,
    ema_200_count: int,
    ema_200_pct: float,
) -> None:
    """Display market condition summary."""
    console = Console()

    condition_20 = _signal(ema_20_pct)
    condition_200 = _signal(ema_200_pct)

    table = Table(title=f"Market Condition Report - {market.upper()}")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_column("Percentage")
    table.add_column("Signal")

    table.add_row(
        "Above 20-day EMA",
        f"{ema_20_count} / {universe}",
        f"{ema_20_pct:.1f}%",
        condition_20,
    )
    table.add_row(
        "Above 200-day EMA",
        f"{ema_200_count} / {universe}",
        f"{ema_200_pct:.1f}%",
        condition_200,
    )

    console.print(table)

    regime, reading = classify_regime(ema_20_pct, ema_200_pct)
    console.print(f"\n[bold]Overall Market Regime: {regime}[/bold]")
    console.print(f"  {reading}\n")
    console.print(
        "  Strong Bull                          200D > 60%  and  20D > 60%\n"
        "  Bullish                              200D > 50%  and  20D > 50%\n"
        "  Long-term Bull, Short-term Pullback  200D > 50%  and  20D < 40%\n"
        "  Recovery Attempt                     200D < 50%  and  20D > 60%\n"
        "  Bearish                              200D < 40%  and  20D < 40%\n"
        "  Mixed                                everything else\n"
    )


@click.command(name="market-condition")
@market_option(default="us", help="Market to analyze.")
@click.option(
    "-n",
    "--universe-size",
    default=500,
    help="Number of top stocks to check (by market cap).",
)
@click.option(
    "--refresh",
    is_flag=True,
    help="Bypass cached TradingView data.",
)
@click.option(
    "--cache-ttl",
    default="15m",
    show_default=True,
    help="TradingView cache TTL, e.g. 30s, 15m, 1h, off.",
)
def market_condition(
    market: str,
    universe_size: int,
    refresh: bool,
    cache_ttl: str,
) -> None:
    """Check market condition via 20-day and 200-day EMA penetration rates.

    Scans the top stocks in a market and counts how many are trading above
    their 20-day and 200-day EMAs. Higher percentages indicate bullish
    conditions; lower percentages suggest caution.
    """
    click.echo(
        f"Scanning top {universe_size} stocks in {market.upper()} market "
        f"for EMA penetration...\n"
    )

    df = fetch_universe(
        market,
        universe_size,
        cache_ttl=parse_ttl(cache_ttl, default=900),
        refresh=refresh,
    )

    # The denominator is the universe actually returned, which can fall short of
    # the requested size on a small market or a degraded fetch.
    universe = len(df)
    if universe == 0:
        raise click.ClickException(
            f"No universe data returned for {market.upper()}; "
            f"rerun with --refresh once connectivity is back."
        )

    ema_20_count = count_above(df, "EMA20")
    ema_200_count = count_above(df, "EMA200")

    ema_20_pct = ema_20_count / universe * 100
    ema_200_pct = ema_200_count / universe * 100

    _print_condition_report(
        market,
        universe,
        ema_20_count,
        ema_20_pct,
        ema_200_count,
        ema_200_pct,
    )


__all__ = ["market_condition"]
