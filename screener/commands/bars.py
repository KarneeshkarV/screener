"""Click commands for the interval-partitioned bar store.

``screener bars record`` appends the trailing window of 1m bars for a market's
active universe to ``~/.screener/bars/{market}/1m/``. Run once per day from
cron (see ``scripts/daily_bars_record.sh``), it turns the provider's ~30-day
free 1m history into a growing archive; coarser intraday requests
(5m/15m/30m/1h) are then served by resampling the stored 1m series locally
instead of separate per-interval downloads.
"""

from __future__ import annotations

from datetime import date, timedelta

import click


@click.group(name="bars")
def bars_group() -> None:
    """Manage the interval-partitioned on-disk bar store."""


@bars_group.command(name="record")
@click.option(
    "-m",
    "--market",
    type=click.Choice(["us", "india"]),
    default="us",
    show_default=True,
    help="Market whose universe is recorded.",
)
@click.option(
    "--days",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Trailing calendar days of 1m bars to append (provider cap ~30d).",
)
@click.option(
    "--universe",
    "universe_name",
    default=None,
    help="Named universe to record (default: the market's default universe).",
)
@click.option(
    "--tickers",
    default=None,
    help="Comma-separated tickers to record instead of a universe.",
)
@click.option(
    "--max-symbols",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Cap the number of recorded symbols (0 = no cap; handy for smoke runs).",
)
def bars_record(
    market: str,
    days: int,
    universe_name: str | None,
    tickers: str | None,
    max_symbols: int,
) -> None:
    """Append the trailing window of 1m bars for the active universe.

    The fetched window is appended to the bar store, merging with what is
    already archived (overlaps dedupe, so daily runs are idempotent). Run
    daily after the close; each run extends the archive past the ~30-day
    provider cap. To repair one symbol, delete its store file and re-record —
    the recorder never rewrites history wholesale.
    """
    from screener.backtester.bar_store import BARS_ROOT, append_bars
    from screener.backtester.data import build_price_fetcher, tv_to_yf
    from screener.markets import get_market

    if tickers:
        symbols = [t.strip() for t in tickers.split(",") if t.strip()]
        universe_label = "custom tickers"
    else:
        from screener.universes import load_current_universe

        name = universe_name or get_market(market).default_universe
        universe = load_current_universe(name)
        symbols = list(universe.symbols)
        universe_label = name
    if max_symbols:
        symbols = symbols[:max_symbols]
    if not symbols:
        raise click.UsageError(f"no symbols resolved for {universe_label!r}")

    yf_symbols = sorted({tv_to_yf(symbol, market) for symbol in symbols})
    end = date.today()
    start = end - timedelta(days=days)
    click.echo(
        f"Recording 1m bars: {len(yf_symbols)} {market} symbols "
        f"({universe_label}), {start} → {end}"
    )

    fetcher = build_price_fetcher(interval="1m", market=market)
    frames = fetcher.fetch(yf_symbols, start, end)
    with_data = 0
    for symbol in yf_symbols:
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            continue
        with_data += 1
        append_bars(symbol, frame, market=market, interval="1m")
    store_dir = BARS_ROOT / market / "1m"
    click.echo(f"appended bars for {with_data}/{len(yf_symbols)} symbols → {store_dir}")
    if with_data < len(yf_symbols):
        missing = [
            symbol for symbol, frame in frames.items() if frame is None or frame.empty
        ]
        click.echo(
            f"no bars returned for {len(missing)} symbol(s): "
            f"{', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}",
            err=True,
        )
