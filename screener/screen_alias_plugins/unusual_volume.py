"""Unusual-volume ``screen -c`` compatibility alias."""

from __future__ import annotations

from datetime import date


def unusual_volume_pipeline(
    *,
    market: str,
    limit: int,
    output_csv: bool = False,
    refresh: bool = False,
    cache_ttl: str = "15m",
) -> None:
    """Run the alias; ignores --csv/--cache-ttl."""
    import click

    if output_csv or cache_ttl != "15m":
        click.echo("unusual-volume ignores --csv/--cache-ttl", err=True)
    from screener.unusual_volume.cli import _resolve_universe, run_unusual_volume
    from screener.unusual_volume.service import UnusualVolumeRequest

    request = UnusualVolumeRequest(
        market=market,
        as_of=date.today(),
        universe=_resolve_universe(market, None, None),
        refresh=refresh,
    )
    run_unusual_volume(request, limit=limit)
