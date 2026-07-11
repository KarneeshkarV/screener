"""CLI command backing up (or restoring) screen-run history via Turso/libSQL."""

from __future__ import annotations

import sys

import click

from screener import history_sync


@click.command(name="history-backup")
@click.option(
    "--restore",
    is_flag=True,
    help="Pull remote runs missing from the local DB instead of pushing.",
)
@click.option(
    "--batch-size",
    type=int,
    default=200,
    show_default=True,
    help="Number of statements per Turso batch round-trip.",
)
def history_backup(restore: bool, batch_size: int) -> None:
    """Mirror local screen-run history to Turso (or restore it with --restore)."""
    client = history_sync.connect()
    if client is None:
        click.echo(
            "Turso is not configured: set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN "
            "(or add them to a .env file) to enable history backup.",
            err=True,
        )
        sys.exit(1)

    try:
        if restore:
            summary = history_sync.restore_history(client)
            click.echo(
                f"Restored {summary.runs_restored} runs "
                f"({summary.rows_restored} rows) from Turso; "
                f"{summary.local_runs} runs now local."
            )
        else:
            backup = history_sync.backup_history(client, batch_size=batch_size)
            click.echo(
                f"Pushed {backup.runs_pushed} runs ({backup.rows_pushed} rows) "
                f"to Turso; remote now holds {backup.remote_runs} runs / "
                f"{backup.remote_rows} rows."
            )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, no traceback
        click.echo(f"history-backup failed: {exc}", err=True)
        sys.exit(1)
    finally:
        client.close()


__all__ = ["history_backup"]
