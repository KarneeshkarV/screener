"""Click commands for inspecting and pruning the on-disk cache directories.

The price cache (``~/.screener/prices``, ``~/.screener/fmp_prices``) and the
panel snapshots appended by :func:`screener.cache.append_panel_snapshot` grow
without bound; ``screener cache status`` shows what is on disk and
``screener cache clean`` prunes files older than a cutoff. Cleaning is
restricted to the known cache directories discovered from the codebase.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import click
from rich.console import Console
from rich.table import Table


def known_cache_dirs() -> dict[str, Path]:
    """Name -> directory for every on-disk cache the codebase uses.

    Resolved lazily from the owning modules so test monkeypatching of the
    module-level constants is respected.
    """
    from screener import cache as _cache
    from screener import universes as _universes
    from screener.backtester import data as _data
    from screener.operator import fetch as _operator_fetch
    from screener.unusual_volume import delivery as _delivery

    return {
        "prices": _data.CACHE_DIR,
        "fmp_prices": _data.FMP_CACHE_DIR,
        "universes": _universes.CACHE_DIR,
        "scanner": _cache.CACHE_ROOT,
        "panels": _cache.PANEL_ROOT,
        "bhavcopy": _delivery.CACHE_DIR,
        "nse_bhavcopy": _operator_fetch.CACHE_ROOT,
    }


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield regular files under ``root``, never escaping it via symlinks."""
    if not root.is_dir():
        return
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            inside = path.resolve().is_relative_to(resolved_root)
        except OSError:
            continue
        if inside:
            yield path


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _resolve_dirs(dir_name: str | None) -> dict[str, Path]:
    dirs = known_cache_dirs()
    if dir_name is None:
        return dirs
    if dir_name not in dirs:
        known = ", ".join(sorted(dirs))
        raise click.BadParameter(
            f"unknown cache dir {dir_name!r}; known dirs: {known}",
            param_hint="--dir",
        )
    return {dir_name: dirs[dir_name]}


@click.group(name="cache")
def cache_group() -> None:
    """Inspect and prune the screener's on-disk caches."""


@cache_group.command(name="status")
def cache_status() -> None:
    """Show file count, size and age for each known cache directory."""
    table = Table(title="Cache status")
    table.add_column("Name")
    table.add_column("Directory")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Oldest")
    table.add_column("Newest")
    for name, root in known_cache_dirs().items():
        stats = [path.stat() for path in _iter_files(root)]
        if stats:
            mtimes = [st.st_mtime for st in stats]
            table.add_row(
                name,
                str(root),
                str(len(stats)),
                _human_size(sum(st.st_size for st in stats)),
                _format_mtime(min(mtimes)),
                _format_mtime(max(mtimes)),
            )
        else:
            table.add_row(name, str(root), "0", "0 B", "-", "-")
    Console().print(table)


@cache_group.command(name="clean")
@click.option(
    "--older-than",
    "older_than",
    type=click.IntRange(min=0),
    required=True,
    help="Delete cache files whose mtime is older than this many days.",
)
@click.option(
    "--dir",
    "dir_name",
    default=None,
    help="Restrict cleaning to one named cache dir (see `cache status`). "
    "Default: all known cache dirs.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Only print what would be removed; delete nothing.",
)
def cache_clean(older_than: int, dir_name: str | None, dry_run: bool) -> None:
    """Delete cache files older than --older-than days."""
    dirs = _resolve_dirs(dir_name)
    cutoff = time.time() - older_than * 86400
    verb = "Would remove" if dry_run else "Removed"
    removed = 0
    reclaimed = 0
    for name, root in dirs.items():
        for path in _iter_files(root):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime >= cutoff:
                continue
            if not dry_run:
                try:
                    path.unlink()
                except OSError as exc:
                    click.echo(f"Failed to remove {path}: {exc}", err=True)
                    continue
            removed += 1
            reclaimed += stat.st_size
            click.echo(f"{verb} [{name}] {path} ({_human_size(stat.st_size)})")
    summary_verb = "Would reclaim" if dry_run else "Reclaimed"
    click.echo(
        f"{summary_verb} {_human_size(reclaimed)} from {removed} file(s) "
        f"older than {older_than} day(s)."
    )
