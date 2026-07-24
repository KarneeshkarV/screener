"""Click commands for inspecting and pruning the on-disk cache directories.

The price cache (``~/.screener/prices``, ``~/.screener/fmp_prices``) and the
panel snapshots appended by :func:`screener.cache.append_panel_snapshot` grow
without bound; ``screener cache status`` shows what is on disk and
``screener cache clean`` prunes files older than a cutoff. Cleaning is
restricted to the known cache directories discovered from the codebase.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping

import click
from rich.console import Console
from rich.table import Table

# The two stores that grow every session; each has an optional size budget (MB)
# read from an env var. Unset ⇒ no budget (watch disabled for that store).
_WATCHED_STORES: dict[str, str] = {
    "bars": "SCREENER_BARS_BUDGET_MB",
    "contracts": "SCREENER_CONTRACTS_BUDGET_MB",
}


def known_cache_dirs() -> dict[str, Path]:
    """Name -> directory for every on-disk cache the codebase uses.

    Resolved through the cache ownership Module so tests and feature Modules
    use one Interface for cache path Locality.
    """
    from screener.cache import known_cache_dirs as cache_known_cache_dirs

    return cache_known_cache_dirs()


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


def _dir_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in _iter_files(root))


def _budget_bytes_from_env(env_var: str) -> int | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    try:
        mb = float(raw)
    except ValueError:
        return None
    return int(mb * 1024 * 1024) if mb > 0 else None


@dataclass(frozen=True)
class StorageStatus:
    """On-disk size of a watched store against its optional budget."""

    name: str
    path: Path
    size_bytes: int
    budget_bytes: int | None

    @property
    def over_budget(self) -> bool:
        return self.budget_bytes is not None and self.size_bytes > self.budget_bytes

    def summary(self) -> str:
        used = _human_size(self.size_bytes)
        if self.budget_bytes is None:
            return f"{self.name}: {used} (no budget)"
        budget = _human_size(self.budget_bytes)
        pct = (self.size_bytes / self.budget_bytes * 100) if self.budget_bytes else 0.0
        state = "OVER" if self.over_budget else "ok"
        return f"{self.name}: {used} / {budget} ({pct:.0f}%, {state})"


def storage_status(
    budgets_mb: Mapping[str, float | None] | None = None,
) -> list[StorageStatus]:
    """Size of each watched store vs. its budget (from ``budgets_mb`` or env).

    ``budgets_mb`` overrides the env-var budgets per store (a value of ``None``
    disables the budget for that store); omit it to read the env entirely.
    """
    dirs = known_cache_dirs()
    out: list[StorageStatus] = []
    for name, env_var in _WATCHED_STORES.items():
        root = dirs.get(name)
        if root is None:
            continue
        if budgets_mb is not None and name in budgets_mb:
            mb = budgets_mb[name]
            budget = int(mb * 1024 * 1024) if mb and mb > 0 else None
        else:
            budget = _budget_bytes_from_env(env_var)
        out.append(StorageStatus(name, root, _dir_bytes(root), budget))
    return out


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
    console = Console()
    console.print(table)
    _print_contract_store_health(console)
    _print_storage_watch(console)


def _print_storage_watch(console: Console) -> None:
    """Surface any watched store that has exceeded its configured size budget."""
    over = [status for status in storage_status() if status.over_budget]
    if not over:
        return
    console.print("Storage watch:")
    for status in over:
        console.print(f"  [red]over budget[/red] {status.summary()}")


def _print_contract_store_health(console: Console) -> None:
    """Report last-snapshot age + gaps for the options contract store."""
    from screener.options.contract_store import store_health

    lines: list[str] = []
    for market in ("us", "india"):
        health = store_health(market)
        if health.last_snapshot is None:
            continue
        marker = "[yellow]stale[/yellow]" if health.is_stale else "[green]fresh[/green]"
        lines.append(f"{marker} {health.summary()}")
    if lines:
        console.print("Contract store:")
        for line in lines:
            console.print(f"  {line}")


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


@cache_group.command(name="storage-watch")
@click.option(
    "--bars-budget-mb",
    type=float,
    default=None,
    help="Size budget (MB) for the bar store; overrides SCREENER_BARS_BUDGET_MB.",
)
@click.option(
    "--contracts-budget-mb",
    type=float,
    default=None,
    help="Size budget (MB) for the contract store; overrides "
    "SCREENER_CONTRACTS_BUDGET_MB.",
)
def cache_storage_watch(
    bars_budget_mb: float | None, contracts_budget_mb: float | None
) -> None:
    """Report watched-store sizes against budgets; exit non-zero if any is over.

    Cron wrappers call this so a runaway bar/contract store surfaces in the log
    (and the non-zero exit flags it). Budgets come from ``--*-budget-mb`` or the
    ``SCREENER_BARS_BUDGET_MB`` / ``SCREENER_CONTRACTS_BUDGET_MB`` env vars;
    stores without a budget are reported but never fail the command.
    """
    overrides: dict[str, float | None] = {}
    if bars_budget_mb is not None:
        overrides["bars"] = bars_budget_mb
    if contracts_budget_mb is not None:
        overrides["contracts"] = contracts_budget_mb
    statuses = storage_status(overrides or None)
    for status in statuses:
        click.echo(status.summary())
    breached = [status for status in statuses if status.over_budget]
    if breached:
        names = ", ".join(status.name for status in breached)
        raise click.ClickException(f"storage budget exceeded: {names}")
