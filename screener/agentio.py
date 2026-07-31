"""Agent-mode output: a bounded stdout digest plus a full-data spill file.

Rich tables are written for humans at a terminal. When an AI agent runs the
CLI the same output is actively hostile: rich falls back to 80 columns off a
TTY, so every wide field is ellipsis-truncated -- the agent pays full token
price for data it cannot read. A three-ticker one-year ``backtest-rolling``
costs 113 lines / 11.5 KB that way.

Agent mode replaces that with a digest whose size does not grow with the
result set, and spills the full rows to a CSV the agent reads only if it
needs them.

Activation resolves in three tiers, first match wins:

1. ``--agent`` / ``--no-agent`` on the CLI.
2. ``SCREENER_AGENT`` in the environment (``1``/``0``).
3. Autodetection from the harness env vars in :data:`_AGENT_ENV_VARS`.

Tier 3 deliberately never keys off ``stdout.isatty()``. Click's ``CliRunner``
is always non-TTY, and the suite drives the CLI through it in 121 places --
a TTY heuristic would silently flip every one of those into agent mode.
``PYTEST_CURRENT_TEST`` guards tier 3 for the same reason from the other
side: ``just test`` run *from* an agent harness inherits ``CLAUDECODE=1``,
which would make the suite behave differently locally than in CI.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rich.console import Console
from rich.table import Table


#: Detail levels, cheapest first. ``summary`` prints headline facts and the
#: spill path only; ``head`` adds a header row plus the first
#: :data:`HEAD_ROWS` rows; ``full`` prints every row and still spills.
AgentDetail = Literal["summary", "head", "full"]

DETAIL_LEVELS: tuple[AgentDetail, ...] = ("summary", "head", "full")

#: Default detail level.
#:
#: ``head`` rather than the cheaper ``summary``, and emphatically not
#: ``full``. Measured against Haiku on graded backtest questions: ``full``
#: never opened the spill CSV, because an inline ledger *looks* complete --
#: but it carries no ``pnl`` column, so the model answered a portfolio PnL
#: question with -37.88 against a true -17,480.36, confidently and at 4x the
#: bytes. ``head``'s explicit "... N more rows" marker reads as incomplete
#: and prompts the model to go compute over the CSV instead of guessing.
DEFAULT_DETAIL: AgentDetail = "head"

#: Rows shown inline at ``head`` detail.
HEAD_ROWS = 5

#: Harness env vars that imply an agent is driving the CLI.
#:
#: Verified by dumping ``env`` from inside each harness on this machine,
#: except where noted. Do not add a var without checking that the harness
#: *sets* it rather than merely reading it -- ``CODEX_COMPANION_SESSION_ID``
#: looked like a codex marker but is exported by the companion plugin into
#: the surrounding shell, so it would fire outside codex entirely.
_AGENT_ENV_VARS = (
    "CLAUDECODE",  # Claude Code
    "CLAUDE_CODE_ENTRYPOINT",  # Claude Code
    "CODEX_THREAD_ID",  # codex, set per session
    "OPENCODE",  # opencode
    "OPENCODE_PID",  # opencode
    "PI_CODING_AGENT",  # pi, set at entrypoint (TUI and RPC)
    "CURSOR_AGENT",  # cursor-agent (unverified)
    "AIDER_CHAT",  # aider (unverified)
    "AI_AGENT",  # generic opt-in marker
)

#: Width used for agent output. Wide enough that no realistic column is
#: ellipsized, which is the whole point -- truncated cells cost tokens and
#: carry no information.
_AGENT_WIDTH = 200


@dataclass
class _State:
    """Explicit overrides captured from the CLI, before env resolution."""

    enabled: bool | None = None
    detail: AgentDetail | None = None


_state = _State()


def configure(enabled: bool | None = None, detail: AgentDetail | None = None) -> None:
    """Record the explicit ``--agent`` / ``--agent-detail`` choices.

    ``None`` means "not specified", which leaves the lower tiers in charge.
    """
    _state.enabled = enabled
    _state.detail = detail


def reset() -> None:
    """Drop explicit overrides. Used by tests to restore env-only resolution."""
    _state.enabled = None
    _state.detail = None


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def is_agent_mode() -> bool:
    """Resolve agent mode through the flag / env / autodetect tiers."""
    if _state.enabled is not None:
        return _state.enabled
    from_env = _env_flag("SCREENER_AGENT")
    if from_env is not None:
        return from_env
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return any(os.environ.get(name) for name in _AGENT_ENV_VARS)


def detail_level() -> AgentDetail:
    """Resolve the digest detail level (flag, then env, then the default)."""
    if _state.detail is not None:
        return _state.detail
    raw = (os.environ.get("SCREENER_AGENT_DETAIL") or "").strip().lower()
    for level in DETAIL_LEVELS:
        if raw == level:
            return level
    return DEFAULT_DETAIL


def spill_dir() -> Path:
    """Directory for full-data spill files (``SCREENER_AGENT_DIR`` or ``~/tmp``)."""
    override = os.environ.get("SCREENER_AGENT_DIR")
    return Path(override).expanduser() if override else Path.home() / "tmp"


#: Flags that change only how the digest is *displayed*, never what the
#: spill file contains. Excluded from the run key so that re-running one
#: backtest at three detail levels reuses a single CSV.
_PRESENTATION_FLAGS = ("--agent-detail", "--agent", "--no-agent")


def run_key() -> str:
    """Short stable hash of the invocation, so re-runs overwrite in place.

    Derived from the resolved argv rather than a timestamp: running the same
    backtest twice should reuse one file instead of littering the spill dir.
    """
    argv: list[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in _PRESENTATION_FLAGS:
            skip_next = token == "--agent-detail"
            continue
        if token.startswith("--agent-detail="):
            continue
        argv.append(token)
    return hashlib.blake2b(" ".join(argv).encode(), digest_size=3).hexdigest()


def spill(df: pd.DataFrame, slug: str) -> Path:
    """Write ``df`` to the spill dir and return the path.

    Failures are non-fatal: a digest that cannot cite a file is still far
    better than dumping the whole table into the agent's context.
    """
    target = spill_dir() / f"{slug}-{run_key()}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def get_console() -> Console:
    """Console for result output -- plain and wide in agent mode."""
    if not is_agent_mode():
        return Console()
    return Console(
        width=_AGENT_WIDTH,
        no_color=True,
        highlight=False,
        emoji=False,
        soft_wrap=False,
    )


def _table_rows(table: Table) -> list[list[str]]:
    """Extract a table's already-formatted cells as plain strings."""
    columns = [list(column.cells) for column in table.columns]
    height = max((len(cells) for cells in columns), default=0)
    return [
        [str(cells[index]) if index < len(cells) else "" for cells in columns]
        for index in range(height)
    ]


def render_table(
    table: Table,
    console: Console,
    *,
    detail: AgentDetail | None = None,
    spill_path: Path | None = None,
) -> None:
    """Print ``table`` borderless and row-capped -- the generic agent floor.

    Any command that renders a rich ``Table`` gets bounded output from this
    without command-specific work; hand-tuned digests are a quality upgrade
    on top, never a prerequisite.
    """
    level = detail or detail_level()
    headers = [str(column.header) for column in table.columns]
    rows = _table_rows(table)
    limit = len(rows) if level == "full" else (0 if level == "summary" else HEAD_ROWS)

    title = str(table.title) if table.title else ""
    shown = rows[:limit]
    if title:
        console.print(f"{title}: {len(rows)} rows" if rows else title)

    if shown:
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in shown))
            for i in range(len(headers))
        ]
        console.print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        for row in shown:
            console.print(
                "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            )

    hidden = len(rows) - len(shown)
    if spill_path is not None:
        suffix = f" ({hidden} more rows)" if hidden else ""
        console.print(f"full: {spill_path}{suffix}")
    elif hidden:
        console.print(f"... {hidden} more rows")


def kv_line(pairs: list[tuple[str, Any]], per_line: int = 4) -> list[str]:
    """Format ``key=value`` pairs into compact lines.

    A metrics table costs one line per metric plus box drawing; the same
    facts packed four to a line cost a quarter of that and read no worse.
    """
    chunks = [pairs[i : i + per_line] for i in range(0, len(pairs), per_line)]
    return [" ".join(f"{key}={value}" for key, value in chunk) for chunk in chunks]


__all__ = [
    "AgentDetail",
    "DEFAULT_DETAIL",
    "DETAIL_LEVELS",
    "HEAD_ROWS",
    "configure",
    "detail_level",
    "get_console",
    "is_agent_mode",
    "kv_line",
    "render_table",
    "reset",
    "run_key",
    "spill",
    "spill_dir",
]
