"""Options snapshot recorder — forward-capture chains into the contract store.

``screener options record`` snapshots delayed CBOE chains (US) and the NSE live
chain API (India) for a watchlist into the first-class contract store
(:mod:`screener.options.contract_store`) during session hours. Run one pass per
invocation from a plain 15-minute cron (``--once``), or run a session-bounded
in-process loop from a single session-open cron (``--every 15m``). Passes are
idempotent, so overlapping runs are safe.

Free intraday option history barely exists, so every session the recorder is
not running is history that can never be recovered — this is why the recorder
ships before the rest of Phase 3/4.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from screener.markets import get_market
from screener.options import contract_store
from screener.options.models import OptionsMarket
from screener.options.provider import OptionsProvider

# Index options are the priority line-item; extend per deployment with
# ``--watchlist`` / ``--watchlist-file`` (chain snapshots at 15-min cadence are
# the storage cost to watch, so the default stays small).
DEFAULT_WATCHLISTS: dict[str, tuple[str, ...]] = {
    "us": ("SPY", "QQQ", "IWM"),
    "india": ("NIFTY", "BANKNIFTY", "FINNIFTY"),
}

# Regular session hours in each market's local timezone, used to gate the loop.
SESSION_HOURS: dict[str, tuple[time, time]] = {
    "us": (time(9, 30), time(16, 0)),
    "india": (time(9, 15), time(15, 30)),
}


@dataclass(frozen=True)
class RecordPassResult:
    """Outcome of one recorder pass over a watchlist."""

    market: str
    recorded: list[tuple[str, int]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def contract_count(self) -> int:
        return sum(count for _, count in self.recorded)


def resolve_watchlist(
    market: str,
    *,
    watchlist: Optional[str] = None,
    watchlist_file: Optional[Path] = None,
    max_underlyings: int = 0,
) -> list[str]:
    """Resolve the underlyings to snapshot, deduped and capped.

    Precedence: explicit ``--watchlist`` > ``--watchlist-file`` > the market's
    small index-options default.
    """
    if watchlist:
        raw = [token.strip() for token in watchlist.split(",")]
    elif watchlist_file is not None:
        text = Path(watchlist_file).read_text()
        raw = [token.strip() for line in text.splitlines() for token in line.split(",")]
    else:
        raw = list(DEFAULT_WATCHLISTS.get(market, ()))
    seen: dict[str, None] = {}
    for token in raw:
        symbol = token.upper()
        if symbol and not symbol.startswith("#"):
            seen.setdefault(symbol, None)
    symbols = list(seen)
    if max_underlyings > 0:
        symbols = symbols[:max_underlyings]
    return symbols


def within_session(market: str, *, now: Optional[datetime] = None) -> bool:
    """True when ``now`` (default: current UTC) is inside the market session."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    local = reference.astimezone(ZoneInfo(get_market(market).timezone))
    if local.weekday() >= 5:  # Saturday/Sunday
        return False
    open_time, close_time = SESSION_HOURS.get(market, (time.min, time.max))
    return open_time <= local.timetz().replace(tzinfo=None) <= close_time


def default_provider(market: OptionsMarket) -> OptionsProvider:
    """Live chain provider for a market (CBOE→yfinance for US, NSE for India)."""
    if market == "us":
        from screener.options.provider import default_us_provider

        return default_us_provider()
    from screener.options.nse_live import NSELiveOptionsProvider

    return NSELiveOptionsProvider()


def run_pass(
    market: OptionsMarket,
    symbols: list[str],
    *,
    provider: OptionsProvider,
    root: Optional[Path] = None,
    refresh: bool = False,
    enrich: bool = True,
) -> RecordPassResult:
    """Snapshot every watchlist underlying once, appending to the store.

    Degrades per symbol: a provider returning ``None`` or raising is recorded
    as missing, never aborting the pass.
    """
    recorded: list[tuple[str, int]] = []
    missing: list[str] = []
    for symbol in symbols:
        try:
            chain = provider.fetch_chain(symbol, market, refresh=refresh)
        except Exception:  # noqa: BLE001 - provider boundary; degrade cleanly
            chain = None
        if chain is None or not chain.contracts:
            missing.append(symbol)
            continue
        contract_store.append_snapshot(chain, market=market, root=root, enrich=enrich)
        recorded.append((symbol, len(chain.contracts)))
    return RecordPassResult(market=market, recorded=recorded, missing=missing)


def record_loop(
    market: OptionsMarket,
    symbols: list[str],
    *,
    provider: OptionsProvider,
    every_seconds: float,
    root: Optional[Path] = None,
    refresh: bool = False,
    echo: Callable[[str], None] = print,
    sleep: Callable[[float], None] = _time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    max_passes: Optional[int] = None,
) -> list[RecordPassResult]:
    """Run session-gated passes at ``every_seconds`` until the session closes.

    Outside session hours each tick is a logged no-op; the loop ends when the
    market is closed and it is past the session (weekend or after close). Tests
    inject ``clock``/``sleep`` and cap ``max_passes`` to run deterministically.
    """
    results: list[RecordPassResult] = []
    passes = 0
    while True:
        now = clock()
        if within_session(market, now=now):
            result = run_pass(
                market, symbols, provider=provider, root=root, refresh=refresh
            )
            results.append(result)
            echo(
                f"[{now:%Y-%m-%d %H:%M}Z] {market}: recorded "
                f"{len(result.recorded)}/{len(symbols)} underlyings, "
                f"{result.contract_count} contracts"
            )
        else:
            echo(f"[{now:%Y-%m-%d %H:%M}Z] {market}: outside session, skipping")
            if _session_over(market, now):
                break
        passes += 1
        if max_passes is not None and passes >= max_passes:
            break
        sleep(every_seconds)
    return results


def _session_over(market: str, now: datetime) -> bool:
    """True once the current day's session has closed (so the loop can stop)."""
    reference = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    local = reference.astimezone(ZoneInfo(get_market(market).timezone))
    if local.weekday() >= 5:
        return True
    _, close_time = SESSION_HOURS.get(market, (time.min, time.max))
    return local.timetz().replace(tzinfo=None) > close_time


__all__ = [
    "DEFAULT_WATCHLISTS",
    "SESSION_HOURS",
    "RecordPassResult",
    "default_provider",
    "record_loop",
    "resolve_watchlist",
    "run_pass",
    "within_session",
]
