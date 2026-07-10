"""CLI-only aliases that expose full workflows through ``screen -c``.

These commands are not composable TradingView criteria. Keeping them in a
separate, explicitly typed table lets the screen application workflow remain a
single tabular flow while preserving the historical CLI spellings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from screener.screen_alias_plugins.garp import garp_pipeline
from screener.screen_alias_plugins.mark_minervini import mark_minervini_pipeline
from screener.screen_alias_plugins.obv_trend import obv_trend_pipeline
from screener.screen_alias_plugins.promoter_buys import promoter_buys_pipeline
from screener.screen_alias_plugins.rs_breakout import rs_breakout_pipeline
from screener.screen_alias_plugins.unusual_volume import unusual_volume_pipeline
from screener.screen_alias_plugins.vol_breakout import vol_breakout_pipeline


class ScreenAliasFn(Protocol):
    def __call__(
        self,
        *,
        market: str,
        limit: int,
        output_csv: bool,
        refresh: bool,
        cache_ttl: str,
    ) -> None: ...


@dataclass(frozen=True)
class ScreenAliasSelection:
    name: str
    runner: ScreenAliasFn


class ScreenAliasSelectionError(ValueError):
    """Raised when a command alias is combined with another criterion."""


SCREEN_ALIASES: dict[str, ScreenAliasFn] = {
    "garp": garp_pipeline,
    "mark-minervini": mark_minervini_pipeline,
    "obv-trend": obv_trend_pipeline,
    "promoter-buys": promoter_buys_pipeline,
    "rs-breakout": rs_breakout_pipeline,
    "unusual-volume": unusual_volume_pipeline,
    "vol-breakout": vol_breakout_pipeline,
}


def resolve_screen_alias(names: Sequence[str]) -> ScreenAliasSelection | None:
    """Return the selected full-workflow alias, if any."""
    aliases = [name for name in names if name in SCREEN_ALIASES]
    if not aliases:
        return None
    if len(names) != 1:
        raise ScreenAliasSelectionError(
            f"Screen alias {aliases[0]!r} cannot be combined with other -c values; "
            f"got {list(names)!r}."
        )
    name = aliases[0]
    return ScreenAliasSelection(name=name, runner=SCREEN_ALIASES[name])


__all__ = [
    "SCREEN_ALIASES",
    "ScreenAliasFn",
    "ScreenAliasSelection",
    "ScreenAliasSelectionError",
    "resolve_screen_alias",
]
