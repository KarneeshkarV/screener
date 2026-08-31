"""The candidate-gate flags, defined once for every command that has them.

A gate decides who is a candidate. The screen and the rolling backtest must
answer that question identically or a screen names names the backtest would
never have entered, so the flags that move the answer live here rather than
being spelled out per command: one name, one help text, one default, one
override rule.

The module is neutral by construction - it imports Click and
:mod:`screener.regime` and nothing else at module scope - so a screen can use
it without pulling in the backtest engine.

``--earnings-buffer`` is deliberately *not* here. It drops finished result rows
whose next earnings date is near, which is a presentation-stage filter on a
screen's output; ``--earnings-blackout`` suppresses the entry signal itself and
therefore changes who is a candidate. They act at different stages and the two
commands are right to differ.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from screener.regime import TREND_LABELS

#: The ``--adv-window`` option default, shared by both Click commands.
ADV_WINDOW_DEFAULT = 20

OptionDecorator = Callable[[Any], Any]

_MIN_PRICE = click.option(
    "--min-price",
    type=float,
    default=None,
    help=(
        "Minimum signal-day close to admit a ticker. Defaults to the market "
        "floor ($1 US / ₹10 India). Pass 0 to disable."
    ),
)

_MIN_AVG_DOLLAR_VOLUME = click.option(
    "--min-avg-dollar-volume",
    type=float,
    default=None,
    help=(
        "Minimum rolling-mean dollar volume (close*volume) over --adv-window. "
        "Defaults to the market floor ($1,000 US / ₹100,000 India). Pass 0 to "
        "disable."
    ),
)

_ADV_WINDOW = click.option(
    "--adv-window",
    type=int,
    default=ADV_WINDOW_DEFAULT,
    help="Lookback bars for the average dollar-volume filter.",
)

_REGIME_FILTER = click.option(
    "--regime-filter",
    "regime_filter_args",
    multiple=True,
    type=click.Choice(list(TREND_LABELS)),
    help=(
        "Only allow entries on days whose benchmark trend regime matches "
        "(repeatable). Warmup days with an unknown regime are suppressed."
    ),
)

_SECTOR_NEUTRAL = click.option(
    "--sector-neutral",
    is_flag=True,
    default=False,
    help=(
        "Z-score rank_score within each sector group per day before ranking "
        "(factor strategies only; no-op when no rank_score column exists)."
    ),
)

_EARNINGS_BLACKOUT = click.option(
    "--earnings-blackout",
    "earnings_blackout_days",
    type=int,
    default=None,
    help=(
        "Suppress entry signals within N calendar days before (and including) "
        "a known earnings date for each ticker. Tickers with no known earnings "
        "dates remain eligible (a warning is recorded)."
    ),
)

_MIN_SCORE = click.option(
    "--min-score",
    type=float,
    default=None,
    help=(
        "Minimum setup_score (0-100) to admit a candidate. The score is a "
        "cross-sectional percentile of the day's eligible field, so this is a "
        "statement about standing in the field, not an absolute bar."
    ),
)

_GATE_OPTIONS: dict[str, OptionDecorator] = {
    "min-price": _MIN_PRICE,
    "min-avg-dollar-volume": _MIN_AVG_DOLLAR_VOLUME,
    "adv-window": _ADV_WINDOW,
    "regime-filter": _REGIME_FILTER,
    "sector-neutral": _SECTOR_NEUTRAL,
    "earnings-blackout": _EARNINGS_BLACKOUT,
    "min-score": _MIN_SCORE,
}

#: Every gate flag, in the order a command should render them.
GATE_OPTION_NAMES = tuple(_GATE_OPTIONS)


def gate_options(*names: str) -> OptionDecorator:
    """Stack the named gate options onto a command, in the order given.

    Passing no names stacks all of them, which is what a command wanting the
    full set should do: it then picks up a gate added here later without any
    edit of its own.
    """
    selected = names or GATE_OPTION_NAMES

    def decorator(command: Any) -> Any:
        for name in reversed(selected):
            command = _GATE_OPTIONS[name](command)
        return command

    return decorator


def gate_overrides(
    *,
    min_price: float | None = None,
    min_avg_dollar_volume: float | None = None,
    adv_window: int = ADV_WINDOW_DEFAULT,
    adv_window_was_explicit: bool = False,
    regime_filter_args: tuple[str, ...] = (),
    earnings_blackout_days: int | None = None,
    sector_neutral: bool = False,
    min_score: float | None = None,
) -> dict[str, Any]:
    """The gates the user actually typed, as
    :func:`~screener.strategies.spec.resolve_strategy_profile` overrides.

    A flag left at its option default is "not given", so it produces no key and
    the strategy's own profile supplies the value. ``adv_window_was_explicit``
    carries Click's parameter source because a typed ``20`` equals the default.
    Anything typed becomes an override and wins. Both commands build their
    overrides here so the same flag cannot mean one thing on a screen and
    another on a backtest.
    """
    overrides: dict[str, Any] = {}
    if min_price is not None:
        overrides["min_price"] = float(min_price)
    if min_avg_dollar_volume is not None:
        overrides["min_avg_dollar_volume"] = float(min_avg_dollar_volume)
    if adv_window_was_explicit or int(adv_window) != ADV_WINDOW_DEFAULT:
        overrides["avg_dollar_volume_window"] = int(adv_window)
    regime_filter = tuple(dict.fromkeys(regime_filter_args))
    if regime_filter:
        overrides["regime_filter"] = regime_filter
    if earnings_blackout_days is not None:
        overrides["earnings_blackout_days"] = int(earnings_blackout_days)
    if sector_neutral:
        overrides["sector_neutral"] = True
    if min_score is not None:
        overrides["min_score"] = float(min_score)
    return overrides
