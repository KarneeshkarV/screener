"""Panel-backed options signal ``screen -c`` aliases."""

from __future__ import annotations

from typing import Any

from screener.options.criteria import OPTIONS_CRITERIA, run_options_criterion


def _pipeline_runner(name: str):
    def run(*, market: str, limit: int, output_csv: bool = False, **_: Any) -> None:
        run_options_criterion(
            name,
            market=market,
            limit=limit,
            output_csv=output_csv,
        )

    run.__name__ = name
    return run


OPTIONS_SIGNAL_ALIASES = {_name: _pipeline_runner(_name) for _name in OPTIONS_CRITERIA}
globals().update(OPTIONS_SIGNAL_ALIASES)
