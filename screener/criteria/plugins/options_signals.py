"""Panel-backed options signal pipeline criteria."""

from __future__ import annotations

from typing import Any

from screener.criteria import criterion
from screener.options.criteria import OPTIONS_CRITERIA, run_options_criterion


def _pipeline_runner(name: str):
    def run(*, market: str, limit: int, output_csv: bool, **_: Any) -> None:
        run_options_criterion(
            name,
            market=market,
            limit=limit,
            output_csv=output_csv,
        )

    run.__name__ = name
    return run


for _name in OPTIONS_CRITERIA:
    globals()[_name] = criterion(_name, pipeline=True)(_pipeline_runner(_name))
