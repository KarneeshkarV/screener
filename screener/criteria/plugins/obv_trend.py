"""obv_trend pipeline criterion — ``screen -c obv-trend``.

On-Balance Volume crossing its EMA. Sweep winner on India Nifty50 was
``ema_window=20`` (no time stop — position runs until OBV crosses back below
its EMA). Use ``screen -c obv-trend -m india`` for the configured combo.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from screener.criteria import criterion


@criterion("obv-trend", pipeline=True)
def obv_trend_pipeline(
    *,
    market: str,
    limit: int,
    **_: Any,
) -> None:
    from screener.commands.live_strategies import run_obv_trend_live

    run_obv_trend_live(
        market=market,
        as_of=date.today(),
        limit=limit,
    )
