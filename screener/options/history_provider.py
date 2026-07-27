"""Provider seam for historical options data.

Deep historical intraday option chains are paid-only for the US (Polygon,
ThetaData) and simply unavailable for India at retail (NSE publishes EOD
bhavcopy only). The :class:`OptionsHistoryProvider` protocol lets a paid
adapter drop in later without touching callers; until then the only concrete
backend is :class:`ContractStoreHistoryProvider`, which serves the
forward-capture contract store (:mod:`screener.options.contract_store`).

US intraday history is therefore capture-start-forward unless a paid adapter is
implemented. India is capture-forward plus EOD bhavcopy, full stop — there is
no historical intraday source to buy at retail.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from screener.options import contract_store
from screener.options.models import OptionChain, OptionContract, OptionsMarket


@runtime_checkable
class OptionsHistoryProvider(Protocol):
    """Read historical option chains and per-contract bars for a backtest."""

    def chains(self, underlying: str, day: date) -> list[OptionChain]:
        """Every snapshot chain observed for ``underlying`` on ``day``."""
        ...

    def contract_bars(
        self,
        contract: OptionContract,
        interval: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Per-contract OHLCV/quote bars over ``[start, end]`` at ``interval``."""
        ...


class ContractStoreHistoryProvider:
    """Default backend: the forward-capture contract store.

    Reconstructs :class:`OptionChain` snapshots from the parquet partitions
    written by the recorder. ``contract_bars`` is not implemented — the store
    holds discrete snapshots, not resampled per-contract bars; that reduction
    lands with Phase 4.2 (intraday options backtesting).
    """

    def __init__(self, market: OptionsMarket, *, root: Optional[Path] = None) -> None:
        self.market = market
        self.root = root

    def chains(self, underlying: str, day: date) -> list[OptionChain]:
        frame = contract_store.load_contracts(
            underlying, market=self.market, day=day, root=self.root
        )
        if frame is None or frame.empty:
            return []
        return contract_store.frame_to_chains(frame, market=self.market)

    def contract_bars(
        self,
        contract: OptionContract,
        interval: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "per-contract bars land with Phase 4.2 (intraday options "
            "backtesting); the store currently holds discrete snapshots"
        )


class PolygonOptionsHistoryProvider:
    """Stub adapter for Polygon.io historical options (paid).

    Implement when/if a Polygon subscription is on the table: normalize
    Polygon's option aggregates and chain snapshots into the contract-store
    schema (:data:`screener.options.contract_store.CONTRACT_COLUMNS`) so the
    rest of the pipeline is unchanged.
    """

    def chains(self, underlying: str, day: date) -> list[OptionChain]:
        raise NotImplementedError(
            "Polygon options history requires a paid subscription; not wired up"
        )

    def contract_bars(
        self,
        contract: OptionContract,
        interval: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "Polygon options history requires a paid subscription; not wired up"
        )


class ThetaDataOptionsHistoryProvider:
    """Stub adapter for ThetaData historical options (paid). See Polygon stub."""

    def chains(self, underlying: str, day: date) -> list[OptionChain]:
        raise NotImplementedError(
            "ThetaData options history requires a paid subscription; not wired up"
        )

    def contract_bars(
        self,
        contract: OptionContract,
        interval: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "ThetaData options history requires a paid subscription; not wired up"
        )


def default_history_provider(
    market: OptionsMarket, *, root: Optional[Path] = None
) -> OptionsHistoryProvider:
    """Return the forward-capture store provider (the only real backend today)."""
    return ContractStoreHistoryProvider(market, root=root)


__all__ = [
    "ContractStoreHistoryProvider",
    "OptionsHistoryProvider",
    "PolygonOptionsHistoryProvider",
    "ThetaDataOptionsHistoryProvider",
    "default_history_provider",
]
