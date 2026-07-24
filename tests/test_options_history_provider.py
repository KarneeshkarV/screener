from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from screener.options import contract_store
from screener.options.history_provider import (
    ContractStoreHistoryProvider,
    OptionsHistoryProvider,
    PolygonOptionsHistoryProvider,
    ThetaDataOptionsHistoryProvider,
    default_history_provider,
)
from screener.options.models import OptionChain, OptionContract

AS_OF = datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc)


def _contract() -> OptionContract:
    return OptionContract(
        symbol="SPYC100",
        underlying="SPY",
        expiry=date(2026, 7, 31),
        strike=100.0,
        right="call",
        oi=100.0,
        volume=10.0,
        iv=0.25,
        bid=4.0,
        ask=6.0,
        last=5.0,
        lot_size=10.0,
        as_of=AS_OF,
        source="stub",
    )


def _chain() -> OptionChain:
    return OptionChain(
        underlying="SPY",
        market="us",
        spot=100.0,
        as_of=AS_OF,
        source="stub",
        contracts=(_contract(),),
    )


def test_store_backed_provider_reconstructs_chains(tmp_path: Path):
    contract_store.append_snapshot(_chain(), market="us", root=tmp_path, enrich=False)
    provider = default_history_provider("us", root=tmp_path)
    assert isinstance(provider, OptionsHistoryProvider)
    chains = provider.chains("SPY", date(2026, 7, 10))
    assert len(chains) == 1
    assert chains[0].underlying == "SPY"
    assert chains[0].contracts[0].strike == 100.0


def test_store_backed_provider_empty_day(tmp_path: Path):
    provider = ContractStoreHistoryProvider("us", root=tmp_path)
    assert provider.chains("SPY", date(2026, 7, 10)) == []


def test_store_backed_contract_bars_not_implemented(tmp_path: Path):
    provider = ContractStoreHistoryProvider("us", root=tmp_path)
    with pytest.raises(NotImplementedError):
        provider.contract_bars(_contract(), "1m", date(2026, 7, 1), date(2026, 7, 10))


@pytest.mark.parametrize(
    "provider",
    [PolygonOptionsHistoryProvider(), ThetaDataOptionsHistoryProvider()],
)
def test_paid_stubs_raise(provider):
    with pytest.raises(NotImplementedError):
        provider.chains("SPY", date(2026, 7, 10))
    with pytest.raises(NotImplementedError):
        provider.contract_bars(_contract(), "1m", date(2026, 7, 1), date(2026, 7, 10))
