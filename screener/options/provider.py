"""Provider protocol for normalized options chains."""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol

from screener.options.models import OptionChain, OptionsMarket


class OptionsProvider(Protocol):
    """Fetch a normalized option chain or return ``None`` when unavailable."""

    def fetch_chain(
        self,
        symbol: str,
        market: OptionsMarket,
        expiry: date | None = None,
        *,
        refresh: bool = False,
    ) -> OptionChain | None: ...


LOG = logging.getLogger(__name__)


class FallbackOptionsProvider:
    """Try providers in order, degrading to ``None`` when all are unavailable."""

    def __init__(self, *providers: OptionsProvider) -> None:
        self.providers = tuple(providers)

    def fetch_chain(
        self,
        symbol: str,
        market: OptionsMarket,
        expiry: date | None = None,
        *,
        refresh: bool = False,
    ) -> OptionChain | None:
        for provider in self.providers:
            try:
                chain = provider.fetch_chain(symbol, market, expiry, refresh=refresh)
            except Exception as exc:  # noqa: BLE001 - provider boundary
                LOG.warning(
                    "options provider %s failed for %s: %s",
                    type(provider).__name__,
                    symbol,
                    exc,
                )
                continue
            if chain is not None:
                return chain
        return None


def default_us_provider() -> FallbackOptionsProvider:
    """CBOE delayed quotes first, yfinance fallback."""
    from screener.options.cboe import CboeOptionsProvider
    from screener.options.yf_chain import YFinanceOptionsProvider

    return FallbackOptionsProvider(CboeOptionsProvider(), YFinanceOptionsProvider())


__all__ = ["FallbackOptionsProvider", "OptionsProvider", "default_us_provider"]
