"""Provider protocol for normalized options chains."""

from __future__ import annotations

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


__all__ = ["OptionsProvider"]
