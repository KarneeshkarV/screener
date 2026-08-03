"""Typed enrichment selection and diagnostics for unusual-volume scans."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Enrichment(str, Enum):
    BUILDUP = "buildup"
    DEEP_INDIA = "deep_india"
    OPTION_CHAIN = "option_chain"
    FII_DII = "fii_dii"
    PLEDGE = "pledge"


MICROSTRUCTURE_ENRICHMENTS = frozenset(
    {Enrichment.OPTION_CHAIN, Enrichment.FII_DII, Enrichment.PLEDGE}
)


class EnrichmentDiagnostic(BaseModel):
    enrichment: Enrichment
    status: Literal["applied", "skipped", "failed"]
    message: str

    model_config = ConfigDict(frozen=True)


__all__ = [
    "MICROSTRUCTURE_ENRICHMENTS",
    "Enrichment",
    "EnrichmentDiagnostic",
]
