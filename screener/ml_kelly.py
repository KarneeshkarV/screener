"""Kelly Criterion position sizing based on ML confidence and regime.

Fractional Kelly with safety caps for practical trading.
"""
from __future__ import annotations

import numpy as np


def kelly_size(
    win_prob: float,
    avg_win: float = 0.10,   # 10% average winner
    avg_loss: float = 0.05,  # 5% average loser
    fraction: float = 0.25,  # Quarter-Kelly for safety
    min_size: float = 0.5,   # Minimum position multiplier
    max_size: float = 2.0,   # Maximum position multiplier
) -> float:
    """Return position-size multiplier (1.0 = baseline).

    K = (p*b - q) / b, where p = win prob, q = 1-p, b = avg_win/avg_loss
    """
    # Edge case handling
    if not np.isfinite(win_prob) or win_prob <= 0.5:
        return min_size

    b = avg_win / avg_loss if avg_loss > 0 else 2.0
    q = 1.0 - win_prob
    kelly = (win_prob * b - q) / b if b > 0 else 0.0

    if kelly <= 0:
        return min_size

    # Fractional Kelly with bounds
    size = 1.0 + (kelly * fraction)
    return float(np.clip(size, min_size, max_size))


def confidence_to_size(
    confidence: float,
    regime_stress: float = 0.0,
    fraction: float = 0.25,
    min_size: float = 0.5,
    max_size: float = 2.0,
) -> float:
    """Map ML confidence (0-1) to position size, with regime stress dampening.

    Higher stress = smaller positions even for high-confidence signals.
    """
    # Stress multiplier: 0 stress -> 1.0, high stress -> 0.5
    stress_mult = 1.0 - (regime_stress * 0.5)
    stress_mult = max(0.5, stress_mult)

    size = kelly_size(
        win_prob=confidence,
        avg_win=0.10,
        avg_loss=0.05,
        fraction=fraction,
        min_size=min_size,
        max_size=max_size,
    )

    return size * stress_mult


def allocate_capital(
    confidence_scores: list[float],
    total_capital: float,
    base_position_pct: float = 0.10,  # 10% per position baseline
    min_size: float = 0.5,
    max_size: float = 2.0,
    max_total_exposure: float = 1.5,  # Cap at 150% of capital
) -> list[float]:
    """Return dollar amounts for each position based on confidence-weighted sizing.

    Args:
        confidence_scores: List of ML confidence values (0-1)
        total_capital: Total portfolio capital
        base_position_pct: Baseline position size as fraction of capital
    """
    if not confidence_scores:
        return []

    sizes = [kelly_size(c, fraction=0.25, min_size=min_size, max_size=max_size)
             for c in confidence_scores]

    # Normalize so total exposure doesn't exceed cap
    raw_total = sum(sizes)
    if raw_total > max_total_exposure:
        scale = max_total_exposure / raw_total
        sizes = [s * scale for s in sizes]

    base_dollar = total_capital * base_position_pct
    allocations = [base_dollar * s for s in sizes]
    return allocations
