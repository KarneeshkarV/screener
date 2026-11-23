"""Offline regressions for iv_sentiment's P/C curve and cheap_earnings_vol.

B15a/B15b: the P/C subscore must be continuous at 1.0 (0.5) and fall to 0.0 by
2.0 instead of jumping back to bullish; a ``None`` median IV from the NSE
provider must fall through to the neutral IV subscore instead of raising
``TypeError``.

B14c: ``cheap_earnings_vol`` must only fire when a known earnings event lands
between ``as_of`` and the front expiry, so a quiet non-earnings weekly cannot
masquerade as cheap earnings volatility.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from screener.earnings_backtest.strategies import iv_sentiment
from screener.options.criteria import screen_options_criterion

AS_OF = date(2026, 7, 6)
FRONT_EXPIRY = date(2026, 7, 17)
EARNINGS_TS = pd.Timestamp("2026-07-10")
# median_iv=35.0 maps to an IV subscore of exactly 0.5.
IV_SUBSCORE_CONTRIBUTION = 0.5 * 0.3


# ── iv_sentiment (B15a / B15b) ───────────────────────────────────────────


def _iv(pc_ratio: float | None, median_iv: float | None = 35.0):
    return iv_sentiment(
        "AAA",
        EARNINGS_TS,
        {"pc_ratio": pc_ratio, "median_iv": median_iv},
    )


def test_pc_subscore_is_monotonic_and_continuous_at_one():
    ratios = [0.5, 0.7, 0.99, 1.0, 1.5, 2.0]
    results = [_iv(ratio) for ratio in ratios]
    scores = [result.score for result in results]

    # Bullish ratios score highest and every further bearish step scores less;
    # no jump back upward at the 1.0 boundary.
    assert scores == sorted(scores, reverse=True)

    neutral = results[ratios.index(1.0)].score
    pc_score_at_one = (neutral - IV_SUBSCORE_CONTRIBUTION) / 0.7
    assert pc_score_at_one == pytest.approx(0.5)
    # 0.99 and 1.00 remain adjacent (only the slope changes at the boundary).
    assert abs(results[ratios.index(0.99)].score - neutral) < 0.02
    for result in results[ratios.index(1.0) :]:
        assert result.score <= 0.5

    # Bearish ratios keep descending: 1.5 -> 0.25 and 2.0 -> 0.0 P/C subscore.
    at_1_5 = (results[ratios.index(1.5)].score - IV_SUBSCORE_CONTRIBUTION) / 0.7
    at_2_0 = (results[ratios.index(2.0)].score - IV_SUBSCORE_CONTRIBUTION) / 0.7
    assert at_1_5 == pytest.approx(0.25)
    assert at_2_0 == pytest.approx(0.0)


def test_none_median_iv_uses_neutral_subscore_without_type_error():
    result = _iv(0.8, median_iv=None)

    assert result.details["median_iv_pct"] is None
    # pc_score(0.8) = 1 - (0.8 - 0.7) / 0.3 * 0.5 = 0.8333...
    expected_pc = 1.0 - (0.8 - 0.7) / 0.3 * 0.5
    assert result.score == pytest.approx(expected_pc * 0.7 + 0.35 * 0.3, abs=1e-3)


def test_non_finite_pc_ratio_is_treated_as_neutral():
    result = _iv(None, median_iv=None)
    assert result.score == pytest.approx(0.5 * 0.7 + 0.35 * 0.3, abs=1e-4)
    assert result.details["pc_ratio"] == 1.0


# ── cheap_earnings_vol (B14c) ────────────────────────────────────────────


def _panel(*, implied: float = 3.0, front_expiry: date = FRONT_EXPIRY):
    return pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp(AS_OF),
                "SYMBOL": "AAA",
                "source": "fixture",
                "implied_move_pct": implied,
                "front_expiry": front_expiry.isoformat(),
            }
        ]
    )


def _history_fetcher(tickers, **_kwargs):
    assert tickers == ["AAA"]
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "earnings_date": pd.to_datetime(["2025-01-10", "2025-04-10", "2025-07-10"]),
        }
    )


def _prices_fixture(tickers, _start, _end):
    assert tickers == ["AAA"]
    index = pd.to_datetime(
        [
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-04-09",
            "2025-04-10",
            "2025-04-11",
            "2025-07-09",
            "2025-07-10",
            "2025-07-11",
        ]
    )
    return {
        "AAA": pd.DataFrame(
            {"close": [100, 102, 105, 100, 104, 108, 100, 101, 106]}, index=index
        )
    }


def _upcoming(mapping: dict[str, date | None]):
    def fetch(tickers, market, *, as_of=None):
        assert market == "us"
        assert as_of == AS_OF
        return {ticker: mapping.get(ticker) for ticker in tickers}

    return fetch


def _screen(*, implied: float = 3.0, upcoming: date | None):
    return screen_options_criterion(
        "cheap_earnings_vol",
        market="us",
        limit=10,
        as_of=AS_OF,
        panel=_panel(implied=implied),
        earnings_fetcher=_history_fetcher,
        price_fetcher=_prices_fixture,
        next_earnings_fetcher=_upcoming({"AAA": upcoming}),
    )


@pytest.mark.parametrize(
    "upcoming",
    [None, date(2026, 7, 18)],
    ids=["no_known_event", "after_front_expiry"],
)
def test_cheap_vol_rejects_expiries_without_upcoming_earnings(upcoming):
    # A ~3% implied move looks "cheap" against the 6% historical median, but
    # no earnings event occurs before the front expiry, so it is not a trade.
    result = _screen(upcoming=upcoming)

    assert result.frame.empty
    assert "earnings event" in result.message


@pytest.mark.parametrize(
    "upcoming",
    [AS_OF, date(2026, 7, 10), FRONT_EXPIRY],
    ids=["on_as_of", "inside_window", "on_front_expiry"],
)
def test_cheap_vol_passes_when_earnings_falls_within_front_expiry(upcoming):
    result = _screen(upcoming=upcoming)

    assert result.frame["SYMBOL"].tolist() == ["AAA"]
    row = result.frame.iloc[0]
    assert row["realized_earnings_move_pct"] == pytest.approx(6.0)
    assert row["vol_edge_pct"] == pytest.approx(3.0)
    assert row["earnings_events"] == 3
